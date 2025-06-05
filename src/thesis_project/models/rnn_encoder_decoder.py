import random
from typing import Optional
import numpy as np
import torch
from torch import nn
import torch.nn.functional as F

from thesis_project.models import OutputType
from thesis_project.models.output_handler import LogitOutputHandler, OutputHandler

# The code in this file is closely based on https://github.com/Maab-Nimir/Neural-Machine-Translation-by-Jointly-Learning-to-Align-and-Translate


class EncoderAttention(nn.Module):

    INPUT_MODE = "last_hidden"

    def __init__(
        self, emb_dim, enc_hid_dim, dec_hid_dim, dropout, n_layers: int = 1
    ):
        super().__init__()

        self.rnn = nn.GRU(emb_dim,
                          enc_hid_dim,
                          bidirectional=True,
                          num_layers=n_layers,
                          dropout=0.8)
        
        self.fc = nn.Linear(enc_hid_dim * 2, dec_hid_dim)

        self.dropout = nn.Dropout(dropout)

    def forward(self, src):

        # src = [src len, batch size]
        # added by me to adapt to spikerates

        if self.INPUT_MODE == "mean":

            src = torch.mean(src, dim=0).unsqueeze(
                dim=0
            )  # .reshape(([1, src.shape[0], 256]))
            outputs, hidden = self.rnn(src)

        elif self.INPUT_MODE == "last_hidden":
            outputs, hidden = self.rnn(src)
            outputs = outputs[-1, :, :].unsqueeze(dim=0)

        else:
            raise Exception("Unknown input mode")

        # embedded = [src len, batch size, emb dim]

        # outputs = [src len, batch size, hid dim * num directions]
        # hidden = [n layers * num directions, batch size, hid dim]

        # hidden is stacked [forward_1, backward_1, forward_2, backward_2, ...]
        # outputs are always from the last layer

        # hidden [-2, :, : ] is the last of the forwards RNN
        # hidden [-1, :, : ] is the last of the backwards RNN

        # initial decoder hidden is final hidden state of the forwards and backwards
        #  encoder RNNs fed through a linear layer
        hidden = torch.tanh(
            self.fc(torch.cat((hidden[-2, :, :], hidden[-1, :, :]), dim=1))
        )

        # outputs = [src len, batch size, enc hid dim * 2]
        # hidden = [batch size, dec hid dim]

        return outputs, hidden


class Attention(nn.Module):
    def __init__(self, enc_hid_dim, dec_hid_dim):
        super().__init__()
        
        self.attn = nn.Linear((enc_hid_dim * 2) + dec_hid_dim, dec_hid_dim)
        self.v = nn.Linear(dec_hid_dim, 1, bias=False)

    def forward(self, hidden, encoder_outputs):

        # hidden = [batch size, dec hid dim]
        # encoder_outputs = [src len, batch size, enc hid dim * 2]

        batch_size = encoder_outputs.shape[1]
        src_len = encoder_outputs.shape[0]

        # repeat decoder hidden state src_len times
        hidden = hidden.unsqueeze(1).repeat(1, src_len, 1)

        encoder_outputs = encoder_outputs.permute(1, 0, 2)

        # hidden = [batch size, src len, dec hid dim]
        # encoder_outputs = [batch size, src len, enc hid dim * 2]

        energy = torch.tanh(self.attn(torch.cat((hidden, encoder_outputs), dim=2)))

        # energy = [batch size, src len, dec hid dim]

        attention = self.v(energy).squeeze(2)

        # attention= [batch size, src len]

        return F.softmax(attention, dim=1)


class DecoderAttention(nn.Module):
    def __init__(
        self,
        output_size,
        emb_dim,
        enc_hid_dim,
        dec_hid_dim,
        dropout,
        attention,
        n_layers=1,
        n_labels=None,
    ):
        super().__init__()

        self.output_size = output_size
        self.attention = attention

        if not n_labels:
            n_labels = output_size

        self.embedding = nn.Embedding(n_labels, emb_dim)

        self.rnn = nn.GRU(
            (enc_hid_dim * 2) + emb_dim, dec_hid_dim, num_layers=n_layers, dropout=0.8
        )

        self.fc_out = nn.Linear((enc_hid_dim * 2) + dec_hid_dim + emb_dim, output_size)

        self.dropout = nn.Dropout(dropout)

    def forward(self, input: torch.tensor, hidden, encoder_outputs):

        # input = [batch size]
        # hidden = [batch size, dec hid dim]
        # encoder_outputs = [src len, batch size, enc hid dim * 2]

        input = input.unsqueeze(0)

        # input = [1, batch size]
        embedded = self.dropout(self.embedding(input))

        # weighted shape torch.Size([1, 8, 1024])
        # embedded torch.Size([1, 3, 10])

        # embedded = [1, batch size, emb dim]

        a = self.attention(hidden, encoder_outputs)

        # a = [batch size, src len]

        a = a.unsqueeze(1)

        # a = [batch size, 1, src len]

        encoder_outputs = encoder_outputs.permute(1, 0, 2)

        # encoder_outputs = [batch size, src len, enc hid dim * 2]

        weighted = torch.bmm(a, encoder_outputs)

        # weighted = [batch size, 1, enc hid dim * 2]

        weighted = weighted.permute(1, 0, 2)

        # weighted = [1, batch size, enc hid dim * 2]

        rnn_input = torch.cat((embedded, weighted), dim=2)

        # rnn_input = [1, batch size, (enc hid dim * 2) + emb dim]

        output, hidden = self.rnn(rnn_input, hidden.unsqueeze(0))

        # output = [seq len, batch size, dec hid dim * n directions]
        # hidden = [n layers * n directions, batch size, dec hid dim]

        # seq len, n layers and n directions will always be 1 in this decoder, therefore:
        # output = [1, batch size, dec hid dim]
        # hidden = [1, batch size, dec hid dim]
        # this also means that output == hidden
        assert (output == hidden).all()

        embedded = embedded.squeeze(0)
        output = output.squeeze(0)
        weighted = weighted.squeeze(0)

        prediction = self.fc_out(torch.cat((output, weighted, embedded), dim=1))

        # prediction = [batch size, output dim]

        return prediction, hidden.squeeze(0)


class RNNEncoderDecoder(nn.Module):

    _START_INDEX = 8

    def __init__(
        self,
        input_size: int = 256,
        encoder_hidden_size: int = 256,
        decoder_hidden_size: int = 256,
        encoder_n_layers: int = 1,
        decoder_n_layers: int = 1,
        output_size: int = 2,
        encoder_dropout: float = 0.1,
        decoder_dropout: float = 0.1,
        device: str = "cpu",
        n_labels: Optional[int] = None,
        output_handler: Optional[OutputHandler] = None,
        output_type: OutputType = OutputType.CLASSIFICATION,
    ):

        super().__init__()

        if decoder_hidden_size is None or np.isnan(decoder_hidden_size):
            decoder_hidden_size = encoder_hidden_size

        if decoder_n_layers is None or np.isnan(decoder_n_layers):
            decoder_n_layers = encoder_n_layers

        self.hidden_size = encoder_hidden_size  # TODO: change

        self.n_layers = encoder_n_layers  # TODO: change

        self.device = device

        if output_handler is None:
            output_handler = LogitOutputHandler()

        self.output_handler = output_handler
        self.accumulated_grad = 0
        self.output_type = output_type

        attention = Attention(encoder_hidden_size, decoder_hidden_size)
        encoder = EncoderAttention(
            input_size,
            encoder_hidden_size,
            decoder_hidden_size,
            encoder_dropout,
        )
        decoder = DecoderAttention(
            output_size,
            output_size,
            encoder_hidden_size,
            decoder_hidden_size,
            decoder_dropout,
            attention,
            n_layers=decoder_n_layers,
            n_labels=n_labels,
        )

        self.encoder = encoder
        self.decoder = decoder

    def forward(self, src, trg, teacher_forcing_ratio=0.5, training_mode=True):

        # src = [src len, batch size]
        # trg = [trg len, batch size]
        # teacher_forcing_ratio is probability to use teacher forcing
        # e.g. if teacher_forcing_ratio is 0.75 we use teacher forcing 75% of the time

        self.accumulated_grad = []

        batch_size = src.shape[1]
        if trg is None:
            trg_len = 3
        else:
            trg_len = trg.shape[0]

        # batch_size = src.shape[1]
        # trg_len = trg.shape[0]
        trg_vocab_size = self.decoder.output_size

        # tensor to store decoder outputs
        outputs = torch.zeros(trg_len, batch_size, trg_vocab_size).to(self.device)

        # encoder_outputs is all hidden states of the input sequence, back and forwards
        # hidden is the final forward and backward hidden states, passed through a linear layer
        encoder_outputs, hidden = self.encoder(src)

        # first input to the decoder is the sequence starting tokens
        if trg is not None:
            input = trg[0, :]
            input = input.to(self.device)
        else:
            input = torch.tensor([self._START_INDEX] * batch_size)
            input = input.to(device=self.device)

        for t in range(1, trg_len):

            # insert input token embedding, previous hidden state and all encoder hidden states
            # receive output tensor (predictions) and new hidden state

            output, hidden = self.decoder(
                input.long().to(self.device), hidden, encoder_outputs
            )

            # place predictions in a tensor holding predictions for each token
            outputs[t] = output

            # decide if we are going to use teacher forcing or not
            if trg is None:
                teacher_force = False
            else:
                rand = random.random()
                teacher_force = rand < teacher_forcing_ratio

            # get the highest predicted token from our predictions
            top1, _ = self.output_handler.decode_logits(output, dim=1)

            # accumulate gradients directly during training to not store
            # batches of output tensors, because they take up a lot of
            # memory
            if training_mode and self.output_type == OutputType.REGRESSION:
                self.accumulated_grad.append(torch.nn.MSELoss()(
                    output.float(),
                    torch.from_numpy(
                        np.asarray(self.output_handler.encode_labels([trg[t]]))
                    )[0]
                    .to(self.device)
                    .float(),
                ))

            # if teacher forcing, use actual next token as next input
            # if not, use predicted token
            input = trg[t] if teacher_force else top1

        if len(self.accumulated_grad):
            self.accumulated_grad = sum(self.accumulated_grad) / len(self.accumulated_grad)

        return outputs
