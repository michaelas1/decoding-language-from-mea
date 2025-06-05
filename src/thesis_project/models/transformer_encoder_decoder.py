from typing import Optional
import torch

from thesis_project.models.output_handler import OutputHandler


class TransformerEncoderDecoder(torch.nn.Module):

    START_TOKEN = 8
    END_TOKEN = 9
    MAX_LENGTH = 20
    PAD_IDX = 3

    def __init__(
        self,
        input_size: int = 256,
        hidden_size: int = 512,
        output_size=2,
        encoder_n_layers: int = 1,
        decoder_n_layers: int = 1,
        dropout: float = 0.1,
        n_heads: int = 8,
        device: str = "cpu",
        n_labels: Optional[int] = None,
        output_handler: Optional[OutputHandler] = None,
        output_type: str = "classification",
    ):

        super().__init__()

        if decoder_n_layers is None:
            decoder_n_layers = encoder_n_layers

        self.hidden_size = hidden_size
        self.encoder_n_layers = encoder_n_layers
        self.decoder_n_layers = decoder_n_layers
        self.device = device
        self.output_type = output_type

        if not n_labels:
            n_labels = output_size

        self.input_reshape = torch.nn.Linear(
            in_features=input_size, out_features=hidden_size, device=device
        )

        self.decoder_embedding = torch.nn.Embedding(
            num_embeddings=n_labels, embedding_dim=hidden_size, device=device
        )

        self.transformer = torch.nn.Transformer(
            nhead=n_heads,
            d_model=hidden_size,
            num_encoder_layers=encoder_n_layers,
            num_decoder_layers=decoder_n_layers,
            dim_feedforward=hidden_size * 4,
            dropout=dropout,
            activation="relu",
            custom_encoder=None,
            custom_decoder=None,
            layer_norm_eps=1e-05,
            batch_first=False,
            norm_first=False,
            bias=True,
            device=device,
            dtype=None,
        )

        self.readout = torch.nn.Linear(
            in_features=hidden_size, out_features=output_size, device=device
        )

        self.output_handler = output_handler

    def generate_autoregressively(self, x):

        input_tensor = torch.Tensor([self.START_TOKEN]).long().to(self.device)
        encoder_output = self.transformer.encoder(x)

        print(encoder_output.shape)

        for i in range(self.MAX_LENGTH - 1):
            embedded = self.decoder_embedding(input_tensor)
            decoder_output = self.transformer.decoder(embedded, encoder_output)
            decoder_output = self.readout(decoder_output)
            decoder_output = torch.argmax(decoder_output, dim=-1)
            input_tensor = torch.cat((input_tensor, decoder_output[-1:]))

        return input_tensor

    def forward_old(self, x, y=None):
        # convert spikerates to input "embedding"

        x = torch.mean(x, dim=0).unsqueeze(dim=0)

        x = self.input_reshape(x)

        if y is None:
            x = self.generate_autoregressively(x)

        else:
            decoder_input = self.decoder_embedding(y)
            tgt_mask = self.transformer.generate_square_subsequent_mask(y.size(0)).to(
                self.device
            )
            tgt_key_padding_mask = (y == self.PAD_IDX).transpose(0, 1)
            x = self.transformer(
                x,
                decoder_input,
                tgt_mask=tgt_mask,
                tgt_is_causal=True,
                tgt_key_padding_mask=tgt_key_padding_mask,
            )
            x = self.readout(x)

        return x

    def forward(
        self,
        x,
        y=None,
        src_mask: Optional[torch.Tensor] = None,
        src_key_padding_mask: Optional[torch.Tensor] = None,
        src_is_causal: Optional[bool] = None,
        mean_first=False,
    ):

        if mean_first:
            # convert spikerates to input "embedding"
            x = torch.mean(x, dim=0).unsqueeze(dim=0)
            x = self.input_reshape(x)
        else:
            self.transformer.batch_first = False

            x = self.input_reshape(x)
            x = self.transformer.encoder(
                x,
                mask=src_mask,
                src_key_padding_mask=src_key_padding_mask,
                is_causal=src_is_causal,
            )

            # x = [[seq_len, batch_size, n_channels]


        if y is None:
            x = self.generate_autoregressively(x)

        else:
            decoder_input = self.decoder_embedding(y)
            tgt_mask = self.transformer.generate_square_subsequent_mask(y.size(0)).to(
                self.device
            )
            tgt_key_padding_mask = (y == self.PAD_IDX).transpose(0, 1)
            x = self.transformer(
                x,
                decoder_input,
                tgt_mask=tgt_mask,
                tgt_is_causal=True,
                tgt_key_padding_mask=tgt_key_padding_mask,
            )
            x = self.readout(x)

        return x
