from typing import Optional
import torch

from thesis_project.models import OutputType

class TransformerEncoderOnly(torch.nn.Module):

    def __init__(
        self,
        input_size: int = 256,
        hidden_size: int = 256,
        output_size: int = 2,
        n_layers: int = 1,
        dropout: float = 0.1,
        n_heads: int = 8,
        device: str = "cpu",
        output_type: OutputType = OutputType.CLASSIFICATION,
    ):

        super().__init__()

        self.hidden_size = hidden_size
        self.n_layers = n_layers
        self.device = device
        self.output_type = output_type

        self.input_reshape = torch.nn.Linear(
            in_features=input_size, out_features=hidden_size
        )

        self.transformer = torch.nn.Transformer(
            d_model=hidden_size,
            nhead=n_heads,
            num_encoder_layers=self.n_layers,
            num_decoder_layers=1,
            dim_feedforward=hidden_size * 4,
            dropout=dropout,
            activation="relu",
            layer_norm_eps=1e-05,
            batch_first=True,
            norm_first=False,
            bias=True,
            device=device,
            dtype=None,
        )

        self.readout = torch.nn.Linear(
            in_features=hidden_size, out_features=output_size
        )

    def forward(
        self,
        src: torch.Tensor,
        tgt: torch.Tensor,
        src_mask: Optional[torch.Tensor] = None,
        src_key_padding_mask: Optional[torch.Tensor] = None,
        src_is_causal: Optional[bool] = None,
        mean_first: bool = False,
    ):

        # src = [seq_len, batch_size, n_channels]
        # tgt = [batch_size] / [batch_size, n_labels]

        if mean_first:
            # Input mode 1 (mean first)
            # convert spikerates to input "embedding"

            src = torch.mean(src, dim=0)
            # src = [batch_size, n_channels]

            encoder_output = self.transformer.encoder(
                src,
                mask=src_mask,
                src_key_padding_mask=src_key_padding_mask,
                is_causal=src_is_causal,
            )

            # encoder_output = [batch_size, hidden_size]
            # x = [batch_size, n_labels]

        else:
            if self.output_type == OutputType.CLASSIFICATION:
                tgt = tgt.unsqueeze(0)

            self.transformer.batch_first = False
            src = self.input_reshape(src)
            encoder_output = self.transformer.encoder(
                src,
                mask=src_mask,
                src_key_padding_mask=src_key_padding_mask,
                is_causal=src_is_causal,
            )
            # encoder_output = [[seq_len, batch_size, n_channels]

            encoder_output = torch.mean(encoder_output, dim=0)
            # encoder_output = [batch_size, n_channels]

        x = self.readout(encoder_output)
        x = x.permute(1, 0)
        return x
