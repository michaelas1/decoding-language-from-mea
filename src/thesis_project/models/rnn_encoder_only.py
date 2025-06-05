import torch
import torch.nn.functional as F

from thesis_project.models import OutputType

class RNNEncoderOnly(torch.nn.Module):

    def __init__(
        self,
        input_size: int = 256,
        hidden_size: int = 256,
        output_size: int = 2,
        n_layers: int = 1,
        dropout: float = 0.1,
        device: str = "cuda",
        bidirectional: bool = False,
        output_type: OutputType = OutputType.CLASSIFICATION,
    ):
        super().__init__()

        self.hidden_size = hidden_size
        self.output_size = output_size
        self.n_layers = n_layers

        self.rnn = torch.nn.GRU(
            input_size,
            hidden_size,
            num_layers=n_layers,
            bias=True,
            batch_first=False,
            dropout=dropout,
            bidirectional=bidirectional,
            device=device,
            dtype=None,
        )

        if bidirectional:
            in_features = hidden_size * 2
        else:
            in_features = hidden_size

        self.readout = torch.nn.Linear(
            in_features=hidden_size, out_features=output_size, bias=True, device=device
        )
        self.output_type = output_type

    def forward(self, x):

        # x = [input_len, batch_size, in_features]

        output, _ = self.rnn(x)

        # output = [input_len, batch_size, hidden_size]

        output = output.permute(1, 0, 2)


        # output = [batch_size, input_len, hidden_size]

        x = torch.mean(output, dim=1)

        # x = [batch_size, hidden_size]


        x = self.readout(x)

        # x = [batch_size, output_size]


        if self.output_type == OutputType.CLASSIFICATION:
            x = F.relu(x)

        return x
