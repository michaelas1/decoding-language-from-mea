import numpy as np
import torch
from torch.optim import Optimizer
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import DataLoader
from tqdm import tqdm
import logging
from typing import Callable, Optional, Tuple
from torcheval.metrics.functional.text import perplexity

from thesis_project.models import OutputType
from thesis_project.models.rnn_encoder_decoder import RNNEncoderDecoder

from thesis_project.models.rnn_encoder_only import RNNEncoderOnly
from thesis_project.models.transformer_encoder_only import TransformerEncoderOnly
from thesis_project.models.transformer_encoder_decoder import TransformerEncoderDecoder
from thesis_project.training.trainer import Trainer

logger = logging.getLogger(__name__)


class NNTrainer(Trainer):
    """
    Trainer class for neural network models.
    """

    _EARLY_STOPPING_DELTA = 0.0001

    def __init__(
        self,
        create_model_func: Optional[Callable] = None,
        dataset_name: str = "unknown_dataset",
        session_id: str = "unknown_session",
        train_data: Optional[DataLoader] = None,
        validation_data: Optional[DataLoader] = None,
        loss_func: Optional[Callable] = None,
        num_epochs: int = 3,
        batch_size: int = 25,
        learning_rate: float = 1e-2,
        weight_decay: float = 1e-3,
        device_name: str = "cpu",
        clip_max_norm: Optional[float] = 1,
        output_type: OutputType = OutputType.CLASSIFICATION,
        metric_dict: dict[str, callable] = None
    ):
        """
        :param create_model_func: Function to create the model to train
        :param dataset_name: Name of the dataset, used in logs/file naming
        :param session_id: Experiment session ID, used in logs/file naming
        :param train_data: The training data loader, expects data in batch second format
        :parm validation_data: The validation data loader, expects data in batch second format
        :param loss_func: The loss function used for training and validation
        :param num_epochs: Number of training epochs
        :param batch_size: Training/validation batch size
        :param learning_rate: Initial learning rate of the optimizer
        :param weight_decay: Weight decay parameter for the optimizer
        :param device_name: Processing unit, expects "cuda" or "cpu"
        :param clip_max_norm: Weight clipping parameter during training,
            only clips weights if the parameter is not None
        :param output_type:
        :param metric_dict:
        """

        self._create_model_func = create_model_func
        self._train_data = train_data
        self._validation_data = validation_data
        self._loss_func = loss_func
        self._num_epochs = num_epochs
        self._batch_size = batch_size
        self._clip_max_norm = clip_max_norm
        self._learning_rate = learning_rate
        self._weight_decay = weight_decay
        self.device_name = device_name
        self.dataset_name = dataset_name
        self.session_id = session_id
        self.model = create_model_func().to(torch.device(device_name))
        self.output_type = output_type
        self.metric_dict = metric_dict

    def train(
        self,
        tensorboard_path: str = "tensorboard/unnamed",
        checkpoint_path: str = "checkpoints/unnamed",
        early_stopping: bool = False,
    ) -> Tuple[float, float]:
        """
        Train the model for a number of epochs.
        :param tensor_path: Path to folder where tensorboard logs are stored
        :param checkpoint_path: Path to folder where checkpoins are stored
        :param include_accuracy: Whether to calculate the accuracy
        """

        writer = SummaryWriter(log_dir=tensorboard_path)

        optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=self._learning_rate,
            weight_decay=self._weight_decay,
        )

        if early_stopping:
            last_loss = 9999999

        for epoch_index in tqdm(range(self._num_epochs)):

            training_metrics = self._train_one_epoch(
                epoch_index,
                self._train_data,
                optimizer,
                writer,
            )

            validation_metrics = self.validate(
                epoch_index, writer
            )

            torch.save(
                {
                    "epoch": epoch_index,
                    "model_state_dict": self.model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "loss": validation_metrics["test_loss"],
                },
                checkpoint_path,
            )

            if early_stopping:
                if (
                    last_loss - validation_metrics["test_loss"]
                    < self._EARLY_STOPPING_DELTA
                ):
                    early_stopping_rounds += 1
                else:
                    early_stopping_rounds = 0

                if early_stopping_rounds > self._N_EARLY_STOPPING_ROUNDS:
                    print(f"Stopped training early at round {epoch_index}")
                    break

                last_loss = validation_metrics["test_loss"]

            # print({**training_metrics, **validation_metrics})

        writer.close()

        return {**training_metrics, **validation_metrics}

    def _pass_one_epoch(
        self,
        data_loader: DataLoader,
        train: bool,
        tb_writer: SummaryWriter,
        epoch_index: int,
        optimizer: Optional[Optimizer] = None,
    ):
        """
        Passes the dataset one time, performs loss and metric calculations
        for each batch and optionally performs optimization of the weights.
        Used by the training and validation functions.
        """
        # from https://pytorch.org/tutorials/beginner/introyt/trainingyt.html

        if train:
            method_name = "train"
        else:
            method_name = "test"

        output_metrics = {f"{method_name}_loss": []}

        for metric_name, _ in self.metric_dict.items():
            output_metrics[f"{method_name}_{metric_name}"] = []

        for i, batch in enumerate(data_loader):

            inputs, labels = batch

            # inputs and labels are expected to be in batch second format:
            # inputs = [input_len, batch_size, input_dim]
            # labels = [output_len, batch_size]

            # accuracy is calculated separately for extended sequence targets

            if train:
                optimizer.zero_grad()

            # forward pass based on model type
            if isinstance(self.model, RNNEncoderOnly):
                # encoder-only model
                idx_outputs = self.model.forward(inputs)
                # idx_outputs = [batch_size, n_labels]

            elif isinstance(self.model, TransformerEncoderOnly):
                idx_outputs = self.model.forward(inputs, labels)
                #if self.label_type == "regression":
                idx_outputs = idx_outputs.T

            elif isinstance(self.model, RNNEncoderDecoder) or isinstance(
                self.model, TransformerEncoderDecoder
            ):
                idx_outputs = self.model.forward(inputs, labels)
                # idx_outputs = [seq_len, batch_size, n_labels]

                output_dim = idx_outputs.shape[-1]


                if isinstance(self.model, RNNEncoderDecoder):

                    original_idx_outputs = idx_outputs[1:]
                    original_labels = labels[1:]

                    idx_outputs = idx_outputs[1:].view(-1, output_dim)
                    labels = labels[1:].reshape(idx_outputs.shape[0])


                elif isinstance(self.model, TransformerEncoderDecoder):
                    if self.model.output_handler and self.output_type == OutputType.REGRESSION:
                        idx_outputs = idx_outputs[:-1].view(-1, output_dim)
                        labels = labels[1:].reshape(idx_outputs.shape[0])

                        #print(labels)
                        labels = self.model.output_handler.encode_labels([labels])[0]
                        labels = torch.from_numpy(labels).to(self.device_name)

                    else:
                        original_idx_outputs = idx_outputs[1:]
                        original_labels = labels[1:]

                        idx_outputs = idx_outputs[:-1].view(-1, output_dim)
                        labels = labels[1:].reshape(idx_outputs.shape[0])

            else:
                raise Exception(f"Unknown model type {type(self.model)}")

            # end model forward_pass

            if self.output_type == OutputType.CLASSIFICATION:

                loss = self._loss_func(idx_outputs, labels).float()
            elif self.output_type == OutputType.REGRESSION and isinstance(self.model, RNNEncoderDecoder):
                loss = self.model.accumulated_grad
            elif self.output_type == OutputType.REGRESSION and isinstance(self.model, TransformerEncoderDecoder):
                loss = self._loss_func(
                    idx_outputs, labels  # .to(torch.float64)
                )
            elif self.output_type == OutputType.REGRESSION:
                loss = self._loss_func(
                    idx_outputs, labels.permute(1, 0)  # .to(torch.float64)
                )
            else:
                raise Exception(f"Unknown label type {self.output_type}")

            if train:
                loss.backward()

                if self._clip_max_norm:
                    torch.nn.utils.clip_grad_norm_(
                        self.model.parameters(), self._clip_max_norm
                    )

                optimizer.step()

            # gather metrics for reporting

            output_metrics[f"{method_name}_loss"].append(loss.item())
            for metric_name, metric_func in self.metric_dict.items():
                if self.output_type == OutputType.CLASSIFICATION:
                    if isinstance(self.model, RNNEncoderDecoder) or isinstance(self.model, TransformerEncoderDecoder):
                        output_metrics[f"{method_name}_{metric_name}"].append(metric_func(original_idx_outputs.detach().cpu(),
                                                                                        original_labels.detach().cpu()))
                    else:
                        output_metrics[f"{method_name}_{metric_name}"].append(metric_func(idx_outputs.argmax(dim=-1).detach().cpu(),
                                                                                        labels.detach().cpu()))
                else:
                    output_metrics[f"{method_name}_{metric_name}"].append(metric_func(idx_outputs.detach().cpu(),
                                                                                    labels.detach().cpu().T))

        for metric_name, metric_val in output_metrics.items():
            output_metrics[metric_name] = sum(metric_val) / len(metric_val)
            if epoch_index:
                tb_writer.add_scalar(f'{metric_name}/{method_name}', sum(metric_val) / len(metric_val), epoch_index)

        return output_metrics

    def _train_one_epoch(
        self,
        epoch_index: int,
        training_loader: DataLoader,
        optimizer: Optimizer,
        tb_writer: SummaryWriter,
    ) -> Tuple[float, float]:
        """
        Inner training loop that trains for one epoch.
        :param epoch_index: Index of the current epoch.
        :param training_loader: Data loader for training data.
        :param optimizer: Optimizer
        :param include_accuracy: Whether to calculate the accuracy
        """

        # from https://pytorch.org/tutorials/beginner/introyt/trainingyt.html

        return self._pass_one_epoch(
            training_loader,
            True,
            tb_writer,
            epoch_index,
            optimizer        )

    def validate(
        self, epoch_index: int, tb_writer: SummaryWriter
    ) -> Tuple[float, float]:
        """
        Validate the model.
        :param tensorboard_path: Path to folder where tensorboard logs are stored
        :param include_accuracy: Whether to calculate the accuracy
        """
        
        # Disable gradient computation and reduce memory consumption.
        with torch.no_grad():

            output_metrics = self._pass_one_epoch(
                self._validation_data,
                False,
                tb_writer,
                epoch_index,
                None
                )

        return output_metrics
