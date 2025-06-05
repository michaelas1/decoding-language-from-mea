import gc
import logging
import os
from collections import defaultdict
from datetime import datetime
from typing import Optional, Union

import numpy as np
import torch
from sklearn.model_selection import KFold, StratifiedKFold
from sklearn.utils.multiclass import type_of_target
from torch.utils.data import DataLoader

from thesis_project.data_loading import SpikerateDataset, data_collate_sequence_first
from thesis_project.models import OutputType
from thesis_project.models.rnn_encoder_only import RNNEncoderOnly
from thesis_project.training.nn_trainer import NNTrainer
from thesis_project.training.svm_trainer import SVMTrainer

logger = logging.getLogger(__name__)


def cross_validate(
    inputs,
    labels,
    create_model_function: callable,
    loss_function: Union[object, callable],
    weight_decay: float = 0.01,
    learning_rate: float = 1e-2,
    k_folds: int = 5,
    k_fold_labels=None,
    num_epochs: int = 5,
    batch_size: int = 32,
    shuffle: bool = True,
    random_state: Optional[bool] = None,
    dataset_name: str = "unknown_dataset",
    session_id: str = "unknown_session",
    mode: str = "NN",
    device_name: str = "cuda",
    output_type: str = OutputType.CLASSIFICATION,
    metric_dict: dict[str, callable] = None
) -> dict:
    """
    Perform cross-validation on the input dataset.
    """

    if k_fold_labels is not None:
        k_fold_labels = labels

    if output_type == OutputType.CLASSIFICATION and type_of_target(labels) in [
        "binary",
        "multiclass",
    ]:
        kfold = StratifiedKFold(
            n_splits=k_folds, shuffle=shuffle, random_state=random_state
        )
        y_labels = labels

    elif (
        output_type == OutputType.REGRESSION
        and type_of_target(labels) == "multiclass-multioutput"
        and k_fold_labels
    ):
        # ensure a stratified split for encoder-only models
        # trained on embedding regression
        kfold = StratifiedKFold(
            n_splits=k_folds, shuffle=shuffle, random_state=random_state
        )
        y_labels = k_fold_labels

    else:
        kfold = KFold(n_splits=k_folds, shuffle=shuffle, random_state=random_state)
        y_labels = labels

    results = defaultdict(lambda: [])

    if mode == "NN":
        path = f"{dataset_name}_{session_id}/{type(create_model_function()).__name__}_{datetime.now()}"
        tensorboard_path = f"tensorboard/{path}"
        checkpoint_path = f"checkpoints/{path}"
        os.makedirs(tensorboard_path)
        os.makedirs(checkpoint_path)

    for fold, (train_ids, test_ids) in enumerate(kfold.split(inputs, y=y_labels)):

        if mode == "NN":

            training_loader = DataLoader(
                SpikerateDataset(
                    inputs[train_ids].to(torch.device(device_name)),
                    labels[train_ids].to(torch.device(device_name)),
                ),
                batch_size=batch_size,
                collate_fn=data_collate_sequence_first,
                shuffle=shuffle,
            )
            validation_loader = DataLoader(
                SpikerateDataset(
                    inputs[test_ids].to(torch.device(device_name)),
                    labels[test_ids].to(torch.device(device_name)),
                ),
                batch_size=batch_size,
                collate_fn=data_collate_sequence_first,
                shuffle=shuffle,
            )

            trainer = NNTrainer(
                create_model_func=create_model_function,
                loss_func=loss_function,
                train_data=training_loader,
                validation_data=validation_loader,
                num_epochs=num_epochs,
                dataset_name=dataset_name,
                session_id=session_id,
                weight_decay=weight_decay,
                learning_rate=learning_rate,
                device_name=device_name,
                output_type=output_type,
                metric_dict=metric_dict
            )

            output_metrics = trainer.train(
                tensorboard_path=f"{tensorboard_path}/train_val_{fold}",
                checkpoint_path=f"{checkpoint_path}/train_val_{fold}",
            )

            # empty cuda cache due to memory limitations
            del trainer
            del training_loader
            del validation_loader
            torch.cuda.empty_cache()

        elif mode == "SVM":
            trainer = SVMTrainer(
                create_model_func=create_model_function,
                inputs=inputs,
                labels=labels,
                loss_func=loss_function,
                train_ids=train_ids,
                test_ids=test_ids,
                metric_dict=metric_dict
            )

            training_metrics = trainer.train()
            validation_metrics = trainer.validate()
            output_metrics = {**training_metrics, **validation_metrics}

        for metric_key, metric_val in output_metrics.items():
            results[metric_key].append(metric_val)

    mean_metrics = {}
    std_metrics = {}

    for metric_key, metric_val in results.items():
        mean_metrics[f"mean_{metric_key}"] = sum(metric_val) / len(metric_val)
        std_metrics[f"std_{metric_key}"] = np.std(metric_val)

    return {**mean_metrics, **std_metrics}
