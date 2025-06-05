from typing import Callable, Optional
import numpy as np
from sklearn.utils.multiclass import type_of_target
from sklearn.metrics import accuracy_score
from thesis_project.models import OutputType
from thesis_project.training.trainer import Trainer


class SVMTrainer(Trainer):
    """
    Trainer class for support vector machines.
    """

    def __init__(
        self,
        create_model_func: Optional[Callable] = None,
        inputs=None,
        labels=None,
        train_ids=None,
        loss_func=None,
        test_ids=None,
        output_type=OutputType.CLASSIFICATION,
        metric_dict: dict[str, callable] = None
    ):

        self._create_model_func = create_model_func
        self._loss_func = loss_func
        self._inputs = inputs
        self._labels = labels
        self._train_ids = train_ids
        self._test_ids = test_ids
        self.output_type = output_type
        self.model = self._create_model_func()
        self.metric_dict = metric_dict

    def train(self):

        self.model.fit(self._inputs[self._train_ids], y=self._labels[self._train_ids])

        output_metrics = {}

        if type_of_target(self._labels) in ["binary", "multiclass"]:
            # classification
            predictions = self.model.predict_proba(self._inputs[self._train_ids])
        else:
            # regression
            predictions = self.model.predict(self._inputs[self._train_ids])

        output_metrics["train_loss"] = self._loss_func(
            self._labels[self._train_ids], predictions
        )

        for metric_name, metric_func in self.metric_dict.items():
            if self.output_type == OutputType.CLASSIFICATION:
                output_metrics[f"train_{metric_name}"] = metric_func(self._labels[self._train_ids], predictions.argmax(-1))
            else:
                output_metrics[f"train_{metric_name}"] = metric_func(self._labels[self._train_ids], predictions)

        return output_metrics


    def validate(self):

        output_metrics = {}

        predictions = self.model.predict(self._inputs[self._test_ids])

        if type_of_target(self._labels) in ["binary", "multiclass"]:
            # classification
            predictions = self.model.predict_proba(self._inputs[self._test_ids])
        else:
            # regression
            predictions = self.model.predict(self._inputs[self._test_ids])

        output_metrics["test_loss"] = self._loss_func(
            self._labels[self._test_ids], predictions
        )

        for metric_name, metric_func in self.metric_dict.items():
            if self.output_type == OutputType.CLASSIFICATION:
                output_metrics[f"test_{metric_name}"] = metric_func(self._labels[self._test_ids], predictions.argmax(-1))
            else:
                output_metrics[f"test_{metric_name}"] = metric_func(self._labels[self._test_ids], predictions)

        return output_metrics
