import numpy as np
from sklearn.metrics import r2_score
from sklearn.metrics import accuracy_score, explained_variance_score, mean_squared_error, precision_score, recall_score, f1_score
from torch.nn.functional import cross_entropy

from torch import cosine_similarity
import torch
from torcheval.metrics.functional import perplexity

def convert_metric_for_multiclass_use(metric_func):
    def classification_metric(x, y):
        if len(np.unique(x)) > 2:
            return metric_func(x, y, average="micro")
        else:
            #return metric_func(x, y, average="binary")
            return metric_func(x, y, average="micro")

        
    return classification_metric

def cosine_similarity_with_typecheck(x, y):
    if not isinstance(x, torch.Tensor):
        x = torch.from_numpy(x)
    if not isinstance(y, torch.Tensor):
       # y = torch.from_numpy(y).mean()#.item()
        y = torch.from_numpy(y).mean().item()

    return cosine_similarity(x, y)


def get_classification_metrics():
    return {
        "accuracy": accuracy_score,
        "precision": convert_metric_for_multiclass_use(precision_score),
        "recall": convert_metric_for_multiclass_use(recall_score),
        "f1": convert_metric_for_multiclass_use(f1_score)
    }

def get_regression_metrics():
    return {
        "mse": mean_squared_error,
        "r2": r2_score,
        "explained_variance": explained_variance_score
    }

def get_sequence_classification_metrics():
    return {
        "perplexity": lambda input, target: perplexity(input, target), #, ignore_index=2),
        #"cross_entropy": cross_entropy
    }
