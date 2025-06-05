from abc import ABC
import csv
from datetime import datetime
import random
from time import time
import json
import os
import pickle
from typing import Any, Callable, Dict, List, Optional

import numpy as np
import pandas as pd
from sklearn.discriminant_analysis import StandardScaler
from sklearn.model_selection import GridSearchCV, RepeatedStratifiedKFold
from sklearn.pipeline import FunctionTransformer, Pipeline
from sklearn.svm import SVC
from skopt import BayesSearchCV
import torch
from tqdm import tqdm

from thesis_project.models import OutputType
from thesis_project.settings import DATA_DIR, EXPERIMENT2_DIR, RESULT_DIR, get_default_hyperparams
from thesis_project.data_loading import load_session_ids
from thesis_project.preprocessing.translation import GERMAN_TO_ENGLISH_DICT
from thesis_project.preprocessing.label_preparation import (
    prepare_labels_for_session,
    prepare_spikerates_for_session,
)
from thesis_project.preprocessing.tokenization import SingleWordTokenizer
from thesis_project.training.cross_validation import cross_validate
from thesis_project.word2vec_embeddings import (
    calculate_w2v_embeddings,
    download_pretrained_model,
)


def try_convert_to_float(values):
    if isinstance(values, list):
        for value in values:
            try:
                value = float(value)
            except Exception:
                ...
        return values
    try:
        values = float(values)
    except Exception:
        ...
    return values


def get_distribution_by_sample_type(sample_type, values):
    if sample_type == "fixed":
        return values
    elif sample_type == "choice":
        # not using np.random.choice to avoid conversion to np dtypes
        return random.choice(values)
    elif sample_type == "uniform":
        return try_convert_to_float(np.random.uniform(low=values[0], high=values[1]))
    elif sample_type == "exp":
        return values[2] ** np.random.uniform(high=values[0], low=values[1])
    else:
        raise ValueError(f"Unknown sample type {sample_type}")


class OptimizationType:

    GRID_SEARCH = "grid"
    RANDOM_SEARCH = "random"
    BAYES_SEARCH = "bayes"


class ParameterOptimization(ABC):

    def __init__(
        self,
        model_name: str,
        task_name: str,
        experiment: str = "experiment1",
        session_ids: Optional[List[str]] = None,
        limit_to_ids: Optional[Dict] = None,
        preprocessing_methods: List[Callable] = None,
        label_names: List[str] = None,
        output_dir: str = None,
        binning_params: Optional[dict] = None,
        n_folds: int = None,
        n_repeats: int = None,
        search_params: Any = None,
        optimization_type: str = None,
        data_dir: str = None,
        random_seed: int = None,
        output_type: str = OutputType.CLASSIFICATION,
        embedding: str = None,
        translate_label_words: bool = False,
        n_random_runs: int = None,
        metric_dict: dict = None
    ):
        self.model_name = model_name
        self.task_name = task_name
        self.experiment = experiment
        self.session_ids = session_ids
        self.limit_to_ids = limit_to_ids
        self.preprocessing_methods = preprocessing_methods
        self.label_names = label_names
        self.output_dir = output_dir
        self.binning_params = binning_params
        self.n_folds = n_folds
        self.n_repeats = n_repeats
        self.search_params = search_params
        self.optimization_type = optimization_type
        self.data_dir = data_dir
        self.random_seed = random_seed
        self.output_type = output_type
        self.embedding = embedding
        self.translate_label_words = (translate_label_words,)
        self.n_random_runs = n_random_runs
        self.metric_dict = metric_dict

    def get_optimization_parameters(self):
        return {
            "experiment": self.experiment,
            "session_ids": self.session_ids,
            "limit_to_ids": self.limit_to_ids,
            "preprocessing_methods": self.preprocessing_methods,
            "label_names": self.label_names,
            "output_dir": self.output_dir,
            "binning_params": self.binning_params,
            "n_folds": self.n_folds,
            "n_repeats": self.n_repeats,
            "search_params": self.search_params,
            "optimization_type": self.optimization_type,
            "data_dir": self.data_dir,
            "embedding": self.embedding,
            "output_mode": self.output_type,
            "random_seed": self.random_seed,
            "translate_label_words": self.translate_label_words,
            "n_random_runs": self.n_random_runs,
            "metric_dict": self.metric_dict
        }
    
            
    def _write_results_row(self, output_subdir, params, result_metrics, metadata):
        with open(f"{output_subdir}/{metadata['label_name']}.csv", mode='a', newline='') as file:
            writer = csv.writer(file)
            for i, params in enumerate(params):
                csv_entry = {**params,
                             **metadata}
                for metric_name in list(self.metric_dict.keys()) + (["loss"] if self.model_name not in ["svm", "linear_regression", "logistic_regression"] else []):
                    csv_entry[f"mean_train_{metric_name}"] = result_metrics[f"mean_train_{metric_name}"][i]
                    csv_entry[f"mean_test_{metric_name}"] = result_metrics[f"mean_test_{metric_name}"][i]
                    csv_entry[f"std_train_{metric_name}"] = result_metrics[f"std_train_{metric_name}"][i]
                    csv_entry[f"std_test_{metric_name}"] = result_metrics[f"std_test_{metric_name}"][i]
                    #print(metric_name, result_metrics[f"mean_train_{metric_name}"][i], result_metrics[f"mean_test_{metric_name}"][i])
                #print()

                result_row = []
                for column_name in self._csv_column_names:
                    result_row.append(csv_entry.get(column_name))

                writer.writerow(result_row)

    def grid_search(
        self,
        spikerates,
        labels,
        label_word_dict,
        preprocessing_method,
        k_fold_labels=None,
    ):
        raise NotImplementedError()

    def random_search(
        self,
        spikerates,
        labels,
        label_word_dict,
        preprocessing_method,
        k_fold_labels=None,
    ):
        raise NotImplementedError()

    def bayes_search(self, spikerates, labels, label_word_dict, preprocessing_method):
        raise NotImplementedError()

    def parameter_search(
        self,
        spikerates,
        labels,
        label_word_dict,
        preprocessing_method,
        k_fold_labels=None,
        output_subdir=None,
        metadata=None
    ):

        if self.optimization_type == OptimizationType.GRID_SEARCH:
            search_func = self.grid_search
        elif self.optimization_type == OptimizationType.RANDOM_SEARCH:
            search_func = self.random_search
        elif self.optimization_type == OptimizationType.BAYES_SEARCH:
            search_func = self.bayes_search
        else:
            raise ValueError(f"Unknown uptimization type {self.optimization_type}")

        return search_func(
            spikerates,
            labels,
            label_word_dict,
            preprocessing_method,
            k_fold_labels=k_fold_labels,
            output_subdir=output_subdir,
            metadata=metadata
        )

    def run(self, output_name: str = "model"):

        timestamp = datetime.now().strftime("%Y-%m-%dT%H:%M:%S%z")
        end_time = time()

        if self.output_dir:
            output_subdir = f"{self.output_dir}/{output_name}_{timestamp}"
            os.mkdir(output_subdir)

            with open(f"{output_subdir}/optimization_parameters.json", "w") as file:
                file.write(
                    json.dumps(
                        self.get_optimization_parameters(),
                        default=lambda o: "<not serializable>",
                        indent=4,
                    )
                )

            if self.experiment == "experiment1":
                if self.task_name == "clf":
                    label_names = ["syncat_labels", "semcat_labels", "labels_words"]
                else:
                    label_names = ["labels_words"]
            
            else:
                label_names = ["sentences"]
            
            for label_name in label_names:
                with open(f"{output_subdir}/{label_name}.csv", mode='a', newline='') as file:
                    writer = csv.writer(file)
                    csv_entry = ["session_id", "label_name", "preprocessing_method", "binning_params", "embedding"]
                    param_list = list(self.search_params.keys()) + list(get_default_hyperparams(self.model_name, self.task_name).keys())
                    if hasattr(self, "model_params"):
                        param_list = param_list + list(self.model_params.keys())
                    if hasattr(self, "fixed_cv_params"):
                        param_list = param_list + list(self.fixed_cv_params.keys())

                    param_list = list(set(param_list))
                    csv_entry.extend(param_list)

                    for metric_name in sorted(list(self.metric_dict.keys()) + (["loss"] if self.model_name != "svm" else [])):
                        csv_entry.append(f"mean_train_{metric_name}")
                        csv_entry.append(f"mean_test_{metric_name}")
                        csv_entry.append(f"std_train_{metric_name}")
                        csv_entry.append(f"std_test_{metric_name}")
                    writer.writerow(csv_entry)
                    self._csv_column_names = csv_entry

        result_dict = {label_name: [] for label_name in self.label_names}

        start_time = time()

        if self.output_type == OutputType.REGRESSION:
            print(f"Loading embedding model {self.embedding} for regression...")
            w2v_model = download_pretrained_model(self.embedding)

        for session_id in tqdm(self.session_ids, desc="session"):

            if self.experiment == "experiment1":
                (
                    spikerates,
                    numeric_labels_words,
                    numeric_labels_img,
                    numeric_syncat_labels,
                    numeric_semcat_labels,
                    labels_words_dict,
                    labels_img_dict,
                    syncat_labels_dict,
                    semcat_labels_dict,
                ) = prepare_labels_for_session(session_id, self.data_dir)

                if self.translate_label_words:
                    labels_words_dict = {
                        label: GERMAN_TO_ENGLISH_DICT[word]
                        for label, word in labels_words_dict.items()
                    }
                if self.task_name == "clf":
                    label_dict = {
                        "syncat_labels": (numeric_syncat_labels, syncat_labels_dict),
                        "semcat_labels": (numeric_semcat_labels, semcat_labels_dict),
                        "labels_words": (numeric_labels_words, labels_words_dict),
                        #"labels_img": (numeric_labels_img, labels_img_dict),
                    }
                else:
                    label_dict = {
                        "labels_words": (numeric_labels_words, labels_words_dict)
                    }

            elif self.experiment == "experiment2":

                sentences_path = f"{EXPERIMENT2_DIR}/sentences_new.pkl"
                with open(sentences_path, "rb") as file:
                    sentences = pickle.load(file)

                tokenizer = SingleWordTokenizer()
                tokenizer_file_path = f"{EXPERIMENT2_DIR}/single_word_token_dict.json"
                tokenizer.token_dict_from_file(tokenizer_file_path)
                sentences, _ = tokenizer.tokenize_samples(sentences)

                label_dict = {
                    "sentences": (
                        torch.from_numpy(sentences).long(),
                        tokenizer.token_dict,
                    )
                }

            else:
                raise ValueError(f"Unknown experiment {self.experiment}")

            for binning_params in tqdm(self.binning_params, desc="binning_params"):

                if binning_params["bin_size"] != 50 or binning_params["blur_sd"] or self.experiment == "experiment2":
                    spikerates = prepare_spikerates_for_session(
                        session_id=session_id,
                        path=self.data_dir,
                        bin_size=binning_params["bin_size"],
                        blur_sd=binning_params["blur_sd"],
                        experiment=self.experiment,
                    )
                if isinstance(spikerates, np.ndarray):
                    spikerates = torch.from_numpy(spikerates)
                spikerates = spikerates.float()

                for preprocessing_method in tqdm(
                    self.preprocessing_methods, desc="preprocessing_method"
                ):
                    for label_name in tqdm(self.label_names, desc="labels"):

                        labels, label_idx_dict = label_dict[label_name]

                        k_fold_labels = None

                        if self.output_type == OutputType.REGRESSION and "seq" not in self.task_name:
                            word_list = [
                                label_idx_dict[i].lower().replace("ü", "ue")
                                for i in range(len(label_idx_dict.keys()))
                            ]
                            vectors = calculate_w2v_embeddings(
                                word_list, w2v_model, normalize=False
                            )
                            k_fold_labels = labels
                            labels = torch.from_numpy(
                                np.asarray([vectors[i.item()] for i in labels])
                            ).float()

                        if self.limit_to_ids:
                            self.parameter_search(
                                spikerates[self.limit_to_ids[session_id]],
                                labels[self.limit_to_ids[session_id]],
                                label_idx_dict,
                                preprocessing_method,
                                k_fold_labels=(
                                    None
                                    if k_fold_labels is None
                                    else k_fold_labels[self.limit_to_ids[session_id]]
                                ),
                                output_subdir=output_subdir,
                                metadata={
                                "session_id": session_id,
                                "label_name": label_name,
                                "preprocessing_method": preprocessing_method,
                                "binning_params": f"{binning_params['bin_size']}_size_{binning_params['blur_sd']}_sd",
                                "embedding": self.embedding
                                }
                            )
                        else:
                            self.parameter_search(
                                spikerates,
                                labels,
                                preprocessing_method,
                                k_fold_labels=k_fold_labels,
                                output_subdir=output_subdir,
                                metadata={
                                "session_id": session_id,
                                "label_name": label_name,
                                "preprocessing_method": preprocessing_method,
                                "binning_params": f"{binning_params['bin_size']}_size_{binning_params['blur_sd']}_sd",
                                "embedding": self.embedding
                                }
                            )
        with open(f"{output_subdir}/timestamp.txt", "w") as file:
                file.write(f"Took {end_time - start_time} seconds")
        return result_dict
