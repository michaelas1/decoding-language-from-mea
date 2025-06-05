import random
from typing import Any, Callable, Dict, List, Optional, Tuple
import numpy as np
import torch
import gc

from tqdm import tqdm
from thesis_project.models import OutputType
from thesis_project.models.model_factory import ModelFactory
from thesis_project.parameter_optimization.parameter_optimization import (
    ParameterOptimization,
    get_distribution_by_sample_type,
)
from thesis_project.training.cross_validation import cross_validate


class NNOptimization(ParameterOptimization):

    def __init__(
        self,
        model_name: str,
        task_name: str,
        experiment: str = "experiment1",
        session_ids: Optional[List[str]] = None,
        preprocessing_methods: List[Callable] = None,
        limit_to_ids: Dict[str, List[int]] = None,
        label_names: List[str] = None,
        output_dir: str = None,
        binning_params: List[Dict] = None,
        n_folds: int = None,
        n_repeats: int = None,
        search_params: Any = None,
        optimization_type: str = None,
        data_dir: str = None,
        random_seed: int = None,
        create_model_func: Callable = None,
        fixed_cv_params: Dict[str, Any] = None,
        model_params: Dict[str, Any] = None,
        output_type: str = OutputType.CLASSIFICATION,
        embedding: str = None,
        translate_label_words: bool = False,
        n_random_runs: int = False,
        metric_dict: dict = None
    ):
        super().__init__(
            model_name,
            task_name,
            experiment,
            session_ids,
            limit_to_ids,
            preprocessing_methods,
            label_names,
            output_dir,
            binning_params,
            n_folds,
            n_repeats,
            search_params,
            optimization_type,
            data_dir,
            random_seed,
            output_type,
            embedding,
            translate_label_words,
            n_random_runs,
            metric_dict
        )
        self.create_model_func = create_model_func
        self.fixed_cv_params = fixed_cv_params
        self.model_params = model_params

    def get_optimization_parameters(self):
        params = super().get_optimization_parameters()
        return {
            **params,
            "model_params": self.model_params,
            "fixed_cv_params": self.fixed_cv_params,
        }

    @staticmethod
    def get_parameter_combinations(
        param_dict: Dict[str, List[Any]]
    ) -> List[Dict[str, Any]]:
        """
        Recursively construct all combinations of parameters in the dictionary to
        run a grid search.
        """

        if len(param_dict.items()) == 1:
            result_list = [
                {param_name: list_entry}
                for param_name, param_list in param_dict.items()
                for list_entry in param_list
            ]
            return result_list

        result_dict_list = []
        param_name, param_list = next(iter(param_dict.items()))

        rest_params = {k: v for k, v in param_dict.items() if k != param_name}
        dict_list = NNOptimization.get_parameter_combinations(rest_params)

        for sub_dict in dict_list:
            for param in param_list:
                result_dict_list.append({**sub_dict, param_name: param})

        return result_dict_list

    @staticmethod
    def sample_from_param_distribution(
        param_bounds: Dict[str, Tuple[str, Any]]
    ) -> Dict[str, Any]:
        param_sample = {}
        print("param bound", param_bounds)
        for param_name, (sample_type, values) in param_bounds.items():
            param_sample[param_name] = get_distribution_by_sample_type(
                sample_type, values
            )

        return param_sample

    def grid_search(
        self,
        spikerates,
        labels,
        label_word_dict,
        preprocessing_method,
        k_fold_labels=None,
        output_subdir=None,
        metadata=None
    ):

        model_params_combinations = NNOptimization.get_parameter_combinations(
            self.model_params
        )
        search_param_combinations = NNOptimization.get_parameter_combinations(
            self.search_params
        )

        result_list = []

        for model_param_sample in model_params_combinations:

            # specify output type depending on model architecture

            n_labels = None
            word_label_dict = None
            
            if self.task_name == "clf":
                n_labels=len(np.unique(labels))
            if self.task_name == "reg":
                n_labels = len(labels[0])
            if self.task_name == "seq_clf":
                n_labels = len(label_word_dict)
            if self.task_name == "seq_reg":
                word_label_dict = label_word_dict

            # create model

            model_factory = ModelFactory(
                self.model_name,
                self.task_name,
                #labels,
                n_labels=n_labels,
                word_label_dict=word_label_dict,
                embedding_name=self.embedding,
                hyperparams=model_param_sample,
            )

            for search_param_sample in search_param_combinations:

                results = cross_validate(
                    spikerates,
                    labels,
                    model_factory.create_model,
                    k_fold_labels=k_fold_labels,
                    k_folds=self.n_folds,
                    metric_dict=self.metric_dict,
                    output_type=self.output_type,
                    **self.fixed_cv_params,
                    **search_param_sample,
                )
                result_list.append(
                    {
                        **results,
                        **model_param_sample,
                        **self.fixed_cv_params,
                        **search_param_sample,
                    }
                )

                self._write_results_row(output_subdir, [{**model_factory.hyperparams, **self.fixed_cv_params, **search_param_sample}],
                                                        {k: [v] for k, v in results.items()},
                                                        metadata)

                # gc.collect()
                # torch.cuda.empty_cache()
                # print("after gc", torch.cuda.mem_get_info()[0])

        return result_list

    def random_search(
        self,
        spikerates,
        labels,
        label_word_dict,
        preprocessing_method,
        k_fold_labels=None,
        output_subdir=None,
        metadata=None
    ):

        result_list = []

        for _ in tqdm(range(self.n_random_runs)):
            model_param_sample = self.sample_from_param_distribution(self.model_params)
            search_param_sample = self.sample_from_param_distribution(
                self.search_params
            )

            print(model_param_sample)
            print(search_param_sample)

            # specify output type depending on model architecture

            n_labels = None
            word_label_dict = None
            
            if self.task_name == "clf":
                n_labels=len(np.unique(labels))
            if self.task_name == "reg":
                n_labels = len(labels[0])
            if self.task_name == "seq_clf":
                n_labels = len(label_word_dict)
            if self.task_name == "seq_reg":
                word_label_dict = label_word_dict


            model_factory = ModelFactory(
                self.model_name,
                self.task_name,
                n_labels=n_labels,
                word_label_dict=word_label_dict,
                embedding_name=self.embedding,
                hyperparams=model_param_sample,
            )

            create_model_func = model_factory.create_model

            results = cross_validate(
                spikerates,
                labels,
                create_model_func,
                k_fold_labels=k_fold_labels,
                k_folds=self.n_folds,
                metric_dict=self.metric_dict,
                output_type=self.output_type,
                **self.fixed_cv_params,
                **search_param_sample,
            )
            result_list.append(
                {
                    **results,
                    **model_param_sample,
                    **self.fixed_cv_params,
                    **search_param_sample,
                }
            )

            self._write_results_row(output_subdir, [{**model_factory.hyperparams, **self.fixed_cv_params, **search_param_sample}],
                                        {k: [v] for k, v in results.items()},
                                        metadata)


            gc.collect()
            torch.cuda.empty_cache()
            #print("after gc", torch.cuda.mem_get_info()[0])

        return result_list
