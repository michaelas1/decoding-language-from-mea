import csv
from typing import Any, Callable, Dict, List
import numpy as np
from scipy import stats
from sklearn.discriminant_analysis import StandardScaler
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.metrics import make_scorer
from sklearn.model_selection import (
    GridSearchCV,
    RandomizedSearchCV,
    RepeatedKFold,
    RepeatedStratifiedKFold,
)
from sklearn.multioutput import MultiOutputRegressor
from sklearn.pipeline import FunctionTransformer, Pipeline
from sklearn.svm import SVC, SVR
from skopt import BayesSearchCV

from thesis_project.models import OutputType
from thesis_project.parameter_optimization.parameter_optimization import (
    ParameterOptimization,
    get_distribution_by_sample_type,
    try_convert_to_float,
)

class ProbabilityDistributionWrapper():
    def __init__(self, sample_type: str, values):
        self.sample_type = sample_type
        self.values = values

    def rvs(self, size=None, random_state=None):
        if self.sample_type == "fixed":
            return self.values
        elif self.sample_type == "choice":
            # not using np.random.choice to avoid conversion to np dtypes
            return try_convert_to_float(np.random.choice(self.values, size=size))

        elif self.sample_type == "uniform":
            return np.random.uniform(low=self.values[0], high=self.values[1])
        elif self.sample_type == "exp":
            return self.values[2] ** np.random.uniform(high=self.values[0], low=self.values[1])
        else:
            raise ValueError(f"Unknown sample type {self.sample_type}")


class FixedParametersKfold:
    """
    Custom Kfold-wrapper that disregards passed parameters in favor of
    stored targets. Used to obtain stratified splits for multi-output
    target datasets (i.e. embedding regression)."""

    def __init__(self, kfold, y):
        self.kfold = kfold
        self.y = y
        self.X = np.zeros(len(y))

    def split(self, X=None, y=None, groups=None):
        new_split = self.kfold.split(self.X, y=self.y, groups=groups)
        return new_split

    def get_n_splits(self, X=None, y=None, groups=None):
        return self.kfold.get_n_splits(self.X, self.y, groups)


class SVMOptimization(ParameterOptimization):

    def __init__(
        self,
        model_name: str = None,
        task_name: str = None,
        experiment: str = "experiment1",
        session_ids: List[str] | None = None,
        limit_to_ids: Dict | None = None,
        preprocessing_methods: List[Callable[..., Any]] = None,
        label_names: List[str] = None,
        output_dir: str = None,
        binning_params: Dict | None = None,
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
        #if task_name == "clf":
        for metric_name, metric_func in self.metric_dict.items():
            self.metric_dict[metric_name] = make_scorer(metric_func)


    def preprocessing_method_to_step(self, preprocessing_method: str):
        if preprocessing_method == "flatten":
            return FunctionTransformer(
                lambda x: x.reshape(x.shape[0], x.shape[1] * x.shape[2])
            )
        elif preprocessing_method == "mean_sequence":
            return FunctionTransformer(lambda x: x.mean(axis=1))
        elif preprocessing_method == "mean_channel":
            return FunctionTransformer(lambda x: x.mean(axis=2))
        else:
            raise ValueError(f"Unknown preprocessing method '{preprocessing_method}'")

    def get_estimator(self):
        if self.model_name == "linear_regression":
            return LinearRegression()
        elif self.model_name == "logistic_regression":
            return LogisticRegression()
        elif self.output_type == OutputType.CLASSIFICATION:
            return SVC()
        elif self.output_type == OutputType.REGRESSION:
            return MultiOutputRegressor(SVR(max_iter=1000000), n_jobs=-1)
        else:
            raise ValueError(f"Unknown output_mode: {self.output_type}")


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

        preprocessing_step = self.preprocessing_method_to_step(preprocessing_method)

        if self.output_type == OutputType.CLASSIFICATION:
            repeated_cv = RepeatedStratifiedKFold(
                n_splits=self.n_folds,
                n_repeats=self.n_repeats,
                random_state=self.random_seed,
            )
        elif self.output_type == OutputType.REGRESSION:

            if k_fold_labels is not None:
                repeated_cv = RepeatedStratifiedKFold(
                    n_splits=self.n_folds,
                    n_repeats=self.n_repeats,
                    random_state=self.random_seed,
                )
                repeated_cv = FixedParametersKfold(repeated_cv, k_fold_labels)

            else:
                repeated_cv = RepeatedKFold(
                    n_splits=self.n_folds,
                    n_repeats=self.n_repeats,
                    random_state=self.random_seed,
                )

        else:
            raise ValueError(f"Unknown output mode {self.output_type}")
        

        grid_search = Pipeline(
            steps=[
                ("preprocessing", preprocessing_step),
                ("normalize", StandardScaler()),
                (
                    "grid_search",
                    GridSearchCV(
                        self.get_estimator(),
                        scoring=self.metric_dict,
                        param_grid=self.search_params,
                        cv=repeated_cv,
                        return_train_score=True,
                        refit=False,
                    ),
                ),
            ]
        )

        grid_search.fit(spikerates, labels)
        cv_results = grid_search["grid_search"].cv_results_

        self._write_results_row(output_subdir, cv_results["params"], cv_results, metadata)


    # def get_distribution_by_sample_type(sample_type, values):
    #     if sample_type == "fixed":
    #         return stats.rv_discrete(name=sample_type, values=(values, [1])),
    #     elif sample_type == "choice":
    #         return values
    #         #return stats.rv_discrete(name=sample_type, values=(values, [1 / len(values)] * len(values))),
    #     elif sample_type == "uniform":
    #         return stats.uniform(loc=values[0], scale=values[0])
    #     elif sample_type == "exp":
    #         return values[2] ** stats.uniform(loc=values[0], scale=values[0])
    #     else:
    #         raise ValueError(f"Unknown sample type {sample_type}")

    def construct_random_search_params(self):
        random_search_params = {}
        for sample_type, values in self.search_params.items():
            random_search_params[sample_type] = ProbabilityDistributionWrapper(list(values.keys())[0], list(values.values())[0])
        return random_search_params


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

        preprocessing_step = self.preprocessing_method_to_step(preprocessing_method)

        if self.output_type == OutputType.CLASSIFICATION:
            repeated_cv = RepeatedStratifiedKFold(
                n_splits=self.n_folds,
                n_repeats=self.n_repeats,
                random_state=self.random_seed,
            )
        elif self.output_type == OutputType.REGRESSION:

            if k_fold_labels is not None:
                repeated_cv = RepeatedStratifiedKFold(
                    n_splits=self.n_folds,
                    n_repeats=self.n_repeats,
                    random_state=self.random_seed,
                )
                repeated_cv = FixedParametersKfold(repeated_cv, k_fold_labels)

            else:
                repeated_cv = RepeatedKFold(
                    n_splits=self.n_folds,
                    n_repeats=self.n_repeats,
                    random_state=self.random_seed,
                )

        else:
            raise ValueError(f"Unknown output mode {self.output_type}")

        random_search_params = self.construct_random_search_params()
        print("random", random_search_params)
        print("metric dict", self.metric_dict)
        print("estimator", self.get_estimator())


        random_search = Pipeline(
            steps=[
                ("preprocessing", preprocessing_step),
                ("normalize", StandardScaler()),
                (
                    "random_search",
                    RandomizedSearchCV(
                        self.get_estimator(),
                        scoring=self.metric_dict,
                        n_iter=self.n_random_runs,
                        param_distributions=random_search_params,
                        cv=repeated_cv,
                        return_train_score=True,
                        refit=False,
                        n_jobs=-1
                    ),
                ),
            ]
        )

        random_search.fit(spikerates, labels)
        cv_results = random_search["random_search"].cv_results_
        self._write_results_row(output_subdir, cv_results["params"], cv_results, metadata)


    def bayes_search(self, spikerates, labels, label_word_dict, preprocessing_method):

        preprocessing_step = self.preprocessing_method_to_step(preprocessing_method)

        repeated_cv = RepeatedStratifiedKFold(
            n_splits=self.n_folds,
            n_repeats=self.n_repeats,
            random_state=self.random_seed,
        )

        pipe_search = Pipeline(
            steps=[
                ("preprocessing", preprocessing_step),
                ("normalize", StandardScaler()),
                ("model", self.get_estimator()),
            ]
        )

        bayes_search = BayesSearchCV(pipe_search,
                                    self.search_params,
                                    scoring=self.metric_dict,
                                    cv=repeated_cv)

        # grid_search.fit(spikerates[random_permutation], labels[random_permutation])
        bayes_search.fit(spikerates, labels)
        return [{**bayes_search.best_params_, "best_acc": bayes_search.best_score_}]
