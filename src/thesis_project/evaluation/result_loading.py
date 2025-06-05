from functools import reduce
import json
import os
from typing import Optional

import pandas as pd

from thesis_project.data_loading import load_session_ids
from thesis_project.settings import EXPERIMENT1_DIR, RESULT_DIR


label_names = ["syncat_labels", "semcat_labels", "labels_words"]

extra_params = ["n_folds", "n_repeats"]


def unify_results(
    path: str,
    dir_prefix: str = None,
    label_names: list[str] = label_names,
    limit_to_number: Optional[int] = None,
) -> dict:
    """
    Load result files from different hyperparameter searches and unify them into a dictionary of
    dataframes.
    :param path: Result folder
    :param dir_prefix: directory prefix for the model and task type (e.g. svm_clf)
    :param limit_to_number: maximum number of result rows per binning parameters and session id to be included in a single dataframe
        (this is relevant to obtain a specific number of run results for fair comparison between different model types)
    """

    dir_names = sorted(os.listdir(path))

    if dir_prefix:
        dir_names = [name for name in dir_names if name.startswith(dir_prefix)]

    result_dict = {name: [] for name in label_names}

    for dir_name in dir_names:

        with open(f"{path}/{dir_name}/optimization_parameters.json", "r") as file:
            param_dict = json.loads(file.read())

        extra_param_dict = {param: param_dict.get(param) for param in extra_params}

        for csv_name in label_names:
            df = pd.read_csv(f"{path}/{dir_name}/{csv_name}.csv")
            for param_name, value in extra_param_dict.items():
                df[param_name] = [value] * len(df)
            if not df.empty:
                result_dict[csv_name].append(df)

    for name, df_list in result_dict.items():
        concatenated_df = pd.concat(df_list)
        if limit_to_number:
            concatenated_df = concatenated_df.groupby(
                ["binning_params", "session_id"]
            ).head(limit_to_number)
        result_dict[name] = concatenated_df

    return result_dict


def get_best_parameter_dict(
    result_dict, quality_metric="mean_test_accuracy", ascending: bool = False
):
    """
    Create a dictionary of dataframes with the best parameter configuration per session.
    """
    df_best_dict = {}

    for label_name, df in result_dict.items():

        new_df = (
            df.assign(group=df[["session_id", quality_metric]].apply(frozenset, axis=1))
            .sort_values(quality_metric, ascending=ascending)
            .groupby("session_id")
            .head(1)
        )

        df_best_dict[label_name] = new_df

    return df_best_dict


def get_best_values(
    prefix,
    input_dir=f"{RESULT_DIR}/final_results/selected_results",
    metric_name: str = "mean_train_accuracy",
    additional_metrics: list[str] = [],
    ascending: bool = False,
    label_names=label_names,
    limit_to_number=None,
):
    """
    Create a dataframe that contains the optimal metric value per session and label type.
    """

    result_dict = unify_results(
        input_dir,
        label_names=label_names,
        dir_prefix=prefix,
        limit_to_number=limit_to_number,
    )
    df_best_dict = get_best_parameter_dict(
        result_dict, quality_metric=metric_name, ascending=ascending
    )
    concatenated_df = pd.concat(df_best_dict.values())

    latex_df = concatenated_df[
        ["session_id", "label_name", metric_name] + additional_metrics
    ]

    df_list = []

    for label_name in latex_df["label_name"].unique():
        new_df = (
            latex_df.loc[latex_df["label_name"] == label_name]
            .reset_index()
            .rename(columns={metric_name: f"{label_name}_{metric_name}"})
        )
        new_df = new_df[["session_id", f"{label_name}_{metric_name}"]]
        df_list.append(new_df)

    df = reduce(
        lambda df1, df2: pd.merge(df1, df2, on="session_id"), df_list
    ).sort_values(by="session_id")

    return df


def create_combined_results_table(results,
                                  metric_name,
                                  label_names):

    session_ids = sorted(load_session_ids(EXPERIMENT1_DIR))
    result_dict = {label_name: {"session_id": [], **{model_name: [] for model_name in results.keys()}} for label_name in label_names}

    for label_name in result_dict.keys():
        for session_id in session_ids:
            result_dict[label_name]["session_id"].append(session_id)
            for model_name, df in results.items():
                current_row = df[(df['session_id'] == int(session_id))]
                result_dict[label_name][model_name].append(current_row[f"{label_name}_{metric_name}"].item())

    return result_dict
