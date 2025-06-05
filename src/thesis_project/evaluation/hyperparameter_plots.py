from typing import Any, Optional

from matplotlib import pyplot as plt
from matplotlib.axes import Axes
import pandas as pd
import seaborn as sns

from thesis_project.settings import FIGURE_DIR, get_default_hyperparams


# dicts that map parameter names to plot labels

MODEL_PREFIX_TO_NAME_DICT = {"svm": "SVM", "rnn": "RNN", "trf": "Transformer"}

BINNING_PARAM_LABEL_DICT = {
    "50_size_None_sd": "50 bin size",
    "50_size_2_sd": "50 bin size 100 blur",
    "20_size_None_sd": "20 bin size",
    "20_size_2_sd": "20 bin size 40 blur",
}

LABEL_NAME_DICT = {
    "syncat_labels": "syncat",
    "semcat_labels": "semcat",
    "labels_words": "words",
}

# parameters that specify how hyperparameters should be plotted (default plotting_func: plt.plot)

PLOTTING_PARAMS = {
    "svm": {
        "C": {"log_x": True, "plotting_func": "scatter"},
        "gamma": {},
        "kernel": {},
    },
    "rnn": {"hidden_size": {}, "dropout": {"plotting_func": "scatter"}, "n_layers": {}},
    "trf": {"hidden_size": {}, "dropout": {"plotting_func": "scatter"}, "n_layers": {}}
}

# 

RELEVANT_PARAMETERS = {
    "svm_clf": ["C", "gamma", "kernel"],
    "svm_reg": ["estimator__C", "estimator__gamma", "estimator__kernel"],
    "rnn_clf": [
        "dropout",
        "hidden_size",
        "n_layers",
        "batch_size",
        "weight_decay",
        "learning_rate",
    ],
    "rnn_reg": [
        "hidden_size",
        "n_layers",
        "batch_size",
        "weight_decay",
        "learning_rate",
        "dropout",
    ],
    "trf_clf": [
        "dropout",
        "hidden_size",
        "n_layers",
        "batch_size",
        "weight_decay",
        "learning_rate",
    ],
    "trf_reg": [
        "dropout",
        "hidden_size",
        "n_layers",
        "batch_size",
        "weight_decay",
        "learning_rate",
    ],
    "rnn_seq_clf": [
        "encoder_hidden_size",
        "decoder_hidden_size",
        "encoder_n_layers",
        "decoder_n_layers",
        "encoder_dropout",
        "decoder_dropout",
        "weight_decay",
        "learning_rate",
        "batch_size",
    ],
    "rnn_seq_reg": [
        "hidden_size",
        "encoder_n_layers",
        "decoder_n_layers",
        "dropout",
        "weight_decay",
        "learning_rate",
    ],
    "trf_seq_clf": [
        "encoder_hidden_size",
        "decoder_hidden_size",
        "encoder_n_layers",
        "decoder_n_layers",
        "encoder_dropout",
        "decoder_dropout",
        "weight_decay",
        "learning_rate",
    ],
    "trf_seq_reg": [
        "hidden_size",
        "encoder_n_layers",
        "decoder_n_layers",
        "dropout",
        "weight_decay",
        "learning_rate",
    ],
}


NON_MODEL_PARAMS = ["batch_size", "learning_rate", "weight_decay"]


def try_cast(value: Any) -> Any:
    try:
        return float(value)
    except (ValueError, TypeError):
        return value


def plot_best_hyperparam_values(
    ax,
    df_dict: dict[str, pd.DataFrame],
    hp_name: str,
    log_x: bool = False,
    log_y: bool = False,
    col_name: str = "test_mean_acc",
    plotting_func: str = "plot",
    ascending: bool = False,
    session_id: Optional[str] = None,
    xlabel: bool = False,
    ylabel: bool = False,
) -> Axes:
    """
    Plot the optimal (highest or lowest) metric values
    against a specific hyperparameter and store the plot in the given
    pyplot axis object.
    """

    ax.grid(linestyle="--", alpha=0.5, zorder=1)

    if session_id:
        session_results = {}
        for label_name in df_dict.keys():
            session_results[label_name] = df_dict[label_name][
                df_dict[label_name]["session_id"] == session_id
            ]

        session_name = f"session_{session_id}"
    else:
        session_results = df_dict
        session_name = "all_sessions"

    for i, (k, df) in enumerate(session_results.items()):
        if hp_name == "model_size":
            new_df = df
            new_df["model_size"] = new_df["hidden_size"] * new_df["n_layers"]
            if ascending:
                new_df = (
                    new_df[[hp_name, col_name]].groupby(hp_name).min().reset_index()
                )
            else:
                new_df = (
                    new_df[[hp_name, col_name]].groupby(hp_name).max().reset_index()
                )
        else:
            if ascending:
                new_df = df[[hp_name, col_name]].groupby(hp_name).min().reset_index()
            else:
                new_df = df[[hp_name, col_name]].groupby(hp_name).max().reset_index()

        if plotting_func == "plot":
            plot = ax.plot
        elif plotting_func == "scatter":
            plot = ax.scatter

        try:
            new_df = new_df.sort_values(by=hp_name)
        except:
            new_df = new_df.sort_values(by=hp_name, key=lambda x: x.astype(str))

        df.loc[:, hp_name] = df[hp_name].apply(try_cast)

        plot(
            new_df[hp_name].tolist(),
            new_df[col_name].tolist(),
            marker="o",
            linestyle="dashed",
            label=k,
        )

        if log_x:
            ax.set_xscale("log")
        if log_y:
            ax.set_yscale("log")

    ax.set_title(
        f"Best {hp_name.replace('_', ' ')} for {session_name.replace('_', ' ')}"
    )
    ax.legend()

    if xlabel:
        ax.set_xlabel(hp_name, fontsize=16)
    if ylabel:
        ax.set_ylabel(col_name, fontsize=16)

    return ax


def plot_hyperparameter_boxplots(
    results: dict[str, pd.DataFrame],
    hp_name: str,
    output_type: str,
    metric_name: str,
    dir_prefix: str,
    session_id: Optional[str] = None,
    label_names: list[str] = ["syncat_labels", "semcat_labels", "labels_words"],
    logy: bool = False,
    logx: bool = False,
    savefig: bool = False,
) -> plt:
    """
    Create boxplots of the specified evaluation metric for different
    label types and settings of the given hyperparameter.
    """

    if session_id:
        session_results = {}
        for label_name in label_names:
            session_results[label_name] = results[label_name][
                results[label_name]["session_id"] == session_id
            ]

        session_name = f"session_{session_id}"
    else:
        session_results = results
        session_name = "all_sessions"

    plotting_dict = {
        "label_name": [],
        hp_name: [],
        f"{output_type}_{metric_name}_mean": [],
        f"{output_type}_{metric_name}_std": [],
    }

    for label_name in label_names:
        for i, row in session_results[label_name].iterrows():
            plotting_dict["label_name"].append(label_name)
            plotting_dict[hp_name].append(row[hp_name])
            plotting_dict[f"{output_type}_{metric_name}_mean"].append(
                row[f"mean_{output_type}_{metric_name}"]
            )
            plotting_dict[f"{output_type}_{metric_name}_std"].append(
                row[f"std_{output_type}_{metric_name}"]
            )

    fig, axs = plt.subplots(ncols=2, figsize=(20, 8))

    for ax in axs:
        if logx:
            ax.set_xscale("log")
        if logy:
            ax.set_yscale("log")

    model_name = MODEL_PREFIX_TO_NAME_DICT[dir_prefix[:3]]

    fig.suptitle(
        f"{model_name} {output_type} {metric_name} by {hp_name} for {session_name.replace('_', ' ')}"
    )

    sns.boxplot(
        x="label_name",
        y=f"{output_type}_{metric_name}_mean",
        hue=hp_name,
        data=plotting_dict,
        ax=axs[0],
    )

    sns.boxplot(
        x="label_name",
        y=f"{output_type}_{metric_name}_std",
        hue=hp_name,
        data=plotting_dict,
        ax=axs[1],
    )
    if savefig:
        plt.savefig(
            f"{FIGURE_DIR}/{hp_name}_{dir_prefix}_{output_type}_{session_name}.svg"
        )
    return plt


def plot_different_hyperparameters_all_sessions(
    results: dict[str, pd.DataFrame],
    model_name: str,
    task_name: str,
    metric_name: str = "mean_test_accuracy",
    output_type: str = "test",
    ascending: bool = False,
    savefig: bool = False,
):
    """
    Create plots of the optimal metrics per hyperparameter setting for
    different model hyperparameters over all sessions.
    """

    plt.rcParams.update({"font.size": 12})

    hpo_params = list(
        get_default_hyperparams(model_name=model_name, task_name=task_name).keys()
    )

    if "n_heads" in hpo_params:
        hpo_params.remove("n_heads")

    plotting_params = PLOTTING_PARAMS[model_name]

    fig, axs = plt.subplots(1, len(hpo_params), figsize=(15, 5))

    for (param_name, kwargs), ax in zip(plotting_params.items(), axs.flatten()):

        param_name_for_plotting = param_name

        if model_name == "svm" and task_name == "reg":
            param_name_for_plotting = "estimator__" + param_name

        plot_best_hyperparam_values(
            ax,
            results,
            param_name_for_plotting,
            col_name=metric_name,
            ascending=ascending,
            session_id=None,
            xlabel=True,
            **kwargs,
        )

    fig.suptitle(
        f"Best {MODEL_PREFIX_TO_NAME_DICT[model_name]} hyperparameters", fontsize=16
    )
    fig.supylabel("mean test accuracy", fontsize=16)

    plt.tight_layout(rect=[0.01, 0, 1, 1])

    if savefig:
        plt.savefig(
            f"{FIGURE_DIR}/hpo_params_{model_name}_{task_name}_{output_type}.svg"
        )


def plot_different_hyperparameters_per_session(
    results: dict[str, pd.DataFrame],
    model_name: str,
    task_name: str,
    metric_name: str = "mean_test_accuracy",
    output_type: str = "test",
    ascending: bool = False,
    savefig: bool = False,
):
    """
    Create plots of the optimal metrics per hyperparameter setting for
    different model hyperparameters with individual subplots per session.
    """
    session_ids = results["labels_words"]["session_id"].unique().tolist()

    plotting_params = PLOTTING_PARAMS[model_name]

    hpo_params = list(
        get_default_hyperparams(model_name=model_name, task_name=task_name).keys()
    )

    if "n_heads" in hpo_params:
        hpo_params.remove("n_heads")

    fig, axs = plt.subplots(len(session_ids), len(hpo_params), figsize=(20, 25))

    for session_id, row in zip(session_ids, axs):
        for (param_name, kwargs), ax in zip(plotting_params.items(), row):

            param_name_for_plotting = param_name

            if model_name == "svm" and task_name == "reg":
                param_name_for_plotting = "estimator__" + param_name

            plot_best_hyperparam_values(
                ax,
                results,
                param_name_for_plotting,
                ascending=ascending,
                col_name=metric_name,
                session_id=session_id,
                **kwargs,
            )

    if savefig:
        plt.savefig(
            f"{FIGURE_DIR}/hpo_params_per_session_{model_name}_{task_name}_{output_type}.svg"
        )


def plot_binning_param_boxplots(
    results: dict[str, pd.DataFrame],
    task_type: str,
    label_names: list[str],
    metric: str = "accuracy",
    output_type: str = "test",
    savefig: bool = False,
    logscale: bool = False,
):
    """
    Create boxplots of the specified metric per session, model name and binning parameter setting
    over different hyperparameter runs.
    """

    fig, axs = plt.subplots(1, 3, figsize=(18, 6))

    plt.rcParams.update({"font.size": 12})

    metric_name = f"mean_{output_type}_{metric}"
    hp_name = "binning_params"

    fig.suptitle(f"Mean test {metric} by binning parameters")

    for model_name, ax in zip(["svm", "rnn", "trf"], axs):

        dir_prefix = f"{model_name}_{task_type}"

        plotting_dict = {"labels": [], "binning params": [], metric: []}

        for label_name in label_names:
            for i, row in results[model_name][label_name].iterrows():
                plotting_dict["labels"].append(LABEL_NAME_DICT[label_name])
                plotting_dict["binning params"].append(
                    BINNING_PARAM_LABEL_DICT[row[hp_name]]
                )
                plotting_dict[metric].append(row[metric_name])

        model_name = MODEL_PREFIX_TO_NAME_DICT[dir_prefix[:3]]

        ax.set_title(model_name, fontsize=16)

        bp = sns.boxplot(
            x="labels", y=metric, hue="binning params", data=plotting_dict, ax=ax
        )
        ax.set_xlabel("")
        ax.set_ylabel("")
        if logscale:
            ax.set_yscale("log")
        bp.tick_params(labelsize=14)

    fig.supxlabel("labels")
    fig.supylabel(metric_name)
    plt.tight_layout(rect=[0.01, 0, 1, 1])

    if savefig:
        plt.savefig(f"{FIGURE_DIR}/{task_type}_{metric_name}_{hp_name}.svg")
