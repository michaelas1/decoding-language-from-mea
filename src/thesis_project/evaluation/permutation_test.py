import torch
from tqdm import tqdm
from thesis_project.data_loading import decode_binning_params, load_train_test_ids
from thesis_project.evaluation.model_inference import ModelInference
from thesis_project.evaluation.result_loading import (
    get_best_parameter_dict,
    unify_results,
)
from thesis_project.preprocessing.label_preparation import (
    get_label_dict,
    prepare_spikerates_for_session,
)
from thesis_project.settings import EXPERIMENT2_DIR, RESULT_DIR
from thesis_project.training.best_model_retraining import INPUT_DIR
from thesis_project.training.metrics import get_sequence_classification_metrics


def run_permutation_test(
    model_type: str,
    task_type: str,
    n_permutations: int,
    session_id: str = "20240708",
    experiment_name: str = "experiment2",
    label_name: str = "sentences",
    embedding_dim=1346,
):

    # load parameters
    result_dict = unify_results(
        INPUT_DIR, label_names=[label_name], dir_prefix=f"{model_type}_{task_type}"
    )
    df_best_dict = get_best_parameter_dict(
        result_dict, quality_metric="mean_test_loss", ascending=True
    )

    # load labels

    data_split = load_train_test_ids(experiment_name)[session_id]  # [label_name]
    labels = get_label_dict(session_id, experiment_name, task_type)[label_name][
        0
    ].cuda()
    train_idx, test_idx = data_split["train_ids"], data_split["test_ids"]

    # extract best parameters

    current_params = df_best_dict[label_name]
    current_params = current_params.loc[
        current_params["session_id"] == int(session_id)
    ]

    binning_params_string = current_params["binning_params"].values[0]
    binning_params = decode_binning_params(binning_params_string)

    spikerates_path = f"{EXPERIMENT2_DIR}/binned_spikerates"

    spikerates = prepare_spikerates_for_session(
        session_id=session_id,
        path=spikerates_path,
        bin_size=binning_params["bin_size"],
        blur_sd=binning_params["blur_sd"],
        experiment=experiment_name,
    )

    # load model

    model_inference = ModelInference(model_type, task_type, session_id, label_name)

    result_metrics = []

    metrics = get_sequence_classification_metrics()

    actual_metrics = {"train": {}, "test": {}}

    # forward pass over unpermuted data for comparison

    actual_pred_train = model_inference.chunked_inference(
        spikerates[train_idx], labels[train_idx], embedding_dim=embedding_dim
    )
    actual_pred_test = model_inference.chunked_inference(
        spikerates[test_idx], labels[test_idx], embedding_dim=embedding_dim
    )

    for metric_name, metric_func in metrics.items():

        actual_metrics["train"][metric_name] = metric_func(
            torch.from_numpy(actual_pred_train)[:, 1:, :],
            labels[train_idx].detach().cpu()[:, 1:],
        )
        actual_metrics["test"][metric_name] = metric_func(
            torch.from_numpy(actual_pred_test)[:, 1:, :],
            labels[test_idx].detach().cpu()[:, 1:],
        )

    # calculate metric values for permuted data

    for i in tqdm(range(n_permutations)):

        # shuffle labels within their train/test sets

        shuffled_train_labels = labels[train_idx][torch.randperm(len(train_idx))]
        shuffled_test_labels = labels[test_idx][torch.randperm(len(test_idx))]

        y_pred_train = model_inference.chunked_inference(
            spikerates[train_idx], shuffled_train_labels, embedding_dim=embedding_dim
        )
        y_pred_test = model_inference.chunked_inference(
            spikerates[test_idx], shuffled_test_labels, embedding_dim=embedding_dim
        )

        # compute result metrics
        metrics = get_sequence_classification_metrics()

        training_metrics = {}
        test_metrics = {}

        for metric_name, metric_func in metrics.items():

            training_metrics[metric_name] = metric_func(
                torch.from_numpy(y_pred_train)[:, 1:, :],
                shuffled_train_labels.detach().cpu()[:, 1:],
            )
            test_metrics[metric_name] = metric_func(
                torch.from_numpy(y_pred_test)[:, 1:, :],
                shuffled_test_labels.detach().cpu()[:, 1:],
            )

        result_metrics.append({"train": training_metrics, "test": test_metrics})

    return result_metrics, actual_metrics
