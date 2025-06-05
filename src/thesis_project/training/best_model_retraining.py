import gc
import os
import pickle
import numpy as np
import pandas as pd
from sklearn.metrics import log_loss, mean_squared_error
import torch
from thesis_project.data_loading import (
    SpikerateDataset,
    data_collate_sequence_first,
    decode_binning_params,
    load_train_test_ids,
)
from thesis_project.evaluation.result_loading import (
    get_best_parameter_dict,
    unify_results,
)
from thesis_project.models import OutputType
from thesis_project.models.model_factory import ModelFactory
from thesis_project.preprocessing.german_to_english import GERMAN_TO_ENGLISH_DICT
from thesis_project.preprocessing.label_preparation import (
    get_label_dict,
    prepare_spikerates_for_session,
)
from thesis_project.settings import EXPERIMENT1_DIR, EXPERIMENT2_DIR, RESULT_DIR
from thesis_project.training.metrics import (
    get_classification_metrics,
    get_regression_metrics,
    get_sequence_classification_metrics,
)
from thesis_project.training.nn_trainer import NNTrainer
from thesis_project.training.svm_trainer import SVMTrainer
from thesis_project.word2vec_embeddings import download_pretrained_model
from torch.nn.functional import cross_entropy
from torch.utils.data import DataLoader
from torch.nn import MSELoss


DEVICE_NAME = "cuda"


RELEVANT_PARAMETERS = {
    "logistic_regression_reg": [],
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

INPUT_DIR = f"{RESULT_DIR}/final_results/selected_results"
OUTPUT_DIR = f"{RESULT_DIR}/final_results/retrained_models"

svm_regression_loss_func = mean_squared_error


def custom_cross_entropy(x, y):
    return cross_entropy(x, y, ignore_index=2)


def create_model_trainer(
    task_type: str,
    model_type: str,
    current_params,
    spikerates,
    labels,
    label_dict,
    train_idx,
    test_idx,
):

    prefix = f"{model_type}_{task_type}"

    relevant_param_dict = {
        parameter: current_params[parameter].values[0]
        for parameter in RELEVANT_PARAMETERS[prefix]
        if parameter in current_params
    }

    if task_type == "clf":
        metric_dict = get_classification_metrics()

    elif task_type == "reg":
        model_name = "glove-twitter-25"
        model = download_pretrained_model(model_name)
        encoded_labels = np.asarray(
            [
                model[GERMAN_TO_ENGLISH_DICT[label_dict[label.item()]]]
                for label in labels
            ]
        )
        metric_dict = get_regression_metrics()

    elif task_type == "seq_clf":
        metric_dict = get_sequence_classification_metrics()

    else:
        metric_dict = {}

    #### SVM models
    if model_type in ["svm", "logistic_regression"]:

        if task_type == "reg":
            relevant_param_dict = {
                key[11:] if key.startswith("estimator__") else key: value
                for key, value in relevant_param_dict.items()
            }
            loss_func = svm_regression_loss_func
            svm_input_labels = encoded_labels
        else:
            loss_func = log_loss
            svm_input_labels = labels

        model_factory = ModelFactory(
            model_type,
            task_type,
            {
                "preprocessing_method": current_params["preprocessing_method"].values[
                    0
                ],
                **relevant_param_dict,
            },
        )

        if model_type == "svm":
            create_model_func = model_factory.create_svm

        trainer = SVMTrainer(
            create_model_func=create_model_func,
            inputs=spikerates,
            labels=svm_input_labels,
            loss_func=loss_func,
            train_ids=train_idx,
            test_ids=test_idx,
            output_type=task_type,
            metric_dict=metric_dict,
        )

        return trainer, relevant_param_dict

    #### NN models

    for key, val in relevant_param_dict.items():
        if isinstance(val, np.int64):
            relevant_param_dict[key] = int(val)

    batch_size = relevant_param_dict.get("batch_size", 32)

    spikerates = torch.from_numpy(spikerates).float().to(torch.device(DEVICE_NAME))

    if task_type == "reg":
        loader_labels = torch.from_numpy(encoded_labels)
    else:
        loader_labels = labels

    loader_labels = loader_labels.to(torch.device(DEVICE_NAME))

    training_loader = DataLoader(
        SpikerateDataset(
            spikerates[train_idx],
            loader_labels[train_idx],
        ),
        batch_size=batch_size,
        collate_fn=data_collate_sequence_first,
        shuffle=True,
    )

    validation_loader = DataLoader(
        SpikerateDataset(
            spikerates[test_idx],
            loader_labels[test_idx],
        ),
        batch_size=batch_size,
        collate_fn=data_collate_sequence_first,
        shuffle=True,
    )

    # specify output type depending on model architecture

    n_labels = None
    word_label_dict = None
    embedding_name = None

    if task_type == "clf":
        n_labels = len(np.unique(labels))
    if task_type == "reg":
        n_labels = len(loader_labels[0])
    if task_type == "seq_clf":
        n_labels = len(label_dict)
    if task_type == "seq_reg":
        word_label_dict = label_dict
        embedding_name = "glove-twitter-25"

    hyperparams = {
        param_name: relevant_param_dict[param_name]
        for param_name in relevant_param_dict
        if param_name not in ["weight_decay", "learning_rate", "batch_size"]
    }

    model_factory = ModelFactory(
        model_type,
        task_type,
        {
            **hyperparams,
            # "hidden_size": relevant_param_dict["hidden_size"],
            # "n_layers": relevant_param_dict["n_layers"],
            # "dropout": relevant_param_dict["dropout"],
            "device": DEVICE_NAME,
        },
        n_labels=n_labels,
        word_label_dict=word_label_dict,
        embedding_name=embedding_name,
    )

    create_model_func = model_factory.create_model

    trainer = NNTrainer(
        create_model_func=create_model_func,
        num_epochs=100,
        loss_func=cross_entropy if "clf" in task_type else MSELoss(),
        train_data=training_loader,
        validation_data=validation_loader,
        # num_epochs=num_epochs,
        # session_id=session_id,
        batch_size=batch_size,
        weight_decay=relevant_param_dict["weight_decay"],
        learning_rate=relevant_param_dict["learning_rate"],
        device_name=DEVICE_NAME,
        output_type=(
            OutputType.CLASSIFICATION if "clf" in task_type else OutputType.REGRESSION
        ),
        metric_dict=metric_dict,
    )

    return trainer, relevant_param_dict


def retrain_model(
    task_type: str,
    model_type: str,
    quality_metric: str = "mean_test_accuracy",
    ascending=False,
    save_model=True,
    limit_to_number=None,
    tensorboard_path: str = "tensorboard/unnamed",
):

    if task_type == "clf":
        label_names = ["labels_words", "syncat_labels", "semcat_labels"]
    elif task_type == "reg":
        label_names = ["labels_words"]
    else:
        label_names = ["sentences"]

    if task_type in ["clf", "reg"]:
        experiment_name = "experiment1"
    else:
        experiment_name = "experiment2"

    result_dict = unify_results(
        INPUT_DIR,
        label_names=label_names,
        dir_prefix=f"{model_type}_{task_type}",
        limit_to_number=limit_to_number,
    )
    df_best_dict = get_best_parameter_dict(
        result_dict, quality_metric=quality_metric, ascending=ascending
    )

    data_split = load_train_test_ids(experiment_name)

    if not os.path.exists(f"{OUTPUT_DIR}/{model_type}_{task_type}"):
        os.mkdir(f"{OUTPUT_DIR}/{model_type}_{task_type}")

    result_df = pd.DataFrame()

    for session_id, split in data_split.items():

        result_entry = {"session_id": session_id}
        label_dict = get_label_dict(session_id, experiment_name, task_type)

        train_idx, test_idx = split["train_ids"], split["test_ids"]

        for label_name, (labels, label_str_dict) in label_dict.items():

            torch.cuda.empty_cache()
            gc.collect()

            # if not label_name in df_best_dict: # TODO: remove
            #     continue

            current_params = df_best_dict[label_name]
            current_params = current_params.loc[
                current_params["session_id"] == int(session_id)
            ]  #

            binning_params_string = current_params["binning_params"].values[0]
            binning_params = decode_binning_params(binning_params_string)

            if experiment_name == "experiment1":
                spikerates_path = EXPERIMENT1_DIR
            else:
                spikerates_path = f"{EXPERIMENT2_DIR}/binned_spikerates"

            spikerates = prepare_spikerates_for_session(
                session_id=session_id,
                path=spikerates_path,
                bin_size=binning_params["bin_size"],
                blur_sd=binning_params["blur_sd"],
                experiment=experiment_name,
            )

            trainer, relevant_param_dict = create_model_trainer(
                task_type,
                model_type,
                current_params,
                spikerates,
                labels,
                label_str_dict,
                train_idx,
                test_idx,
            )

            # include_accuracy = task_Type == "clf"
            if model_type in ["svm", "logistic_regression"]:
                training_metrics = trainer.train()
                validation_metrics = trainer.validate()
                output_metrics = {**training_metrics, **validation_metrics}
            else:
                output_metrics = trainer.train(tensorboard_path=tensorboard_path)

            # if model_type == "svm":
            #     validation_metrics = trainer.validate()
            # else:
            #     validation_metrics = trainer.validate()

            result_df = result_df._append(
                {
                    **result_entry,
                    "label_name": label_name,
                    **output_metrics,
                    **relevant_param_dict,
                    "preprocessing_method": current_params[
                        "preprocessing_method"
                    ].values[0],
                    "binning_params": binning_params_string,
                },
                ignore_index=True,
            )

            if save_model:
                with open(
                    f"{OUTPUT_DIR}/{model_type}_{task_type}/model_{session_id}_{label_name}.pkl",
                    "wb",
                ) as file:
                    pickle.dump(trainer.model, file)

    if save_model:
        result_df.to_csv(f"{OUTPUT_DIR}/{model_type}_{task_type}/results.csv")

    return result_df
