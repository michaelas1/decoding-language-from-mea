import pickle
from typing import Optional
import numpy as np
import torch
from thesis_project.data_loading import (
    construct_spikerates_filename,
    get_different_labels,
    labels_words_to_numeric,
    load_session_ids,
    load_spike_data,
)
from thesis_project.preprocessing.spike_preprocessing import normalize_spikerates
from thesis_project.preprocessing.tokenization import SingleWordTokenizer
from thesis_project.settings import EXPERIMENT1_DIR, EXPERIMENT2_DIR


def pad_spikerates(spikerates: np.ndarray):
    """
    Pad spikerates by appending zeros to the end of shorter rows.
    """

    max_length = max([run.shape[0] for run in spikerates])
    padded_spikerates = np.ndarray(
        (spikerates.shape[0], max_length, spikerates[0].shape[1])
    )

    for i, run in enumerate(spikerates):

        if run.shape[0] < max_length:
            missing_length = max_length - run.shape[0]
            padded = np.pad(
                run,
                pad_width=((0, missing_length), (0, 0)),
                mode="constant",
                constant_values=(0, 0),
            )
            padded_spikerates[i] = padded

        else:
            padded_spikerates[i] = run

    return padded_spikerates


def prepare_labels_for_session(session_id: str, data_dir: str):
    # load data and labels

    if session_id == "all":
        
        spikerates_list = []
        labels_words_list = []
        labels_list = []

        session_ids = load_session_ids(data_dir=data_dir)
        for session_id in session_ids:

            labels_words, labels, spikerates, _ = load_spike_data(
                f"{data_dir}/{session_id}_naming"
            )

            if spikerates.dtype == object:
                spikerates = pad_spikerates(spikerates)

            spikerates_list.append(spikerates)
            labels_words_list.append(labels_words)
            labels_list.append(labels)


        spikerates = np.concatenate(spikerates_list)
        labels = np.concatenate(labels_list)
        labels_words = np.concatenate(labels_words_list)


    
    else:
        labels_words, labels, spikerates, spiketimes = load_spike_data(
        f"{data_dir}/{session_id}_naming"
    )

        if spikerates.dtype == object:
            spikerates = pad_spikerates(spikerates)

    numeric_labels_words, labels_words_dict = labels_words_to_numeric(labels_words)
    numeric_labels_img, labels_img_dict = labels_words_to_numeric(labels)

    semcat_labels, syncat_labels = get_different_labels(labels_words)

    numeric_syncat_labels, syncat_labels_dict = labels_words_to_numeric(syncat_labels)
    numeric_semcat_labels, semcat_labels_dict = labels_words_to_numeric(semcat_labels)

    return (
        spikerates,
        numeric_labels_words,
        numeric_labels_img,
        numeric_syncat_labels,
        numeric_semcat_labels,
        labels_words_dict,
        labels_img_dict,
        syncat_labels_dict,
        semcat_labels_dict,
    )


def prepare_spikerates_for_session(
    session_id: str,
    path: str,
    bin_size: Optional[int] = None,
    blur_sd: Optional[float] = None,
    experiment: Optional[str] = None
):
    if session_id == "all":
        spikerates_list = []
        session_ids = load_session_ids(data_dir=path)
        for session_id in session_ids:
                spikerates_path = construct_spikerates_filename(
                    session_id, path, bin_size=bin_size, blur_sd=blur_sd, experiment=experiment
                )
                spikerates = np.load(spikerates_path, allow_pickle=True)

                try:
                    spikerates = np.asarray(spikerates)
                except Exception:
                    spikerates = np.asarray(spikerates, dtype=object)

                if spikerates.dtype == object:
                    spikerates = pad_spikerates(spikerates)

                spikerates_list.append(spikerates)

        spikerates = np.concatenate(spikerates_list)

    else:
        spikerates_path = construct_spikerates_filename(
            session_id, path, bin_size=bin_size, blur_sd=blur_sd, experiment=experiment
        )
        spikerates = np.load(spikerates_path, allow_pickle=True)

        try:
            spikerates = np.asarray(spikerates)
        except Exception:
            spikerates = np.asarray(spikerates, dtype=object)

        if spikerates.dtype == object:
            spikerates = pad_spikerates(spikerates)

    spikerates = normalize_spikerates(spikerates)


    return spikerates


def extend_label_to_sequence(labels, start_token_idx, end_token_idx):
    """
    Turns single-target classification labels into sequence classification by
    appending start and end token idx. Used by encoder-decoder models.
    """

    # do not include start token because it is not generated
    # return torch.stack([labels, torch.Tensor([end_token_idx] * len(labels))], dim=-1).reshape(-1, 2).long()

    return (
        torch.stack(
            [
                torch.Tensor([start_token_idx] * len(labels)),
                labels,
                torch.Tensor([end_token_idx] * len(labels)),
            ],
            dim=-1,
        )
        .reshape(-1, 3)
        .long()
    )


def get_label_dict(session_id: str, experiment_name: str, task_type: str):
    
    if experiment_name == "experiment1":
        spikerates, numeric_labels_words, numeric_labels_img, numeric_syncat_labels, numeric_semcat_labels, \
        labels_words_dict, labels_img_dict, syncat_labels_dict, semcat_labels_dict = \
        prepare_labels_for_session(session_id, data_dir=EXPERIMENT1_DIR)

        label_dict = {
            "labels_words": (numeric_labels_words, labels_words_dict)}
        
        if task_type == "clf":
            label_dict["syncat_labels"] = (numeric_syncat_labels, syncat_labels_dict)
            label_dict["semcat_labels"] = (numeric_semcat_labels, semcat_labels_dict)

    if experiment_name == "experiment2":
        sentences_path = f"{EXPERIMENT2_DIR}/sentences_new.pkl"
        with open(sentences_path, "rb") as file:
            sentences = pickle.load(file)
            
        tokenizer = SingleWordTokenizer()
        tokenizer_file_path = f"{EXPERIMENT2_DIR}/single_word_token_dict.json"
        tokenizer.token_dict_from_file(tokenizer_file_path)
        sentences, _ = tokenizer.tokenize_samples(sentences)

        label_dict = {
            "sentences": (torch.from_numpy(sentences).long(), tokenizer.token_dict)
        }

    return label_dict