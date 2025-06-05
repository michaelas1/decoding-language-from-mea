import json
import os
import re
import pickle
from typing import Optional
from sklearn.discriminant_analysis import StandardScaler
import torch

import numpy as np
from torch.utils.data import Dataset, DataLoader
from torch.utils.data._utils.collate import default_collate

from thesis_project.settings import EXPERIMENT1_DIR, EXPERIMENT2_DIR

TRANSLATED_LABELS = {
    "Bett": "bed",
    "Tisch": "table",
    "Schaf": "sheep",
    "Fisch": "fish",
    "bauen": "build",
    "kehren": "sweep",
    "sichten": "sight",
    "scheren": "shear",
}

TARGET_WORDS = [
    "Tisch_1",
    "Herd_1",
    "Haus_1",
    "Saal_1",
    "Bett_1",
    "Wand_1",
    "Tür_1",
    "Topf_1",
    "Fisch_1",
    "Pferd_1",
    "Maus_1",
    "Wal_1",
    "Schaf_1",
    "Kuh_1",
    "Reh_1",
    "Wolf_1",
    "kehren_1",
    "streichen_1",
    "putzen_1",
    "wischen_1",
    "bauen_1",
    "machen_1",
    "öffnen_1",
    "spülen_1",
    "scheren_1",
    "streicheln_1",
    "nutzen_1",
    "fischen_1",
    "sichten_1",
    "melken_1",
    "jagen_1",
    "fangen_1",
    "Fliese_1",
    "Liege_1",
    "Schale_1",
    "Stufe_1",
    "Teppich_1",
    "Sofa_1",
    "Dose_1",
    "Blume_1",
    "Fliege_1",
    "Ziege_1",
    "Schabe_1",
    "Stute_1",
    "Biene_1",
    "Lama_1",
    "Zecke_1",
    "Zebra_1",
    "weglegen_1",
    "anpassen_1",
    "einräumen_1",
    "betreten_1",
    "verlegen_1",
    "verrücken_1",
    "zudrehen_1",
    "abschneiden_1",
    "wegfegen_1",
    "einzäunen_1",
    "verscheuchen_1",
    "anbeten_1",
    "verjagen_1",
    "versorgen_1",
    "abstreifen_1",
    "anfassen_1",
    "Vieh_1",
    "Schwein_1",
    "Hahn_1",
    "Hund_1",
    "Hai_1",
    "Bär_1",
    "Fuchs_1",
    "Gans_1",
    "Knie_1",
    "Bein_1",
    "Zahn_1",
    "Mund_1",
    "Arm_1",
    "Hand_1",
    "Kopf_1",
    "Hals_1",
    "meiden_1",
    "retten_1",
    "schießen_1",
    "sehen_1",
    "treiben_1",
    "essen_1",
    "rufen_1",
    "hören_1",
    "reiben_1",
    "recken_1",
    "schließen_1",
    "drehen_1",
    "ziehen_1",
    "dehnen_1",
    "heben_1",
    "strecken_1",
    "Katze_1",
    "Eber_1",
    "Made_1",
    "Hase_1",
    "Löwe_1",
    "Affe_1",
    "Taube_1",
    "Tiger_1",
    "Glatze_1",
    "Leber_1",
    "Wade_1",
    "Nase_1",
    "Finger_1",
    "Zehe_1",
    "Auge_1",
    "Rücken_1",
    "verschenken_1",
    "erlegen_1",
    "auftischen_1",
    "erschießen_1",
    "behandeln_1",
    "vertreiben_1",
    "bewundern_1",
    "entfernen_1",
    "verrenken_1",
    "erheben_1",
    "abwischen_1",
    "verschließen_1",
    "entnehmen_1",
    "massieren_1",
    "anstoßen_1",
    "rasieren_1",
    "Tisch_2",
    "Herd_2",
    "Haus_2",
    "Saal_2",
    "Bett_2",
    "Wand_2",
    "Tür_2",
    "Topf_2",
    "Fisch_2",
    "Pferd_2",
    "Maus_2",
    "Wal_2",
    "Schaf_2",
    "Kuh_2",
    "Reh_2",
    "Wolf_2",
    "kehren_2",
    "streichen_2",
    "putzen_2",
    "wischen_2",
    "bauen_2",
    "machen_2",
    "öffnen_2",
    "spülen_2",
    "scheren_2",
    "streicheln_2",
    "nutzen_2",
    "fischen_2",
    "sichten_2",
    "melken_2",
    "jagen_2",
    "fangen_2",
    "Fliese_2",
    "Liege_2",
    "Schale_2",
    "Stufe_2",
    "Teppich_2",
    "Sofa_2",
    "Dose_2",
    "Blume_2",
    "Fliege_2",
    "Ziege_2",
    "Schabe_2",
    "Stute_2",
    "Biene_2",
    "Lama_2",
    "Zecke_2",
    "Zebra_2",
    "weglegen_2",
    "anpassen_2",
    "einräumen_2",
    "betreten_2",
    "verlegen_2",
    "verrücken_2",
    "zudrehen_2",
    "abschneiden_2",
    "wegfegen_2",
    "einzäunen_2",
    "verscheuchen_2",
    "anbeten_2",
    "verjagen_2",
    "versorgen_2",
    "abstreifen_2",
    "anfassen_2",
    "Vieh_2",
    "Schwein_2",
    "Hahn_2",
    "Hund_2",
    "Hai_2",
    "Bär_2",
    "Fuchs_2",
    "Gans_2",
    "Knie_2",
    "Bein_2",
    "Zahn_2",
    "Mund_2",
    "Arm_2",
    "Hand_2",
    "Kopf_2",
    "Hals_2",
    "meiden_2",
    "retten_2",
    "schießen_2",
    "sehen_2",
    "treiben_2",
    "essen_2",
    "rufen_2",
    "hören_2",
    "reiben_2",
    "recken_2",
    "schließen_2",
    "drehen_2",
    "ziehen_2",
    "dehnen_2",
    "heben_2",
    "strecken_2",
    "Katze_2",
    "Eber_2",
    "Made_2",
    "Hase_2",
    "Löwe_2",
    "Affe_2",
    "Taube_2",
    "Tiger_2",
    "Glatze_2",
    "Leber_2",
    "Wade_2",
    "Nase_2",
    "Finger_2",
    "Zehe_2",
    "Auge_2",
    "Rücken_2",
    "verschenken_2",
    "erlegen_2",
    "auftischen_2",
    "erschießen_2",
    "behandeln_2",
    "vertreiben_2",
    "bewundern_2",
    "entfernen_2",
    "verrenken_2",
    "erheben_2",
    "abwischen_2",
    "verschließen_2",
    "entnehmen_2",
    "massieren_2",
    "anstoßen_2",
    "rasieren_2",
]
SEMANTIC_CAT = [
    "house",
    "house",
    "house",
    "house",
    "house",
    "house",
    "house",
    "house",
    "animal",
    "animal",
    "animal",
    "animal",
    "animal",
    "animal",
    "animal",
    "animal",
    "house",
    "house",
    "house",
    "house",
    "house",
    "house",
    "house",
    "house",
    "animal",
    "animal",
    "animal",
    "animal",
    "animal",
    "animal",
    "animal",
    "animal",
    "house",
    "house",
    "house",
    "house",
    "house",
    "house",
    "house",
    "house",
    "animal",
    "animal",
    "animal",
    "animal",
    "animal",
    "animal",
    "animal",
    "animal",
    "house",
    "house",
    "house",
    "house",
    "house",
    "house",
    "house",
    "house",
    "animal",
    "animal",
    "animal",
    "animal",
    "animal",
    "animal",
    "animal",
    "animal",
    "animal",
    "animal",
    "animal",
    "animal",
    "animal",
    "animal",
    "animal",
    "animal",
    "body",
    "body",
    "body",
    "body",
    "body",
    "body",
    "body",
    "body",
    "animal",
    "animal",
    "animal",
    "animal",
    "animal",
    "animal",
    "animal",
    "animal",
    "body",
    "body",
    "body",
    "body",
    "body",
    "body",
    "body",
    "body",
    "animal",
    "animal",
    "animal",
    "animal",
    "animal",
    "animal",
    "animal",
    "animal",
    "body",
    "body",
    "body",
    "body",
    "body",
    "body",
    "body",
    "body",
    "animal",
    "animal",
    "animal",
    "animal",
    "animal",
    "animal",
    "animal",
    "animal",
    "body",
    "body",
    "body",
    "body",
    "body",
    "body",
    "body",
    "body",
    "house",
    "house",
    "house",
    "house",
    "house",
    "house",
    "house",
    "house",
    "animal",
    "animal",
    "animal",
    "animal",
    "animal",
    "animal",
    "animal",
    "animal",
    "house",
    "house",
    "house",
    "house",
    "house",
    "house",
    "house",
    "house",
    "animal",
    "animal",
    "animal",
    "animal",
    "animal",
    "animal",
    "animal",
    "animal",
    "house",
    "house",
    "house",
    "house",
    "house",
    "house",
    "house",
    "house",
    "animal",
    "animal",
    "animal",
    "animal",
    "animal",
    "animal",
    "animal",
    "animal",
    "house",
    "house",
    "house",
    "house",
    "house",
    "house",
    "house",
    "house",
    "animal",
    "animal",
    "animal",
    "animal",
    "animal",
    "animal",
    "animal",
    "animal",
    "animal",
    "animal",
    "animal",
    "animal",
    "animal",
    "animal",
    "animal",
    "animal",
    "body",
    "body",
    "body",
    "body",
    "body",
    "body",
    "body",
    "body",
    "animal",
    "animal",
    "animal",
    "animal",
    "animal",
    "animal",
    "animal",
    "animal",
    "body",
    "body",
    "body",
    "body",
    "body",
    "body",
    "body",
    "body",
    "animal",
    "animal",
    "animal",
    "animal",
    "animal",
    "animal",
    "animal",
    "animal",
    "body",
    "body",
    "body",
    "body",
    "body",
    "body",
    "body",
    "body",
    "animal",
    "animal",
    "animal",
    "animal",
    "animal",
    "animal",
    "animal",
    "animal",
    "body",
    "body",
    "body",
    "body",
    "body",
    "body",
    "body",
    "body",
]
SYNTACTIC_CAT = [
    "noun",
    "noun",
    "noun",
    "noun",
    "noun",
    "noun",
    "noun",
    "noun",
    "noun",
    "noun",
    "noun",
    "noun",
    "noun",
    "noun",
    "noun",
    "noun",
    "verb",
    "verb",
    "verb",
    "verb",
    "verb",
    "verb",
    "verb",
    "verb",
    "verb",
    "verb",
    "verb",
    "verb",
    "verb",
    "verb",
    "verb",
    "verb",
    "noun",
    "noun",
    "noun",
    "noun",
    "noun",
    "noun",
    "noun",
    "noun",
    "noun",
    "noun",
    "noun",
    "noun",
    "noun",
    "noun",
    "noun",
    "noun",
    "verb",
    "verb",
    "verb",
    "verb",
    "verb",
    "verb",
    "verb",
    "verb",
    "verb",
    "verb",
    "verb",
    "verb",
    "verb",
    "verb",
    "verb",
    "verb",
    "noun",
    "noun",
    "noun",
    "noun",
    "noun",
    "noun",
    "noun",
    "noun",
    "noun",
    "noun",
    "noun",
    "noun",
    "noun",
    "noun",
    "noun",
    "noun",
    "verb",
    "verb",
    "verb",
    "verb",
    "verb",
    "verb",
    "verb",
    "verb",
    "verb",
    "verb",
    "verb",
    "verb",
    "verb",
    "verb",
    "verb",
    "verb",
    "noun",
    "noun",
    "noun",
    "noun",
    "noun",
    "noun",
    "noun",
    "noun",
    "noun",
    "noun",
    "noun",
    "noun",
    "noun",
    "noun",
    "noun",
    "noun",
    "verb",
    "verb",
    "verb",
    "verb",
    "verb",
    "verb",
    "verb",
    "verb",
    "verb",
    "verb",
    "verb",
    "verb",
    "verb",
    "verb",
    "verb",
    "verb",
    "noun",
    "noun",
    "noun",
    "noun",
    "noun",
    "noun",
    "noun",
    "noun",
    "noun",
    "noun",
    "noun",
    "noun",
    "noun",
    "noun",
    "noun",
    "noun",
    "verb",
    "verb",
    "verb",
    "verb",
    "verb",
    "verb",
    "verb",
    "verb",
    "verb",
    "verb",
    "verb",
    "verb",
    "verb",
    "verb",
    "verb",
    "verb",
    "noun",
    "noun",
    "noun",
    "noun",
    "noun",
    "noun",
    "noun",
    "noun",
    "noun",
    "noun",
    "noun",
    "noun",
    "noun",
    "noun",
    "noun",
    "noun",
    "verb",
    "verb",
    "verb",
    "verb",
    "verb",
    "verb",
    "verb",
    "verb",
    "verb",
    "verb",
    "verb",
    "verb",
    "verb",
    "verb",
    "verb",
    "verb",
    "noun",
    "noun",
    "noun",
    "noun",
    "noun",
    "noun",
    "noun",
    "noun",
    "noun",
    "noun",
    "noun",
    "noun",
    "noun",
    "noun",
    "noun",
    "noun",
    "verb",
    "verb",
    "verb",
    "verb",
    "verb",
    "verb",
    "verb",
    "verb",
    "verb",
    "verb",
    "verb",
    "verb",
    "verb",
    "verb",
    "verb",
    "verb",
    "noun",
    "noun",
    "noun",
    "noun",
    "noun",
    "noun",
    "noun",
    "noun",
    "noun",
    "noun",
    "noun",
    "noun",
    "noun",
    "noun",
    "noun",
    "noun",
    "verb",
    "verb",
    "verb",
    "verb",
    "verb",
    "verb",
    "verb",
    "verb",
    "verb",
    "verb",
    "verb",
    "verb",
    "verb",
    "verb",
    "verb",
    "verb",
]

WORD_TO_SEMCAT = {word[:-2]: semcat for word, semcat in zip(TARGET_WORDS, SEMANTIC_CAT)}
WORD_TO_SYNCAT = {word[:-2]: pos for word, pos in zip(TARGET_WORDS, SYNTACTIC_CAT)}


def construct_spikerates_filename(
    session_id: str,
    path: str,
    bin_size: Optional[int] = None,
    blur_sd: Optional[float] = None,
    experiment: str = "experiment1",
):

    if experiment == "experiment1":
        filename = f"{path}/{session_id}_naming_spikerates"
    elif experiment == "experiment2":
        filename = f"{path}/{session_id}_spikerates"
    else:
        raise ValueError(f"Unknown experiment name '{experiment}'")

    if not (bin_size == 50 and blur_sd is None):
        if bin_size:
            filename += f"_bin_size_{bin_size}"

        if blur_sd:
            filename += f"_blur_sd_{blur_sd}"

    if experiment == "experiment1":
        filename += "_prep_and_production"

    filename += ".pkl"

    return filename


def load_spike_data(path: str, translate: bool = False):

    labels_path = path + "_labels.pkl"
    labels_words_path = path + "_labels_words.pkl"
    spikerates_path = path + "_spikerates_prep_and_production.pkl"
    spiketimes_path = path + "_spiketimes_prep_and_production.pkl"

    with open(labels_path, "rb") as file:
        labels = np.asarray(pickle.load(file))

    with open(labels_words_path, "rb") as file:
        labels_words = np.asarray(pickle.load(file))

    if translate:
        labels_words = np.asarray([TRANSLATED_LABELS[label] for label in labels_words])

    with open(spiketimes_path, "rb") as file:
        spiketimes = pickle.load(file)

    with open(spikerates_path, "rb") as file:
        spikerates = pickle.load(file)
        try:
            spikerates = np.asarray(spikerates)
        except Exception:
            spikerates = np.asarray(spikerates, dtype=object)

    return labels_words, labels, spikerates, spiketimes


def load_session_ids(data_dir: str = "data"):
    available_files = os.listdir(data_dir)
    labels_paths = [
        file for file in available_files if file.endswith("naming_labels.pkl")
    ]
    session_ids = [
        re.match(r"^\d+", labels_path).group(0) for labels_path in labels_paths
    ]
    return session_ids


def load_spikerates_all_sessions(data_dir: str = "data", task: str = "_naming"):
    available_files = os.listdir(data_dir)
    labels_paths = [
        file for file in available_files if file.endswith(task + "_labels.pkl")
    ]
    labels_words_paths = [
        file for file in available_files if file.endswith(task + "_labels_words.pkl")
    ]
    spikerates_paths = [
        file
        for file in available_files
        if file.endswith(task + "_spikerates_prep_and_production.pkl")
    ]

    session_idx = []
    labels = []
    labels_words = []
    spikerates = []

    for labels_path in labels_paths:
        session_date = re.match(r"^\d+", labels_path).group(0)

        with open(data_dir + "/" + labels_path, "rb") as file:
            session_labels = np.asarray(pickle.load(file))
            labels.extend(session_labels)
            session_idx.extend([session_date] * len(session_labels))

        with open(f"{data_dir}/{session_date}{task}_labels_words.pkl", "rb") as file:
            labels_words.extend(np.asarray(pickle.load(file)))

        with open(
            f"{data_dir}/{session_date}{task}_spikerates_prep_and_production.pkl", "rb"
        ) as file:
            spikerates.extend(pickle.load(file))

    return (
        np.asarray(session_idx),
        np.asarray(labels),
        np.asarray(labels_words),
        spikerates,
    )


def get_different_labels(labels_words):

    semcat_labels = np.asarray([WORD_TO_SEMCAT[word] for word in labels_words])
    syncat_labels = np.asarray([WORD_TO_SYNCAT[word] for word in labels_words])

    return semcat_labels, syncat_labels


def labels_words_to_numeric(labels_words):
    labels_words_dict = {word: i for i, word in enumerate(np.unique(labels_words))}
    numeric_labels_words = torch.from_numpy(
        np.asarray([labels_words_dict[word] for word in labels_words])
    ).long()
    reverse_label_word_dict = {v: str(k) for k, v in labels_words_dict.items()}
    return numeric_labels_words, reverse_label_word_dict


def load_vectors(path: str = "data/vectors.npy"):
    return np.load(path)


class SpikerateDataset(Dataset):
    """Custom pytorch datset for spikerates"""

    def __init__(self, features, labels):
        self.features = features
        self.labels = labels

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return self.features[idx], self.labels[idx]


def data_collate_sequence_first(batch):
    """
    Data collation function that returns the data in the
    format [seq_len x batch_size x ...] rather than the
    default [batch_size x seq_len x ...]
    """
    collated_batch = default_collate(batch)
    collated_batch[0] = collated_batch[0].permute(1, 0, 2)
    
    if collated_batch[1].dim() == 2:
        collated_batch[1] = collated_batch[1].permute(1, 0)

    return collated_batch

def decode_binning_params(binning_params: str,
                          binning_regex: str = r"(\d+)_size_(.+)_sd") -> dict[str, int]:
    """
    Parse a binning parameter string and return its components as a dict.
    """
    re_match = re.search(binning_regex, binning_params)
    blur_sd = re_match.group(2)

    if blur_sd == "None":
        blur_sd = None
    elif "." in blur_sd:
        blur_sd = float(blur_sd)
    else:
        blur_sd = int(blur_sd)

    return {"bin_size": int(re_match.group(1)), "blur_sd": blur_sd}
        

def load_train_test_ids(experiment: str, fold_idx: int = 0):

    if experiment == "experiment1":
        train_test_path = f'{EXPERIMENT1_DIR}/train_test_split/train_test_split.json'
        with open(train_test_path, 'r') as file:
            train_test_split = json.loads(file.read())
        current_fold_split = {session_id: folds[fold_idx] for session_id, folds in train_test_split.items()}
        return current_fold_split
        
    elif experiment == "experiment2":
        train_test_path = f'{EXPERIMENT2_DIR}/train_test_split/train_test_split.json'
        with open(train_test_path, 'r') as file:
            train_test_split = json.loads(file.read())

        return {"20240708": train_test_split[fold_idx]}
    

def pickle_object(object_to_pickle, output_path):
    with open(output_path, "wb") as f:
        pickle.dump(object_to_pickle, f)

def unpickle_object(input_path):
    with open(input_path, "rb") as f:
        return pickle.load(f)