import copy
from matplotlib import pyplot as plt
from matplotlib.patches import Patch
import numpy as np
from sklearn.inspection import permutation_importance
import torch
from tqdm import tqdm

from thesis_project.settings import RESULT_DIR

BRAIN_AREA_DICT = {
"AG": [145, 150, 18, 2, 152, 20, 149, 25, 140, 146, 22, 8, 148, 24, 144, 28, 134, 141, 154, 14, 143, 27, 138, 30, 128, 135, 151, 19, 137, 29, 132, 31, 0, 129, 147, 23, 131, 159, 4, 191, 6, 1, 142, 26, 3, 158, 10, 188, 12, 7, 136, 157, 9, 156, 16, 185, 17, 13, 130, 155, 15, 153, 21, 181],
"SMG": [177, 45, 184, 37, 189, 161, 58, 166, 173, 49, 180, 40, 186, 33, 61, 163, 169, 53, 176, 44, 183, 36, 220, 160, 139, 57, 172, 48, 179, 39, 216, 32, 133, 60, 168, 52, 175, 43, 182, 35, 5, 63, 165, 56, 171, 47, 178, 38, 11, 190, 162, 59, 167, 51, 174, 42, 41, 187, 34, 62, 164, 55, 170, 46],
"MFG": [50, 207, 76, 214, 69, 223, 194, 91, 54, 204, 79, 211, 73, 219, 66, 95, 88, 200, 82, 208, 77, 215, 70, 255, 92, 196, 85, 205, 80, 212, 74, 253, 221, 192, 89, 201, 83, 209, 78, 250, 217, 64, 93, 197, 86, 206, 81, 246, 213, 68, 222, 193, 90, 202, 84, 241, 210, 72, 218, 65, 94, 198, 87, 236],
"IFG": [203, 118, 237, 114, 233, 120, 98, 235, 199, 122, 232, 119, 229, 124, 102, 231, 195, 125, 228, 123, 225, 249, 106, 227, 67, 127, 224, 126, 97, 244, 111, 99, 71, 254, 96, 252, 101, 239, 116, 103, 75, 251, 100, 248, 105, 234, 121, 107, 108, 247, 104, 243, 110, 230, 245, 112, 113, 242, 109, 238, 115, 226, 240, 117]
}

BRAIN_COLOR_DICT = {"AG": "r", "SMG": "b", "MFG": "g", "IFG": "y"}


# SVM
def calculate_svm_permutation_importance(model, spikerates, labels):
    print("n_repeats")
    transformed_input = model.steps[0][1].transform(spikerates)

    if model.steps[2][1].kernel == "linear":
        importances = model.steps[2][1]._coef
    else:
        importances = permutation_importance(
            model.steps[2][1], transformed_input, labels
        )

    return importances


# NN models
def calculate_torch_permutation_importance(
    model_inference, spikerates, labels, permutation_axis="channel"
):

    if permutation_axis == "channel":
        axis_numeric = 2

    elif permutation_axis == "timestep":
        axis_numeric = 0
        raise NotImplementedError()

    # baseline accuracy for comparison
    y_pred_test = model_inference.chunked_inference(spikerates, labels)
    baseline_accuracy = torch.sum(y_pred_test == labels) / len(labels)
    perm_importance = np.zeros(spikerates.shape[axis_numeric])
    for current_feature_idx in tqdm(range(spikerates.shape[axis_numeric])):

        # shuffle channel or timestep dimension
        shuffled_spikerates = copy.deepcopy(spikerates)

        if permutation_axis == "channel":
            feature_dimension = shuffled_spikerates[:, :, current_feature_idx]
            feature_dimension = feature_dimension[
                :, torch.randperm(spikerates.shape[1])
            ]
            shuffled_spikerates[:, :, current_feature_idx] = feature_dimension

        # evaluate model
        y_pred_compare = model_inference.chunked_inference(shuffled_spikerates, labels)
        comparison_accuracy = torch.sum(y_pred_compare == labels) / len(labels)

        # calculate permutation importance for current dimension
        perm_importance[current_feature_idx] = baseline_accuracy - comparison_accuracy

    return perm_importance


def calculate_permutation_importance(model_inference, spikerates, labels):
    if model_inference.model_type == "svm":
        return calculate_svm_permutation_importance(
            model_inference.model, spikerates, labels
        )
    else:
        return calculate_torch_permutation_importance(
            model_inference, spikerates, labels
        )


def get_permutation_importance_metadata():
    idx_color_list = [0] * 256
    for i in range(len(idx_color_list)):
        for brain_area, area_idx in BRAIN_AREA_DICT.items():
            if i in area_idx:
                idx_color_list[i] = BRAIN_COLOR_DICT[brain_area]
                break

    feature_names = [str(i) for i in list(range(256))]
    features = np.array(feature_names)
    return features, idx_color_list


def plot_permutation_importance(
    model_type,
    task_type,
    visualization_dict,
    label_name: str = "labels_words",
    input_type: str = "test",
    savefig: bool = False,
):
    fig, axs = plt.subplots(2, 3, figsize=(50, 30))
    flattened_axs = axs.flatten()
    fig.supxlabel("Permutation Importance")
    fig.supylabel("Electrode ID")

    sorted_session_ids = [
        str(session_id)
        for session_id in sorted(
            [
                int(session_id)
                for session_id in visualization_dict.importances[label_name][
                    input_type
                ]
            ]
        )
    ]

    features, idx_color_list = get_permutation_importance_metadata()

    for i, session_id in enumerate(sorted_session_ids):

        importances = visualization_dict.importances[label_name][input_type][session_id]
        if model_type == "svm":
            importances = importances["importances_mean"]
        sorted_idx = importances.argsort()
        non_null_idx = np.asarray(
            [i for i, imp in enumerate(importances[sorted_idx]) if imp]
        )

        if not len(non_null_idx):
            continue

        flattened_axs[i].barh(
            features[sorted_idx][non_null_idx],
            importances[sorted_idx][non_null_idx],
            color=np.asarray(idx_color_list)[sorted_idx][non_null_idx],
        )
        flattened_axs[i].set_title(session_id)

        handles = [
            Patch(color=color, label=label) for label, color in BRAIN_COLOR_DICT.items()
        ]
        flattened_axs[i].legend(handles=handles)

    plt.tight_layout()
    if savefig:
        output_path = f"{RESULT_DIR}/final_results/retrained_models/{model_type}_{task_type}/{model_type}_{task_type}_perm_importance_{label_name}_{input_type}.svg"
        fig.savefig(output_path)
