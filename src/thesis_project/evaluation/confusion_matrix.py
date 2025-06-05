from matplotlib import pyplot as plt
import pandas as pd
from sklearn.metrics import confusion_matrix
import seaborn as sns

from thesis_project.settings import RESULT_DIR


def calculate_confusion_matrix(labels, predictions, labels_dict):

    cm = confusion_matrix(labels, predictions)
    df_cm = pd.DataFrame(
        cm,
        columns=[word for i, word in labels_dict.items()],
        index=[word for i, word in labels_dict.items()],
    )
    return df_cm


def plot_confusion_matrix(
    model_type: str,
    task_type: str,
    visualization_dict: dict,
    label_name: str = "syncat_labels",
    input_type: str = "test",
    savefig: bool = False,
):

    plt.rcParams.update({"font.size": 16})

    fig, axs = plt.subplots(2, 3, figsize=(20, 12))
    flattened_axs = axs.flatten()

    sorted_session_ids = [
        str(session_id)
        for session_id in sorted(
            [
                int(session_id)
                for session_id in visualization_dict.confusion_matrix[label_name][
                    input_type
                ]
            ]
        )
    ]

    for i, (session_id) in enumerate(sorted_session_ids):
        matrix = visualization_dict.confusion_matrix[label_name][input_type][
            session_id
        ]
        g = sns.heatmap(matrix, annot=True, ax=flattened_axs[i])
        g.set_yticklabels(g.get_yticklabels(), rotation=0)
        g.tick_params(labelsize=16)
        g.set_title(session_id, fontsize=16)

    plt.tight_layout()
    if savefig:
        output_path = f"{RESULT_DIR}/final_results/retrained_models/{model_type}_{task_type}/{model_type}_{task_type}_cm_{label_name}_{input_type}.svg"
        fig.savefig(output_path)
