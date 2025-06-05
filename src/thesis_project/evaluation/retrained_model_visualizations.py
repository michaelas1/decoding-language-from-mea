import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
from tqdm import tqdm
from thesis_project.data_loading import decode_binning_params, load_train_test_ids
from thesis_project.evaluation.confusion_matrix import calculate_confusion_matrix
from thesis_project.evaluation.decoding_accuracies import calculate_decoding_accuracies
from thesis_project.evaluation.model_inference import ModelInference
from thesis_project.evaluation.permutation_importance import (
    calculate_permutation_importance,
)
from thesis_project.preprocessing.german_to_english import GERMAN_TO_ENGLISH_DICT
from thesis_project.preprocessing.label_preparation import (
    get_label_dict,
    prepare_spikerates_for_session,
)
from thesis_project.settings import EXPERIMENT1_DIR, EXPERIMENT2_DIR, RESULT_DIR
from thesis_project.word2vec_embeddings import download_pretrained_model


### container classes for visualization results


class TrainTestResultContainer(dict):

    def __init__(self):
        super().__init__({"train": {}, "test": {}})

    def __getitem__(self, key):
        if key in ["train", "test"]:
            return super().__getitem__(key)
        else:
            raise ValueError(f"Unknown key {key}")

    def __setitem__(self, key, value):
        if key in ["train", "test"]:
            super().__setitem__(key, value)
        else:
            raise ValueError(f"Unknown key {key}")

    def __repr__(self):
        return super().__repr__()


class LabelResultContainer(dict):
    def __init__(self, label_names):
        self.label_names = label_names
        super().__init__()
        for label_name in label_names:
            self[label_name] = TrainTestResultContainer()

    def __getitem__(self, key):
        return super().__getitem__(key)

    def __setitem__(self, key, value):
        super().__setitem__(key, value)

    def __repr__(self):
        return super().__repr__()


class VisualizationResults:

    def __init__(self, visualization_options, label_names):

        if "confusion_matrix" in visualization_options:
            self.confusion_matrix = LabelResultContainer(label_names)

        if "permutation_importance" in visualization_options:
            self.importances = LabelResultContainer(label_names)

        if "accuracy" in visualization_options:
            self.nonstrict_predictions = LabelResultContainer(label_names)
            self.strict_accuracies = LabelResultContainer(label_names)
            self.nonstrict_accuracies = LabelResultContainer(label_names)

        if "semantic_similarity" in visualization_options:
            self.sem_sim_dict = {"glove-twitter-25": {}, "german.model": {}}


###


def calculate_semantic_similarity(labels_dict, w2v_model, translate=True):

    index = [word for i, word in labels_dict.items()]
    if translate:
        index = [GERMAN_TO_ENGLISH_DICT[word] for word in index]
    else:
        index = [
            word.replace("ü", "ue")
            .replace("ä", "ae")
            .replace("ö", "oe")
            .replace("machen", "mache")
            for word in index
        ]

    word_vectors = [w2v_model[word] for word in index]
    sim_matrix = cosine_similarity(word_vectors)

    return pd.DataFrame(sim_matrix, columns=index, index=index)


def calculate_visualization_data(
    model_type: str,
    task_type: str,
    label_names: list[str],
    visualization_options: list[str],
    translate: bool = True,
) -> VisualizationResults:
    """
    Calculate different values to plot, such as confusion matrix and permutation importance.
    """
    output_dir = f"{RESULT_DIR}/final_results/retrained_models/{model_type}_{task_type}"
    result_df = pd.read_csv(f"{output_dir}/results.csv")

    if "seq" in task_type:
        experiment_name = "experiment2"
    else:
        experiment_name = "experiment1"

    # initialize output data structure
    results = VisualizationResults(visualization_options, label_names)

    data_split = load_train_test_ids(experiment_name)

    if task_type == "reg" or "semantic_similarity" in visualization_options:
        w2v_model = download_pretrained_model("glove-twitter-25")
    if "semantic_similarity" in visualization_options:
        german_model = download_pretrained_model("german.model")

    # go over sessions and calculate values for visualizations
    for session_id, split in tqdm(data_split.items()):

        # pre-emptively empty cache
        # torch.cuda.empty_cache()
        # gc.collect()

        label_dict = get_label_dict(session_id, experiment_name, task_type)
        train_idx, test_idx = split["train_ids"], split["test_ids"]

        for label_name, (labels, labels_dict) in label_dict.items():

            if label_name not in label_names:
                continue

            current_params = result_df.loc[
                (result_df["session_id"] == int(session_id))
                & (result_df["label_name"] == label_name)
            ]

            if not len(current_params):
                continue

            binning_params = decode_binning_params(
                current_params["binning_params"].values[0]
            )

            if experiment_name == "experiment1":
                spikerate_path = EXPERIMENT1_DIR
            else:
                spikerate_path = f"{EXPERIMENT2_DIR}/binned_spikerates"

            spikerates = prepare_spikerates_for_session(
                session_id=session_id,
                path=spikerate_path,
                bin_size=binning_params["bin_size"],
                blur_sd=binning_params["blur_sd"],
                experiment=experiment_name,
            )

            model_inference = ModelInference(
                model_type, task_type, session_id, label_name
            )

            # decoding accuracy
            if task_type == "reg" and "accuracy" in visualization_options:
                (
                    results.strict_accuracies[label_name]["train"][session_id],
                    results.nonstrict_accuracies[label_name]["train"][session_id],
                    train_strict_predictions,
                    results.nonstrict_predictions[label_name]["train"][session_id],
                ) = calculate_decoding_accuracies(
                    model_inference,
                    spikerates[train_idx],
                    labels[train_idx],
                    labels_dict,
                    w2v_model,
                    translate=translate,
                )

                (
                    results.strict_accuracies[label_name]["test"][session_id],
                    results.nonstrict_accuracies[label_name]["test"][session_id],
                    test_strict_predictions,
                    results.nonstrict_predictions[label_name]["test"][session_id],
                ) = calculate_decoding_accuracies(
                    model_inference,
                    spikerates[test_idx],
                    labels[test_idx],
                    labels_dict,
                    w2v_model,
                    translate=translate,
                )

            # confusion matrix
            if "confusion_matrix" in visualization_options:
                if task_type == "clf":
                    train_strict_predictions = model_inference.chunked_inference(
                        spikerates[train_idx], labels[train_idx]
                    )
                    test_strict_predictions = model_inference.chunked_inference(
                        spikerates[test_idx], labels[test_idx]
                    )
                elif "accuracy" not in visualization_options:
                    raise ValueError(
                        "regression confusion matrix requires decoding accuracies"
                    )

                train_cm = calculate_confusion_matrix(
                    labels[train_idx],
                    train_strict_predictions,
                    labels_dict,
                )

                test_cm = calculate_confusion_matrix(
                    labels[test_idx],
                    test_strict_predictions,
                    labels_dict,
                )

                results.confusion_matrix[label_name]["train"][session_id] = train_cm
                results.confusion_matrix[label_name]["test"][session_id] = test_cm

            # permutation importance
            if "permutation_importance" in visualization_options:
                train_permutation_importance = calculate_permutation_importance(
                    model_inference, spikerates[train_idx], labels[train_idx]
                )
                test_permutation_importance = calculate_permutation_importance(
                    model_inference, spikerates[test_idx], labels[test_idx]
                )
                results.importances[label_name]["train"][
                    session_id
                ] = train_permutation_importance
                results.importances[label_name]["test"][
                    session_id
                ] = test_permutation_importance

            # semantic similarity
            if (
                "semantic_similarity" in visualization_options
                and label_name == "labels_words"
            ):
                results.sem_sim_dict["glove-twitter-25"][session_id] = (
                    calculate_semantic_similarity(labels_dict, w2v_model)
                )
                results.sem_sim_dict["german.model"][session_id] = (
                    calculate_semantic_similarity(
                        labels_dict, german_model, translate=False
                    )
                )

    return results
