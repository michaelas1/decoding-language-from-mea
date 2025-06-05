import json

import torch
from thesis_project.data_loading import TARGET_WORDS
from thesis_project.models.output_handler import Word2VecOutputHandler
from thesis_project.preprocessing.german_to_english import GERMAN_TO_ENGLISH_DICT
from thesis_project.settings import DATA_DIR


def calculate_decoding_accuracies(
    model_inference, spikerates, labels, labels_dict, w2v_model, translate=True
):

    if translate:
        labels_dict = {k: GERMAN_TO_ENGLISH_DICT[v] for k, v in labels_dict.items()}

    # load list of decoding options for nonstrict decoding
    with open(f"{DATA_DIR}/brown_50_frequent_words.list") as f:
        # print(f.read())
        brown_50_words = json.loads(f.read().replace("'", '"'))
        target_words = list(set(GERMAN_TO_ENGLISH_DICT[t[:-2]] for t in TARGET_WORDS))
        brown_50_words.extend(target_words)

    output_handler = Word2VecOutputHandler(
        w2v_model,
        {v: k for k, v in labels_dict.items()},
        nonstrict_words=brown_50_words,
    )

    # predict embeddings and decode classes
    decoded_embeddings = model_inference.chunked_inference(
        spikerates, labels
    )  # .permute(1, 0, 2)

    pred_classes_strict = (
        output_handler.decode_logits(
            torch.from_numpy(decoded_embeddings).float().cuda(),
            use_torch=True,
            strict=True,
        )[0]
        .cpu()
        .detach()
    )
    output_classes = output_handler.decode_logits(
        torch.from_numpy(decoded_embeddings).float().cuda(),
        use_torch=True,
        strict=False,
    )
    actual_words = [labels_dict[label.item()] for label in labels]

    label_nonstrict_predictions = [
        (actual, predicted)
        for actual, predicted in zip(actual_words, output_classes[1])
    ]
    pred_classes_nonstrict = output_classes[0].cpu().detach()

    # calculate accuracies
    strict_acc = sum(labels == pred_classes_strict) / len(labels)
    nonstrict_acc = sum(labels == pred_classes_nonstrict) / len(labels)

    if model_inference.model_type != "svm":
        strict_acc = strict_acc.item()
        nonstrict_acc = nonstrict_acc.item()

    return strict_acc, nonstrict_acc, pred_classes_strict, label_nonstrict_predictions
