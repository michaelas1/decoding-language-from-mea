from os import environ
from dotenv import load_dotenv

load_dotenv()
PROJECT_DIR = environ["PROJECT_DIR"]

DATA_DIR = f"{PROJECT_DIR}/data"
EXPERIMENT1_DIR = f"{DATA_DIR}/neuro_data/experiment1"
EXPERIMENT2_DIR = f"{DATA_DIR}/neuro_data/experiment2"

RESULT_DIR = f"{PROJECT_DIR}/results"
FIGURE_DIR = f"{RESULT_DIR}/figures"
TENSORBOARD_DIR = f"{PROJECT_DIR}/tensorboard"
EMBEDDING_DIR = "thesis_project/embeddings"


DEFAULT_HYPERPARAMETERS = {
    "svm": {"C": 1.0, "gamma": "scale", "kernel": "rbf"},  # based on SVM default values
    "rnn_encoder_only": {  # based on Justin Jude
        "hidden_size": 256,
        "n_layers": 1,
        "dropout": 0.1,
    },
    "rnn_encoder_decoder": {  # based on Justin Jude
        "encoder_hidden_size": 512,
        "decoder_hidden_size": 256,
        "encoder_n_layers": 3,
        "decoder_n_layers": 1,
        "encoder_dropout": 0.8,
        "decoder_dropout": 0.1
    },
    "trf_encoder_only": {
        "hidden_size": 128,
        "n_layers": 1,
        "dropout": 0.1,
        "n_heads": 8,
    },
    "trf_encoder_decoder": {  # fix this later
        "hidden_size": 128,
        "encoder_n_layers": 1,
        "decoder_n_layers": 1,
        "dropout": 0.1
    },
}

def get_default_hyperparams(model_name: str,
                            task_name: str):
    if model_name == "svm" and task_name == "clf":
        return DEFAULT_HYPERPARAMETERS["svm"]
    if model_name =="svm" and task_name == "reg":
        return {"estimator__" + k: v for k, v in DEFAULT_HYPERPARAMETERS["svm"].items()}
    if model_name == "rnn" and task_name in ("reg", "clf"):
        return DEFAULT_HYPERPARAMETERS["rnn_encoder_only"]
    if model_name == "trf" and task_name in ("reg", "clf"):
        return DEFAULT_HYPERPARAMETERS["trf_encoder_only"]
    if model_name == "rnn":
        return DEFAULT_HYPERPARAMETERS["rnn_encoder_decoder"]
    if model_name == "trf":
        return DEFAULT_HYPERPARAMETERS["trf_encoder_decoder"]