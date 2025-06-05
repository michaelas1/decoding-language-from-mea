from enum import Enum
from statistics import LinearRegression

from sklearn.discriminant_analysis import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.multioutput import MultiOutputRegressor
from sklearn.pipeline import FunctionTransformer, Pipeline
from sklearn.svm import SVC, SVR

from thesis_project.models import OutputType
from thesis_project.models.output_handler import (
    LogitOutputHandler,
    Word2VecOutputHandler,
)
from thesis_project.models.rnn_encoder_decoder import RNNEncoderDecoder
from thesis_project.models.rnn_encoder_only import RNNEncoderOnly
from thesis_project.models.transformer_encoder_decoder import TransformerEncoderDecoder
from thesis_project.models.transformer_encoder_only import TransformerEncoderOnly
from thesis_project.settings import DEFAULT_HYPERPARAMETERS, get_default_hyperparams
from thesis_project.word2vec_embeddings import download_pretrained_model


def mean_sequence(x):
    return x.mean(axis=1)


def flatten(x):
    return x.reshape(x.shape[0])


class ModelFactory:

    """
    Factory class to create models of different types.
    """

    def __init__(
        self,
        model_type: str,
        task_type: str,
        hyperparams: dict,
        device: str = "cuda",
        n_labels: int = None,
        word_label_dict: dict = None,
        embedding_name: str = None,
    ):

        self.model_type = model_type
        self.task_type = task_type
        self.hyperparams = hyperparams
        self.device = device
        self.n_labels = n_labels

        if task_type == "seq_clf":
            self.output_handler = LogitOutputHandler()

        elif task_type == "seq_reg":
            w2v_model = download_pretrained_model(embedding_name)
            self.output_handler = Word2VecOutputHandler(
                w2v_model, word_label_dict, device=self.device
            )

        self.hyperparams = {**get_default_hyperparams(model_type, task_type), **self.hyperparams}


    def create_model(self):
        if self.model_type == "logistic_regression":
            return self.create_logistic_regression()
        if self.model_type == "svm":
            return self.create_svm()
        elif self.model_type == "rnn" and self.task_type in ["clf", "reg"]:
            return self.create_rnn_encoder()
        elif self.model_type == "trf" and self.task_type in ["clf", "reg"]:
            return self.create_trf_encoder()
        elif self.model_type == "rnn" and self.task_type in ["seq_clf", "seq_reg"]:
            return self.create_rnn_encoder_decoder()
        elif self.model_type == "trf" and self.task_type in ["seq_clf", "seq_reg"]:
            return self.create_trf_encoder_decoder()

    # --- SVM CLF ---
    def create_svm(self):

        prefix = "" if self.task_type == "clf" else "estimator__"

        gamma = self.hyperparams.get(f"{prefix}gamma")
        C = self.hyperparams.get(f"{prefix}C")
        kernel = self.hyperparams.get(f"{prefix}kernel")

        preprocessing_method = self.hyperparams.get("preprocessing_method")

        if isinstance(gamma, str) and gamma.replace(".", "").isnumeric():
            gamma = float(gamma)

        if isinstance(C, str) and C.replace(".", "").isnumeric():
            C = float(C)

        if self.task_type == "clf":
            svm_stage = ("svc", SVC(kernel=kernel, gamma=gamma, C=C, probability=True))
        else:
            svm_stage = (
                "svr",
                MultiOutputRegressor(SVR(kernel=kernel, gamma=gamma, C=C), n_jobs=-1),
            )

        if preprocessing_method == "mean_sequence":
            preprocessing = ("mean_sequence", FunctionTransformer(mean_sequence))
        elif preprocessing_method == "flatten":
            preprocessing = ("flatten", FunctionTransformer(flatten))

        return Pipeline(steps=[preprocessing, ("scaler", StandardScaler()), svm_stage])
    

    def create_logistic_regression(self):
        preprocessing_method = self.hyperparams.get("preprocessing_method")

        if preprocessing_method == "mean_sequence":
            preprocessing = ("mean_sequence", FunctionTransformer(mean_sequence))
        elif preprocessing_method == "flatten":
            preprocessing = ("flatten", FunctionTransformer(flatten))

        regression_stage = ("logistic_regression", LogisticRegression())

        return Pipeline(steps=[preprocessing, ("scaler", StandardScaler()), regression_stage])



    # --- RNN encoder ---
    def create_rnn_encoder(self):
        return RNNEncoderOnly(output_size=self.n_labels, **self.hyperparams)

    ### --- TRF encoder ---
    def create_trf_encoder(self):
        return TransformerEncoderOnly(
            output_size=self.n_labels,
            output_type="classification" if self.task_type == "clf" else "regression",
            **self.hyperparams
        )

    ### --- RNN encoder-decoder ---
    def create_rnn_encoder_decoder(self):
        if self.task_type == "seq_clf":
            output_type = OutputType.CLASSIFICATION
            output_size = self.n_labels
            n_labels = self.n_labels
        else:
            output_type = OutputType.REGRESSION
            output_size = self.output_handler._model.vector_size
            n_labels = len(self.output_handler._word_label_dict) + 1

        self.hyperparams["n_labels"] = n_labels

        model = RNNEncoderDecoder(
            output_size=output_size,
            output_handler=self.output_handler,
            output_type=output_type,
            #n_labels=n_labels,
            **self.hyperparams
        )
        return model

    ### --- TRF encoder-decoder ---
    def create_trf_encoder_decoder(self):

        if self.task_type == "seq_clf":
            output_type = OutputType.CLASSIFICATION
            output_size = self.n_labels
            n_labels = self.n_labels
        else:
            output_type = OutputType.REGRESSION
            output_size = self.output_handler._model.vector_size
            n_labels = len(self.output_handler._word_label_dict) + 1

        model = TransformerEncoderDecoder(
            n_labels=n_labels,
            output_size=output_size,
            output_type=output_type,
            output_handler=self.output_handler,
            **self.hyperparams
        )
        return model
