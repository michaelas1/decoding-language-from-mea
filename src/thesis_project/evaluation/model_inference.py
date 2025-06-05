import pickle

import numpy as np
import torch
from thesis_project.settings import RESULT_DIR


class ModelInference:
    def __init__(self, model_type, task_type, session_id, label_name, path=None):
        self.model_type = model_type
        self.task_type = task_type
        if path:
            self.path = path
        else:
            self.path = (
                f"{RESULT_DIR}/final_results/retrained_models/{model_type}_{task_type}/model_{session_id}_{label_name}.pkl"
            )
        self._load_model(session_id, label_name)

    def _load_model(self, session_id, label_name):
        with open(self.path, "rb") as file:
            self.model = pickle.load(file)

        if self.model_type == "rnn":
            if "seq" in self.task_type:
                self.model.encoder.rnn.flatten_parameters()
            else:
                self.model.rnn.flatten_parameters()

    def chunked_inference(
        self, input_dataset, targets=None, chunk_size=8, embedding_dim=25, sentence_len=29
    ):

        # svm inference
        if self.model_type == "svm":
            return self.model.predict(input_dataset)

        # torch model inference

        permuted_input = torch.from_numpy(input_dataset).permute(1, 0, 2).float().cuda()
        #permuted_input = torch.from_numpy(input_dataset).float().cuda()

        if self.task_type == "clf":
            prediction = np.zeros(permuted_input.shape[1])  # , len(labels_dict)))
            for i in range(0, permuted_input.shape[1], chunk_size):
                if self.model_type == "rnn":
                    prediction[i : i + chunk_size] = np.argmax(
                        self.model.forward(permuted_input[:, i : i + chunk_size])
                        .cpu()
                        .detach()
                        .numpy(),
                        axis=-1,
                    )
                else:
                    prediction[i : i + chunk_size] = np.argmax(
                        self.model.forward(
                            permuted_input[:, i : i + chunk_size],
                            tgt=targets[i : i + chunk_size],
                        )
                        .cpu()
                        .detach()
                        .numpy(),
                        axis=0,
                    )
        elif self.task_type == "reg":
            prediction = np.zeros((len(input_dataset), embedding_dim))
            for i in range(0, permuted_input.shape[1], chunk_size):
                if self.model_type == "rnn":
                    output_embeds = self.model.forward(
                        permuted_input[:, i : i + chunk_size]
                    )
                elif self.model_type == "trf":
                    output_embeds = self.model.forward(
                        permuted_input[:, i : i + chunk_size],
                        tgt=targets[i : i + chunk_size],
                    ).T
                prediction[i : i + chunk_size] = output_embeds.cpu().detach()

        elif self.task_type == "seq_clf":
            prediction = np.zeros((embedding_dim, targets.shape[0], sentence_len))
            for i in range(0, prediction.shape[1], chunk_size):
                forward_pass_result = self.model.forward(permuted_input[:, i:i+chunk_size],
                                                                        targets[i:i+chunk_size].T).cpu().detach()
                prediction[:,i:i+chunk_size,:] = forward_pass_result.permute(2, 1, 0)
            prediction = np.transpose(prediction, (1, 2, 0))

        return prediction
