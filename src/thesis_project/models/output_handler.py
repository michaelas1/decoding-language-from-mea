from abc import ABC
from typing import Dict, Optional
import torch
import logging
import numpy as np

from thesis_project.word2vec_embeddings import get_most_similar_w2v_words
from torchmetrics.functional import pairwise_cosine_similarity

logger = logging.getLogger(__name__)

class OutputHandler(ABC):
    """
    Interface to inject a strategy for the
    decoding of output logits from the last layer
    of a decoder, producing the next word for
    auto-regressive generation.
    """
    def decode_logits(logits: torch.Tensor):
        raise NotImplementedError("Abstract base class")


class Word2VecOutputHandler(OutputHandler):
    """
    OutputHandler for regression, which returns
    index of most similar word as output.
    """
    def __init__(self,
                 model,
                 word_label_dict: dict[str, int],
                 device: str = "cuda",
                 nonstrict_words: Optional[list[str]] = None) -> None:
        """
        :param model: the word2vec model
        :param word_label_dict: dict mapping words to class labels
        :param device: "cuda" or "cpu"
        :param nonstrict_words: optional list of words that nonstrict decoding will be limited to
        """

        super().__init__()
        self._model = model
        self._word_label_dict = word_label_dict

        self._model_matrix = torch.from_numpy(self._model.vectors).float().to(device)
        self._model_matrix.requires_grad = False

        self._labels_to_matrix_idx = {}

        for k, v in self._word_label_dict.items():
            index = self._model.key_to_index.get(k.lower())
            if index:
                self._labels_to_matrix_idx[v] = index

        self._matrix_idx_to_labels = {v: k for k, v in self._labels_to_matrix_idx.items()}

        self.unknown_idx = len(self._word_label_dict)
        self._word_label_dict["UNK"] = self.unknown_idx
        self.matrix_unknown_idx = 0

        if nonstrict_words:
            self._nonstrict_idx = [self._model.key_to_index.get(w.lower()) for w in nonstrict_words]
            self._nonstrict_idx = torch.from_numpy(np.asarray([i for i in self._nonstrict_idx if i is not None])).to(device)


    def _word_to_label(self, word):
        
        label = self._word_label_dict.get(word)
        if label is None:
            return self._word_label_dict['UNK']
        
        
    def decode_logits(self, logits: torch.Tensor, dim=None, use_torch=True, strict=False):

        # logits = [batch_size, embedding_dim]
        if use_torch:
            similarities = pairwise_cosine_similarity(logits, self._model_matrix, reduction=None)

            if strict:
                matrix_idx = torch.from_numpy(np.asarray(list(self._matrix_idx_to_labels.keys()))).cuda()
                sub_matrix_similarities = pairwise_cosine_similarity(logits, self._model_matrix[matrix_idx])
                sub_matrix_similarities = sub_matrix_similarities.argmax(dim=-1)
                similarities = matrix_idx[sub_matrix_similarities]

                # only compare known words
                # mask = [self._labels_to_matrix_idx[i] for i in sorted(self._labels_to_matrix_idx.keys())]
                # subset_idx = torch.argmax(similarities[:,mask], dim=-1)
                # parent_idx = torch.arange(similarities.shape[1]).cuda()[mask][subset_idx]
                # similarities = parent_idx
                
            else:
                if self._nonstrict_idx is not None:
                    sub_matrix_similarities = pairwise_cosine_similarity(logits, self._model_matrix[self._nonstrict_idx])
                    sub_matrix_similarities = sub_matrix_similarities.argmax(dim=-1)
                    similarities = self._nonstrict_idx[sub_matrix_similarities]
                else:
                    similarities = pairwise_cosine_similarity(logits, self._model_matrix, reduction=None)
                    similarities = similarities.argmax(dim=-1)

            words = [self._model.index_to_key[entry] for entry in similarities]
            view = similarities.flatten()
            for i, x in enumerate(view):
                view[i] = self._matrix_idx_to_labels.get(x.item(), self.unknown_idx)

            return similarities, words
                                

        else:
            most_similar = [get_most_similar_w2v_words([logit], self._model, num_words=1)[0][0][0] for logit in logits]

            words = [word for word in most_similar]
            labels = [self._word_to_label(word) for word in words]
            return torch.Tensor(labels), words
    
    
    def encode_labels(self, labels, dim=None, use_torch=True):

        if use_torch:
            #print(self._labels_to_matrix_idx)
            indices = [[self._labels_to_matrix_idx.get(key.item(), self.matrix_unknown_idx) for key in key_list] for key_list in labels]
            embedded = self._model.vectors[indices]

        else:
            # TODO: improve performance
            indices = [[self._model.index_to_key[key] for key in key_list] for key_list in labels]
            embedded = torch.from_numpy(np.array([self._model[key_list] for key_list in indices]))

        return embedded

class LogitOutputHandler(OutputHandler):
    """
    OutputHandler for classification, which samples
    from the logit distribution.
    """

    def __init__(self, label_dict=None):
        self.label_dict = label_dict

    def decode_idx(self, idx_to_decode):
        if self.label_dict is None:
            return idx_to_decode
            
        return [self.label_dict[idx.item()] for idx in idx_to_decode]

    def decode_logits(self, logits: torch.Tensor, dim=-1):
        logger.debug(("Outputs logits dim", logits.shape))
        probs = torch.softmax(logits, dim=dim)
        decoded_idx = torch.argmax(probs, dim=dim)
        return decoded_idx, self.decode_idx(decoded_idx)
    