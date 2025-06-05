import json
from re import split
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch


class SingleWordTokenizer:
    """Tokenizer which assigns an index to each individual word."""

    DEFAULT_START_TOKEN_IDX = 0
    DEFAULT_END_TOKEN_IDX = 1
    DEFAULT_PAD_TOKEN_IDX = 2
    
    def __init__(self,
                 token_dict: Optional[Dict] = None,
                 start_token_idx: Optional[int] = None,
                 end_token_idx: Optional[int] = None,
                 pad_token_idx: Optional[int] = None):
         
        self.token_dict = token_dict

        if self.token_dict:
            self.reverse_token_dict = {v: k for k, v in self.token_dict.items()}
        self.start_token_idx = start_token_idx if start_token_idx else self.DEFAULT_START_TOKEN_IDX
        self.end_token_idx = end_token_idx if end_token_idx else self.DEFAULT_END_TOKEN_IDX
        self.pad_token_idx = pad_token_idx if pad_token_idx else self.DEFAULT_PAD_TOKEN_IDX

    
    @property
    def n_labels(self):
        return len(self.token_dict)

    def token_dict_to_file(self, token_dict_path: str):
        """
        Save token dictionary in a JSON file.
        """
        with open(token_dict_path, "w") as file:
            file.write(json.dumps(self.token_dict, indent=4))


    def token_dict_from_file(self, token_dict_path: str):
        """
        Load token dictionary from a JSON file.
        """
        with open(token_dict_path, "r") as file:
            self.token_dict = json.loads(file.read())

        self.reverse_token_dict = {v: k for k, v in self.token_dict.items()}



    def token_dict_from_samples(self, samples: List[str]):
        """
        Create a token dictionary by assigning an index to each word in the list of text
        samples. Ignores capitalization.
        """

        split_samples = [token.lower() for sample in samples for token in split(" |-", sample.replace(".", ""))]
        tokens = list(set(split_samples))
        if "" in tokens:
            tokens.remove("")

        # TODO: fix this for other start/end token idx
        token_dict = {token: i + 2 for i, token in enumerate(tokens)}

        token_dict["<start>"] = self.start_token_idx
        token_dict["<end>"] = self.end_token_idx
        token_dict["<pad>"] = self.pad_token_idx

        self.token_dict = token_dict


    def tokenize_sample(self, sample: str) -> List[int]:
        """
        Tokenize a text sample.
        """
        split_sample = split(" |-", sample.lower().replace(".", ""))

        while '' in split_sample:
            split_sample.remove('')
            
        tokenized_sample = [self.start_token_idx] + \
                           [self.token_dict[token] for token in split_sample] + \
                           [self.end_token_idx]
        
        return tokenized_sample


    def tokenize_samples(self, samples: List[str]) -> Tuple[np.ndarray, Dict]:
        """Tokenize text samples and pad them to the same length."""

        index_to_token = {i: token for token, i in self.token_dict.items()}

        labels = [self.tokenize_sample(sample) for sample in samples]
        
        max_length = max([len(sentence) for sentence in labels])
        extended_labels = np.zeros((len(labels), max_length))

        for i, sample in enumerate(labels):

            if len(sample) < max_length:
                sample = sample + [self.pad_token_idx] * (max_length - len(sample))
            extended_labels[i] = sample

        return extended_labels, index_to_token
    
    def decode(self, encoded_samples):
        texts = []
        for sample in encoded_samples:
            texts.append([self.reverse_token_dict.get(token) for token in sample])

        return texts


# # requires huggingface installation
# def tokenize_with_bert(sentences):
#     tokenizer = AutoTokenizer.from_pretrained("dbmdz/bert-base-german-uncased")
#     tokenized_sentences = tokenizer(sentences, padding=True)
#     extended_labels = torch.Tensor(tokenized_sentences['input_ids'])
#     n_labels = extended_labels.unique().shape[0]
