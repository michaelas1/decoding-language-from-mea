from typing import List, Optional, Union

import gensim.downloader as api
import gensim.models.keyedvectors as word2vec
import numpy as np
import sklearn.preprocessing

from thesis_project.settings import DATA_DIR

GERMAN_W2V_PATH = f"{DATA_DIR}/embeddings/german.model"

GLOVE_TWITTER_25_PATH = "embeddings/glove-twitter-25.model"


def download_pretrained_model(name: str):

    if name == "german.model":
        return load_w2v_model()
    
    model = api.load(name)
    # model.save(name + ".model") #, encoding="utf-8")
    # model.save(name + ".model", binary=True, encoding="utf-8")
    return model


def load_w2v_model(path: str = GERMAN_W2V_PATH):
    return word2vec.load_word2vec_format(path, binary=True, encoding="utf-8")


def print_average_vectors():
    """
    Evaluation function
    """

    model = load_w2v_model("german.model")

    words = ["Bett", "Fisch", "Schaf", "Tisch", "bauen", "kehren", "scheren", "sichten"]

    print("Similarities between test words\n")
    for word in words:
        specific_most_similar = model.distances(word, words)
        # sort descending
        sort_idx = np.argsort(-specific_most_similar)
        specific_word_and_val = [
            (words[idx], specific_most_similar[idx]) for idx in sort_idx
        ]

        print(word)
        print(specific_word_and_val)
        print()

    most_similar = model.most_similar(
        model.get_mean_vector(
            ["Bett", "Fisch", "Schaf", "Tisch", "bauen", "kehren", "scheren", "sichten"]
        )
    )
    most_similar_words = [m[0] for m in most_similar]
    print("\nMost similar to averaged vector:", most_similar, "\n")

    print("Rank by centrality:", model.rank_by_centrality(words), "\n")

    print("Most similar average vector results to label words:\n")

    for word in words:
        specific_most_similar = model.distances(word, most_similar_words)
        # sort descending
        sort_idx = np.argsort(-specific_most_similar)
        specific_word_and_val = [
            (most_similar_words[idx], specific_most_similar[idx]) for idx in sort_idx
        ]

        print(word)
        print(specific_word_and_val)
        print()

    print("Most similar label words to average vector results:\n")

    for vector_result in most_similar_words:
        specific_most_similar = model.distances(vector_result, words)
        # sort descending
        sort_idx = np.argsort(-specific_most_similar)
        specific_word_and_val = [
            (words[idx], specific_most_similar[idx]) for idx in sort_idx
        ]

        print(vector_result)
        print(specific_word_and_val)
        print()


class ExtremeWordRetriever:
    def __init__(self, path: str = GERMAN_W2V_PATH):
        self.model = load_w2v_model(path)
        all_words = [self.model.index_to_key[i] for i in range(608130)]
        self.all_vectors = np.asarray([self.model[word] for word in all_words])

    def get_extreme_words_on_dimension(self, dim, top_k=5):
        # ascending
        idx = np.argsort(self.all_vectors[:, dim])

        smallest = [
            (self.model.index_to_key[idx[i]], self.all_vectors[idx[i]][dim])
            for i in range(top_k)
        ]

        # descending
        largest = [
            (self.model.index_to_key[idx[-1 - i]], self.all_vectors[idx[-1 - i]][dim])
            for i in range(top_k)
        ]

        return smallest, largest


def get_most_distinctive_dimensions(n_dims=10):
    """
    Evaluation function
    """

    model = load_w2v_model("german.model")

    words = ["Bett", "Fisch", "Schaf", "Tisch", "bauen", "kehren", "scheren", "sichten"]
    vectors = [model[word] for word in words]

    for word, vector in zip(words, vectors):
        print(word)
        for word2, vector2 in zip(words, vectors):
            if word == word2:
                continue
            diff = np.abs(vector - vector2)
            most_distinct = np.argsort(-diff)
            print(f"  distinguishing dims {word2}:{most_distinct[:n_dims]}")

        print()

    print("\nNormalized vectors\n")

    normed_vectors = model.get_normed_vectors()
    normed_model = lambda word: normed_vectors[model.key_to_index[word]]

    vectors = [normed_model(word) for word in words]

    for word, vector in zip(words, vectors):
        print(word)
        for word2, vector2 in zip(words, vectors):
            if word == word2:
                continue
            diff = np.abs(vector - vector2)
            most_distinct = np.argsort(-diff)
            print(f"  distinguishing dims {word2}:{most_distinct[:n_dims]}")

        print()


def test_vector_arithmetics():
    model = load_w2v_model("german.model")

    test = model["Prinz"] - model["Junge"] + model["Maedchen"]
    print(model.most_similar(test))
    print(model.most_similar_cosmul(test))

    test = model["London"] - model["England"] + model["Paris"]
    print(model.most_similar(test))
    print(model.most_similar_cosmul(test))

    print("\nNormalized vectors\n")
    # can also use sklearn.preprocessing.normalize instead (same result)

    normed_vectors = model.get_normed_vectors()
    normed_model = lambda word: normed_vectors[model.key_to_index[word]]

    test = normed_model("Prinz") - normed_model("Junge") + normed_model("Maedchen")
    print(model.most_similar(test))
    print(model.most_similar_cosmul(test))

    test = normed_model("London") - normed_model("England") + normed_model("Paris")
    print(model.most_similar(test))
    print(model.most_similar_cosmul(test))

    # test w/ model.evaluate_word_analogies


def calculate_w2v_embeddings(
    word_list: List[str],
    model,
    normalize: bool = False,
    output_path: Optional[str] = None,
    unknown_word_str: str = "unbekannt",
):
    if normalize:
        normed_vectors = model.get_normed_vectors()
        normed_model = lambda word: normed_vectors[model.key_to_index[word]]
    vectors = np.ndarray((len(word_list), model.vector_size))
    for i, word in enumerate(word_list):
        try:
            if normalize:
                vectors[i] = normed_model(word)
            else:
                vectors[i] = model[word]
        except KeyError as e:
            # word unknonwn by model
            if normalize:
                vectors[i] = normed_model(unknown_word_str)
            else:
                vectors[i] = model[unknown_word_str]

    if output_path:
        np.save(output_path, vectors)

    return vectors


def get_most_similar_w2v_words(
    vectors: Union[np.ndarray, List[np.ndarray]], model, num_words=1
):
    most_similar = [model.most_similar(vector, topn=num_words) for vector in vectors]
    return most_similar
