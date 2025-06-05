from nltk import download, pos_tag
from nltk.corpus.reader.util import ConcatenatedCorpusView
from nltk.corpus import brown
from nltk.corpus import stopwords
import collections

from thesis_project.settings import DATA_DIR


# POS tags in the Brown corpus
# see https://varieng.helsinki.fi/CoRD/corpora/BROWN/tags.html

noun_tags = ["NN", "NN$", "NNS", "NNS$", "NP", "NPS", "NPS$", "NR", "NRS"]
verb_tags = ["VB", "VBD", "VBG", "VBN", "VBZ"]


def get_most_frequent_from_corpus(n_words: int,
                                  stoplist: list[str],
                                  corpus_words: ConcatenatedCorpusView,
                                  pos_tags: list[str] = None,
                                  extra_words: int = 1000,
                                  pos_extra_words: int = 20000):

  """Extract most frequently occurring words from a corpus, excluding stopwords.

  :param n_words: Number of words to return.
  :param stoplist: List of stopwords.
  :param corpus_words: The corpus from which the words are extracted.
  :param pos_tags: List of POS tag abbreviations that extracted words are
    	  allowed to have. If None, all POS tags are included.
  :param extra_words: Buffer to make sure that the number of initially
      extracted words is large enough to return n_words after filtering
      multi-word tokens.
  :param pos_extra_words: Buffer to make sure that the number of initially
      extracted words is large enough to return n_words after filtering
      POS-tags.
  """

  words = [word.lower() for word in corpus_words if word not in stoplist and word.isalpha()]

  if pos_tags:
      frequent_words = collections.Counter(words).most_common()[:pos_extra_words]
      frequent_words = [word for word in frequent_words if pos_tag([word[0]])[0][1] in pos_tags][:n_words + extra_words]
  else:
      frequent_words = collections.Counter(words).most_common()[:n_words + extra_words]
      frequent_words = collections.Counter(words).most_common()[:n_words + extra_words]

  #words = ['Bett', 'Fisch', 'Schaf', 'Tisch', 'bauen', 'kehren', 'scheren', 'sichten']
  frequent_words = [word[0] for word in frequent_words][:n_words]

  return frequent_words


def extract_brown_frequent_words(n_words: int = 25):
    download_list = ['stopwords', 'brown', 'averaged_perceptron_tagger']

    for item in download_list:
        download(item)

    stoplist = stopwords.words('english')
    corpus_words = brown.words()

    brown_noun_frequent_words = get_most_frequent_from_corpus(n_words, stoplist, corpus_words, pos_tags=noun_tags)
    brown_verb_frequent_words = get_most_frequent_from_corpus(n_words, stoplist, corpus_words, pos_tags=verb_tags) 

    output_path = f"{DATA_DIR}/brown_{n_words*2}_frequent_words.list"

    print(f"Writing to {output_path}...")

    print(len(brown_noun_frequent_words + brown_verb_frequent_words))

    with open(output_path, "w") as f:
       f.write(str(brown_noun_frequent_words + brown_verb_frequent_words))


if __name__ == "__main__":
    extract_brown_frequent_words(n_words=25)
