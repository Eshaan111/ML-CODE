"""Small, readable data helpers shared by the three interview projects.

The Word2Vec path deliberately reproduces the preprocessing from
``NPM/Word2Vec/main.ipynb``: lowercase -> sentence tokenize -> word tokenize ->
remove English stopwords -> keep alphabetic tokens -> top 4,999 + ``<UNK>``.
"""

from __future__ import annotations

from collections.abc import Iterable
from pathlib import Path

import nltk
import pandas as pd
from nltk.corpus import stopwords
from nltk.tokenize import sent_tokenize, word_tokenize


ROOT = Path(__file__).resolve().parents[1]
CORPUS_DIR = ROOT / "datasets" / "hp_books"
NLTK_DATA_DIR = ROOT / "datasets" / "nltk_data"
WORD2VEC_DIR = ROOT / "NPM" / "Word2Vec"


def configure_nltk() -> None:
    """Point NLTK at the project-local data downloaded during setup."""
    nltk_path = str(NLTK_DATA_DIR)
    if nltk_path not in nltk.data.path:
        nltk.data.path.insert(0, nltk_path)
    nltk.data.find("tokenizers/punkt")
    nltk.data.find("corpora/stopwords")


def book_paths(corpus_dir: Path = CORPUS_DIR) -> list[Path]:
    """Return Book1..Book7 in deterministic order and fail clearly if missing."""
    paths = sorted(corpus_dir.glob("Book*.txt"), key=lambda path: path.name)
    expected = [f"Book{i}.txt" for i in range(1, 8)]
    actual = [path.name for path in paths]
    if actual != expected:
        raise FileNotFoundError(
            f"Expected {expected} in {corpus_dir}, but found {actual}."
        )
    return paths


def tokenize_corpus(
    *,
    remove_stopwords: bool,
    corpus_dir: Path = CORPUS_DIR,
) -> list[list[str]]:
    """Tokenize all books into sentences using the repo's original rules."""
    configure_nltk()
    english_stopwords = set(stopwords.words("english")) if remove_stopwords else set()
    tokenized_sentences: list[list[str]] = []

    for path in book_paths(corpus_dir):
        text = " ".join(path.read_text(encoding="utf-8").split()).lower()
        for sentence in sent_tokenize(text):
            tokens = [
                token
                for token in word_tokenize(sentence)
                if token.isalpha()
                and (not remove_stopwords or token not in english_stopwords)
            ]
            tokenized_sentences.append(tokens)

    return tokenized_sentences


def build_vocabulary(
    tokenized_sentences: Iterable[list[str]],
    vocab_size: int = 5000,
) -> tuple[dict[str, int], dict[int, str]]:
    """Build ``<UNK>`` + most frequent words like the original notebook."""
    all_words = [word for sentence in tokenized_sentences for word in sentence]
    frequent_words = (
        pd.Series(all_words).value_counts().head(vocab_size - 1).index.tolist()
    )
    vocabulary = ["<UNK>", *frequent_words]
    word_to_index = {word: index for index, word in enumerate(vocabulary)}
    index_to_word = {index: word for word, index in word_to_index.items()}
    return word_to_index, index_to_word


def encode_sentences(
    tokenized_sentences: Iterable[list[str]],
    word_to_index: dict[str, int],
) -> list[list[int]]:
    """Replace words by integer IDs, mapping missing words to ``<UNK>``."""
    unknown = word_to_index["<UNK>"]
    return [
        [word_to_index.get(word, unknown) for word in sentence]
        for sentence in tokenized_sentences
    ]


def make_chunks(
    tokenized_sentences: Iterable[list[str]],
    word_to_index: dict[str, int],
    chunk_size: int = 50,
) -> list[list[int]]:
    """Flatten the corpus and divide it into fixed-length language-model chunks."""
    unknown = word_to_index["<UNK>"]
    token_ids = [
        word_to_index.get(word, unknown)
        for sentence in tokenized_sentences
        for word in sentence
    ]
    # Drop the incomplete tail so every DataLoader batch has a stable shape.
    usable = len(token_ids) - (len(token_ids) % chunk_size)
    return [
        token_ids[start : start + chunk_size]
        for start in range(0, usable, chunk_size)
    ]

