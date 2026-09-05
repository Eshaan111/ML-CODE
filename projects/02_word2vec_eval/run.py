"""Project 2: evaluate the repo's trained CBOW Word2Vec checkpoints.

Run from the repository root:
    python projects/02_word2vec_eval/run.py
"""

from __future__ import annotations

import json
import math
import sys
from collections import defaultdict
from pathlib import Path

PROJECT_DIR = Path(__file__).resolve().parent
ROOT = PROJECT_DIR.parents[1]
sys.path.insert(0, str(ROOT))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.manifold import TSNE

from projects.data_utils import (
    WORD2VEC_DIR,
    build_vocabulary,
    encode_sentences,
    tokenize_corpus,
)


SEED = 42
VOCAB_SIZE = 5000
EMBEDDING_DIM = 50
WINDOW_SIZE = 2
EVAL_EXAMPLES = 10_000


class SimpleW2V(nn.Module):
    """The same CBOW model defined in NPM/Word2Vec/main.py."""

    def __init__(self, vocab_size: int, embedding_dim: int) -> None:
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.output = nn.Linear(embedding_dim, vocab_size)

    def forward(self, context: torch.Tensor) -> torch.Tensor:
        embedded = self.embedding(context)       # (B, 4, 50)
        context_mean = embedded.mean(dim=1)      # (B, 50)
        return self.output(context_mean)          # (B, 5000)


def make_eval_examples(
    encoded_sentences: list[list[int]],
    limit: int = EVAL_EXAMPLES,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Build a deterministic subset of real CBOW context/target examples."""
    contexts: list[list[int]] = []
    targets: list[int] = []
    for sentence in encoded_sentences:
        for center in range(WINDOW_SIZE, len(sentence) - WINDOW_SIZE):
            context = (
                sentence[center - WINDOW_SIZE : center]
                + sentence[center + 1 : center + WINDOW_SIZE + 1]
            )
            contexts.append(context)
            targets.append(sentence[center])
            if len(contexts) >= limit:
                return torch.tensor(contexts), torch.tensor(targets)
    return torch.tensor(contexts), torch.tensor(targets)


def cbow_loss(
    model: SimpleW2V,
    contexts: torch.Tensor,
    targets: torch.Tensor,
    device: torch.device,
) -> float:
    model.to(device).eval()
    total_loss = 0.0
    with torch.inference_mode():
        for start in range(0, len(contexts), 256):
            context = contexts[start : start + 256].to(device)
            target = targets[start : start + 256].to(device)
            loss = F.cross_entropy(model(context), target, reduction="sum")
            total_loss += loss.item()
    return total_loss / len(contexts)


def nearest_neighbors(
    weights: torch.Tensor,
    word: str,
    word_to_index: dict[str, int],
    index_to_word: dict[int, str],
    top_k: int = 8,
) -> list[dict[str, float | str]]:
    if word not in word_to_index:
        return []
    normalized = F.normalize(weights.float().cpu(), dim=1)
    similarities = normalized @ normalized[word_to_index[word]]
    candidates = torch.topk(similarities, top_k + 1).indices.tolist()
    return [
        {"word": index_to_word[index], "score": round(similarities[index].item(), 4)}
        for index in candidates
        if index_to_word[index] != word
    ][:top_k]


def analogy(
    weights: torch.Tensor,
    words: tuple[str, str, str],
    word_to_index: dict[str, int],
    index_to_word: dict[int, str],
    top_k: int = 5,
) -> list[dict[str, float | str]]:
    if any(word not in word_to_index for word in words):
        return []
    a, b, c = words
    normalized = F.normalize(weights.float().cpu(), dim=1)
    vector = normalized[word_to_index[a]] - normalized[word_to_index[b]] + normalized[word_to_index[c]]
    vector = F.normalize(vector.unsqueeze(0), dim=1).squeeze(0)
    similarities = normalized @ vector
    candidates = torch.topk(similarities, top_k + 3).indices.tolist()
    excluded = set(words)
    return [
        {"word": index_to_word[index], "score": round(similarities[index].item(), 4)}
        for index in candidates
        if index_to_word[index] not in excluded
    ][:top_k]


WORD_GROUPS = {
    "characters": [
        "harry", "ron", "hermione", "dumbledore", "voldemort", "malfoy",
        "hagrid", "snape", "sirius", "lupin", "ginny", "neville",
    ],
    "houses": ["gryffindor", "slytherin", "ravenclaw", "hufflepuff"],
    "places": ["hogwarts", "azkaban", "hogsmeade", "privet", "diagon", "ministry"],
    "magic": [
        "wand", "spell", "potion", "broom", "magic", "wizard", "witch",
        "owl", "dragon", "cloak", "dementor", "quidditch",
    ],
}


def cohesion_score(weights: torch.Tensor, word_to_index: dict[str, int]) -> float:
    """Simple transparent heuristic: average within-category cosine similarity."""
    normalized = F.normalize(weights.float().cpu(), dim=1)
    scores: list[float] = []
    for words in WORD_GROUPS.values():
        indices = [word_to_index[word] for word in words if word in word_to_index]
        for left in range(len(indices)):
            for right in range(left + 1, len(indices)):
                scores.append((normalized[indices[left]] @ normalized[indices[right]]).item())
    return float(np.mean(scores))


def create_tsne_plot(
    weights: torch.Tensor,
    word_to_index: dict[str, int],
) -> list[dict[str, float | str]]:
    records: list[tuple[str, str, int]] = []
    for category, words in WORD_GROUPS.items():
        records.extend(
            (category, word, word_to_index[word])
            for word in words
            if word in word_to_index
        )

    vectors = weights[[record[2] for record in records]].float().cpu().numpy()
    perplexity = min(10, max(2, len(records) // 4))
    coordinates = TSNE(
        n_components=2,
        perplexity=perplexity,
        init="pca",
        learning_rate="auto",
        random_state=SEED,
    ).fit_transform(vectors)

    colors = {
        "characters": "#2563eb",
        "houses": "#dc2626",
        "places": "#16a34a",
        "magic": "#9333ea",
    }
    plt.figure(figsize=(13, 9))
    for category in WORD_GROUPS:
        positions = [i for i, record in enumerate(records) if record[0] == category]
        plt.scatter(
            coordinates[positions, 0],
            coordinates[positions, 1],
            label=category.title(),
            color=colors[category],
            s=55,
            alpha=0.85,
        )
    for i, (_, word, _) in enumerate(records):
        plt.annotate(word, coordinates[i], xytext=(4, 4), textcoords="offset points", fontsize=9)
    plt.title("Harry Potter Word2Vec Embeddings (t-SNE)")
    plt.xlabel("t-SNE dimension 1")
    plt.ylabel("t-SNE dimension 2")
    plt.legend()
    plt.grid(alpha=0.15)
    plt.tight_layout()
    plt.savefig(PROJECT_DIR / "word2vec_tsne.png", dpi=180)
    plt.close()

    return [
        {
            "category": category,
            "word": word,
            "x": round(float(coordinates[i, 0]), 5),
            "y": round(float(coordinates[i, 1]), 5),
        }
        for i, (category, word, _) in enumerate(records)
    ]


def main() -> None:
    torch.manual_seed(SEED)
    np.random.seed(SEED)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    print("Tokenizing the seven books with the original stopword-free rules...")
    tokenized = tokenize_corpus(remove_stopwords=True)
    word_to_index, index_to_word = build_vocabulary(tokenized, VOCAB_SIZE)
    encoded = encode_sentences(tokenized, word_to_index)
    contexts, targets = make_eval_examples(encoded)

    assert len(word_to_index) == VOCAB_SIZE
    assert contexts.shape[1] == WINDOW_SIZE * 2
    print(f"Sentences: {len(tokenized):,}")
    print(f"Vocabulary: {len(word_to_index):,}")
    print(f"Evaluation contexts: {tuple(contexts.shape)}")
    print(f"Targets: {tuple(targets.shape)}")
    print("Top vocabulary words:", [index_to_word[i] for i in range(20)])

    checkpoint_results: dict[str, dict] = {}
    loaded_models: dict[str, SimpleW2V] = {}
    for path in sorted(WORD2VEC_DIR.glob("*.pth")):
        checkpoint = torch.load(path, map_location="cpu")
        assert checkpoint["vocab_size"] == VOCAB_SIZE
        assert checkpoint["feature_size"] == EMBEDDING_DIM
        assert checkpoint["window_size"] == WINDOW_SIZE
        model = SimpleW2V(VOCAB_SIZE, EMBEDDING_DIM)
        model.load_state_dict(checkpoint["model"])
        loss = cbow_loss(model, contexts, targets, device)
        checkpoint_results[path.name] = {
            "cbow_loss_on_first_10000_examples": round(loss, 6),
            "cbow_perplexity": round(math.exp(min(loss, 20)), 4),
        }
        loaded_models[path.name] = model.cpu()
        print(f"{path.name}: CBOW loss={loss:.4f}")

    selected_checkpoint = min(
        checkpoint_results,
        key=lambda name: checkpoint_results[name]["cbow_loss_on_first_10000_examples"],
    )
    model_names = sorted(loaded_models)
    reference_state = loaded_models[model_names[0]].state_dict()
    checkpoint_tensors_identical = all(
        torch.equal(reference_state[key], loaded_models[name].state_dict()[key])
        for name in model_names[1:]
        for key in reference_state
    )
    selected_model = loaded_models[selected_checkpoint]

    matrix_results: dict[str, dict] = {}
    for matrix_name, weights in {
        "embedding.weight": selected_model.embedding.weight.detach(),
        "output.weight": selected_model.output.weight.detach(),
    }.items():
        matrix_results[matrix_name] = {
            "category_cohesion_score": round(cohesion_score(weights, word_to_index), 6),
            "neighbors": {
                word: nearest_neighbors(weights, word, word_to_index, index_to_word)
                for word in ["harry", "ron", "hermione", "dumbledore", "voldemort", "hogwarts", "wand"]
            },
        }

    selected_matrix = max(
        matrix_results,
        key=lambda name: matrix_results[name]["category_cohesion_score"],
    )
    selected_weights = (
        selected_model.embedding.weight.detach()
        if selected_matrix == "embedding.weight"
        else selected_model.output.weight.detach()
    )

    analogy_inputs = [
        ("harry", "gryffindor", "slytherin"),
        ("dumbledore", "hogwarts", "azkaban"),
        ("harry", "ron", "hermione"),
        ("wizard", "wand", "broom"),
    ]
    analogies = {
        " - ".join(words[:2]) + " + " + words[2]: analogy(
            selected_weights, words, word_to_index, index_to_word
        )
        for words in analogy_inputs
    }
    tsne_records = create_tsne_plot(selected_weights, word_to_index)

    results = {
        "device": str(device),
        "sentences": len(tokenized),
        "vocab_size": len(word_to_index),
        "window_size": WINDOW_SIZE,
        "selected_checkpoint": selected_checkpoint,
        "selected_matrix": selected_matrix,
        "checkpoint_model_tensors_identical": checkpoint_tensors_identical,
        "checkpoint_results": checkpoint_results,
        "matrix_results": matrix_results,
        "analogies": analogies,
        "tsne_words": tsne_records,
    }
    (PROJECT_DIR / "results.json").write_text(
        json.dumps(results, indent=2), encoding="utf-8"
    )
    (PROJECT_DIR / "vocabulary.json").write_text(
        json.dumps(word_to_index, indent=2), encoding="utf-8"
    )

    print(f"Selected checkpoint: {selected_checkpoint}")
    print(f"Selected matrix: {selected_matrix}")
    print(f"Checkpoint model tensors identical: {checkpoint_tensors_identical}")
    print("Analogy results:")
    for prompt, output in analogies.items():
        print(f"  {prompt}: {[item['word'] for item in output]}")
    print(f"Saved {PROJECT_DIR / 'results.json'}")
    print(f"Saved {PROJECT_DIR / 'word2vec_tsne.png'}")


if __name__ == "__main__":
    main()
