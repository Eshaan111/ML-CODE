"""Train the Project 1 winner on a stopword-inclusive vocabulary.

Why retrain?  The faithful Project 1 data removes words such as ``the`` and
``at``.  A readable generator needs those words, so this script builds a new
language-model vocabulary while transferring every compatible Word2Vec vector.

Run:
    python projects/03_text_generation/train_generator.py --epochs 5
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import math
import random
import sys
import time
from pathlib import Path

PROJECT_DIR = Path(__file__).resolve().parent
ROOT = PROJECT_DIR.parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np
import torch
import torch.nn.functional as F
from torch.optim import Adam
from torch.utils.data import DataLoader, Dataset

from projects.data_utils import WORD2VEC_DIR, build_vocabulary, make_chunks, tokenize_corpus

MODEL_FILE = ROOT / "projects" / "01_language_modeling_ladder" / "models.py"
spec = importlib.util.spec_from_file_location("ladder_models", MODEL_FILE)
models_module = importlib.util.module_from_spec(spec)
assert spec.loader is not None
spec.loader.exec_module(models_module)
MODEL_CLASSES = models_module.MODEL_CLASSES


SEED = 42
VOCAB_SIZE = 5000
EMBEDDING_DIM = 50
HIDDEN_DIM = 128
CHUNK_SIZE = 50
BATCH_SIZE = 32
LEARNING_RATE = 1e-3


class NextTokenDataset(Dataset):
    def __init__(self, chunks: list[list[int]]) -> None:
        data = torch.tensor(chunks, dtype=torch.long)
        self.context = data[:, :-1]
        self.target = data[:, 1:]

    def __len__(self) -> int:
        return len(self.context)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        return self.context[index], self.target[index]


def set_seed() -> None:
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(SEED)


def transferred_embeddings(
    lm_word_to_index: dict[str, int],
) -> tuple[torch.Tensor, int, dict[str, str]]:
    """Copy pretrained vectors by word, not by index, into the new vocabulary."""
    project2 = ROOT / "projects" / "02_word2vec_eval"
    old_word_to_index = json.loads(
        (project2 / "vocabulary.json").read_text(encoding="utf-8")
    )
    word2vec_results = json.loads(
        (project2 / "results.json").read_text(encoding="utf-8")
    )
    checkpoint = torch.load(
        WORD2VEC_DIR / word2vec_results["selected_checkpoint"], map_location="cpu"
    )
    matrix_name = word2vec_results["selected_matrix"]
    old_weights = checkpoint["model"][matrix_name]

    generator = torch.Generator().manual_seed(SEED)
    new_weights = torch.empty(VOCAB_SIZE, EMBEDDING_DIM)
    new_weights.normal_(mean=0.0, std=1 / math.sqrt(EMBEDDING_DIM), generator=generator)
    transferred = 0
    for word, new_index in lm_word_to_index.items():
        old_index = old_word_to_index.get(word)
        if old_index is not None:
            new_weights[new_index] = old_weights[old_index]
            transferred += 1
    return new_weights, transferred, {
        "checkpoint": word2vec_results["selected_checkpoint"],
        "matrix": matrix_name,
    }


def evaluate(
    model: torch.nn.Module,
    loader: DataLoader,
    device: torch.device,
) -> tuple[float, float]:
    model.eval()
    total_loss = 0.0
    total_tokens = 0
    with torch.inference_mode():
        for context, target in loader:
            context = context.to(device, non_blocking=True)
            target = target.to(device, non_blocking=True)
            logits = model(context)
            total_loss += F.cross_entropy(
                logits.reshape(-1, VOCAB_SIZE),
                target.reshape(-1),
                reduction="sum",
            ).item()
            total_tokens += target.numel()
    average_loss = total_loss / total_tokens
    return average_loss, math.exp(min(average_loss, 20))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=5)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    set_seed()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    project1_results = json.loads(
        (ROOT / "projects" / "01_language_modeling_ladder" / "results.json").read_text(encoding="utf-8")
    )
    winner = min(
        project1_results["models"],
        key=lambda name: project1_results["models"][name]["best_val_perplexity"],
    )

    print("Tokenizing with stopwords retained for readable generation...")
    tokenized = tokenize_corpus(remove_stopwords=False)
    word_to_index, index_to_word = build_vocabulary(tokenized, VOCAB_SIZE)
    chunks = make_chunks(tokenized, word_to_index, CHUNK_SIZE)
    rng = random.Random(SEED)
    rng.shuffle(chunks)
    split_index = int(len(chunks) * 0.9)
    train_dataset = NextTokenDataset(chunks[:split_index])
    val_dataset = NextTokenDataset(chunks[split_index:])
    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        generator=torch.Generator().manual_seed(SEED),
        pin_memory=device.type == "cuda",
        num_workers=0,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        pin_memory=device.type == "cuda",
        num_workers=0,
    )

    embedding_weights, transferred, embedding_source = transferred_embeddings(word_to_index)
    model = MODEL_CLASSES[winner](
        embedding_weights,
        HIDDEN_DIM,
        freeze_embedding=False,
    ).to(device)
    optimizer = Adam(model.parameters(), lr=LEARNING_RATE)

    sample_context, sample_target = next(iter(train_loader))
    sample_logits = model(sample_context[:2].to(device))
    assert sample_logits.shape == (2, CHUNK_SIZE - 1, VOCAB_SIZE)
    print(f"Device: {device}")
    print(f"Project 1 winner: {winner}")
    print(f"Sentences: {len(tokenized):,}")
    print(f"Vocabulary: {len(word_to_index):,}")
    print(f"Chunks: {len(chunks):,} (train={len(train_dataset):,}, val={len(val_dataset):,})")
    print(f"Transferred Word2Vec rows: {transferred:,}/{VOCAB_SIZE:,}")
    print(f"New rows to learn: {VOCAB_SIZE - transferred:,}")
    print(f"Shape check: context={tuple(sample_context[:2].shape)}, logits={tuple(sample_logits.shape)}")

    history: list[dict] = []
    best_perplexity = float("inf")
    best_state = None
    for epoch in range(1, args.epochs + 1):
        model.train()
        total_loss = 0.0
        total_tokens = 0
        if device.type == "cuda":
            torch.cuda.synchronize()
        start = time.perf_counter()
        for context, target in train_loader:
            context = context.to(device, non_blocking=True)
            target = target.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            logits = model(context)
            loss = F.cross_entropy(
                logits.reshape(-1, VOCAB_SIZE), target.reshape(-1)
            )
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            total_loss += loss.item() * target.numel()
            total_tokens += target.numel()

        if device.type == "cuda":
            torch.cuda.synchronize()
        seconds = time.perf_counter() - start
        train_loss = total_loss / total_tokens
        val_loss, val_perplexity = evaluate(model, val_loader, device)
        row = {
            "epoch": epoch,
            "train_loss": round(train_loss, 6),
            "val_loss": round(val_loss, 6),
            "val_perplexity": round(val_perplexity, 4),
            "epoch_seconds": round(seconds, 3),
        }
        history.append(row)
        if val_perplexity < best_perplexity:
            best_perplexity = val_perplexity
            best_state = {key: value.detach().cpu() for key, value in model.state_dict().items()}
        print(
            f"Epoch {epoch}/{args.epochs}: train_loss={train_loss:.4f}, "
            f"val_loss={val_loss:.4f}, perplexity={val_perplexity:.2f}, time={seconds:.1f}s"
        )

    artifact_dir = PROJECT_DIR / "artifacts"
    artifact_dir.mkdir(exist_ok=True)
    model_path = artifact_dir / "generator_model.pth"
    torch.save(
        {
            "model_name": winner,
            "model": best_state,
            "hidden_dim": HIDDEN_DIM,
            "vocab_size": VOCAB_SIZE,
            "embedding_dim": EMBEDDING_DIM,
            "word_to_index": word_to_index,
            "index_to_word": index_to_word,
            "best_val_perplexity": best_perplexity,
            "history": history,
        },
        model_path,
    )
    results = {
        "device": str(device),
        "model_name": winner,
        "epochs": args.epochs,
        "sentences": len(tokenized),
        "vocab_size": VOCAB_SIZE,
        "chunks": len(chunks),
        "train_chunks": len(train_dataset),
        "val_chunks": len(val_dataset),
        "embedding_source": embedding_source,
        "transferred_embedding_rows": transferred,
        "new_embedding_rows": VOCAB_SIZE - transferred,
        "best_val_perplexity": round(best_perplexity, 4),
        "history": history,
        "checkpoint": str(model_path.relative_to(ROOT)),
    }
    (PROJECT_DIR / "training_results.json").write_text(
        json.dumps(results, indent=2), encoding="utf-8"
    )
    print(f"Saved {model_path}")


if __name__ == "__main__":
    main()

