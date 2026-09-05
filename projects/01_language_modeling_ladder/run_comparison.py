"""Train and compare a vanilla RNN, a custom LSTM, and ``nn.LSTM``.

Fast correctness run:
    python projects/01_language_modeling_ladder/run_comparison.py --smoke-test

Portfolio run (default is three epochs):
    python projects/01_language_modeling_ladder/run_comparison.py --epochs 3
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

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from torch.optim import Adam
from torch.utils.data import DataLoader, Dataset

from projects.data_utils import WORD2VEC_DIR, make_chunks, tokenize_corpus

spec = importlib.util.spec_from_file_location("ladder_models", PROJECT_DIR / "models.py")
models_module = importlib.util.module_from_spec(spec)
assert spec.loader is not None
spec.loader.exec_module(models_module)
MODEL_CLASSES = models_module.MODEL_CLASSES


SEED = 42
VOCAB_SIZE = 5000
HIDDEN_DIM = 128
BATCH_SIZE = 32
LEARNING_RATE = 1e-3
CHUNK_SIZE = 50


class NextTokenDataset(Dataset):
    """Turn each 50-token chunk into 49 inputs and 49 next-token targets."""

    def __init__(self, chunks: list[list[int]]) -> None:
        data = torch.tensor(chunks, dtype=torch.long)
        self.context = data[:, :-1]
        self.target = data[:, 1:]

    def __len__(self) -> int:
        return len(self.context)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        return self.context[index], self.target[index]


def set_seed(seed: int = SEED) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def load_project2_choices() -> tuple[dict[str, int], torch.Tensor, dict[str, str]]:
    project2 = ROOT / "projects" / "02_word2vec_eval"
    word_to_index = json.loads((project2 / "vocabulary.json").read_text(encoding="utf-8"))
    results = json.loads((project2 / "results.json").read_text(encoding="utf-8"))
    checkpoint = torch.load(
        WORD2VEC_DIR / results["selected_checkpoint"], map_location="cpu"
    )
    matrix_name = results["selected_matrix"]
    weights = checkpoint["model"][matrix_name].detach().clone()
    return word_to_index, weights, {
        "checkpoint": results["selected_checkpoint"],
        "matrix": matrix_name,
    }


def make_loaders(
    train_dataset: Dataset,
    val_dataset: Dataset,
    device: torch.device,
) -> tuple[DataLoader, DataLoader]:
    generator = torch.Generator().manual_seed(SEED)
    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        generator=generator,
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
    return train_loader, val_loader


def evaluate(
    model: torch.nn.Module,
    loader: DataLoader,
    device: torch.device,
    max_batches: int | None = None,
) -> tuple[float, float]:
    model.eval()
    total_loss = 0.0
    total_tokens = 0
    with torch.inference_mode():
        for batch_index, (context, target) in enumerate(loader):
            if max_batches is not None and batch_index >= max_batches:
                break
            context = context.to(device, non_blocking=True)
            target = target.to(device, non_blocking=True)
            logits = model(context)
            loss = F.cross_entropy(
                logits.reshape(-1, logits.size(-1)),
                target.reshape(-1),
                reduction="sum",
            )
            total_loss += loss.item()
            total_tokens += target.numel()
    average_loss = total_loss / total_tokens
    return average_loss, math.exp(min(average_loss, 20))


def train_one_model(
    name: str,
    model: torch.nn.Module,
    train_dataset: Dataset,
    val_dataset: Dataset,
    device: torch.device,
    epochs: int,
    max_train_batches: int | None,
    max_val_batches: int | None,
) -> dict:
    train_loader, val_loader = make_loaders(train_dataset, val_dataset, device)
    optimizer = Adam((p for p in model.parameters() if p.requires_grad), lr=LEARNING_RATE)
    model.to(device)

    # A visible correctness check before spending time training.
    sample_context, sample_target = next(iter(train_loader))
    sample_logits = model(sample_context[:2].to(device))
    assert sample_logits.shape == (2, CHUNK_SIZE - 1, VOCAB_SIZE)
    sample_loss = F.cross_entropy(
        sample_logits.reshape(-1, VOCAB_SIZE),
        sample_target[:2].to(device).reshape(-1),
    )
    sample_loss.backward()
    optimizer.zero_grad(set_to_none=True)
    print(
        f"{name} shape check: context={tuple(sample_context[:2].shape)}, "
        f"logits={tuple(sample_logits.shape)}, loss={sample_loss.item():.4f}"
    )

    history: list[dict] = []
    best_perplexity = float("inf")
    best_state = None
    if device.type == "cuda":
        torch.cuda.synchronize()
    training_start = time.perf_counter()

    for epoch in range(1, epochs + 1):
        model.train()
        epoch_loss = 0.0
        epoch_tokens = 0
        epoch_start = time.perf_counter()
        for batch_index, (context, target) in enumerate(train_loader):
            if max_train_batches is not None and batch_index >= max_train_batches:
                break
            context = context.to(device, non_blocking=True)
            target = target.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            logits = model(context)
            loss = F.cross_entropy(
                logits.reshape(-1, logits.size(-1)), target.reshape(-1)
            )
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            epoch_loss += loss.item() * target.numel()
            epoch_tokens += target.numel()

        if device.type == "cuda":
            torch.cuda.synchronize()
        epoch_seconds = time.perf_counter() - epoch_start
        train_loss = epoch_loss / epoch_tokens
        val_loss, val_perplexity = evaluate(
            model, val_loader, device, max_batches=max_val_batches
        )
        history.append(
            {
                "epoch": epoch,
                "train_loss": round(train_loss, 6),
                "val_loss": round(val_loss, 6),
                "val_perplexity": round(val_perplexity, 4),
                "epoch_seconds": round(epoch_seconds, 3),
            }
        )
        if val_perplexity < best_perplexity:
            best_perplexity = val_perplexity
            best_state = {key: value.detach().cpu() for key, value in model.state_dict().items()}
        print(
            f"{name} epoch {epoch}/{epochs}: train_loss={train_loss:.4f}, "
            f"val_loss={val_loss:.4f}, perplexity={val_perplexity:.2f}, "
            f"time={epoch_seconds:.1f}s"
        )

    if device.type == "cuda":
        torch.cuda.synchronize()
    training_seconds = time.perf_counter() - training_start
    artifact_dir = PROJECT_DIR / "artifacts"
    artifact_dir.mkdir(exist_ok=True)
    slug = name.lower().replace(".", "").replace(" ", "_")
    checkpoint_path = artifact_dir / f"{slug}.pth"
    torch.save(
        {
            "model_name": name,
            "model": best_state,
            "hidden_dim": HIDDEN_DIM,
            "vocab_size": VOCAB_SIZE,
            "embedding_frozen": True,
            "history": history,
        },
        checkpoint_path,
    )
    total_parameters = sum(parameter.numel() for parameter in model.parameters())
    trainable_parameters = sum(
        parameter.numel() for parameter in model.parameters() if parameter.requires_grad
    )
    del model
    if device.type == "cuda":
        torch.cuda.empty_cache()

    return {
        "best_val_perplexity": round(best_perplexity, 4),
        "training_seconds": round(training_seconds, 3),
        "total_parameters": total_parameters,
        "trainable_parameters": trainable_parameters,
        "checkpoint": str(checkpoint_path.relative_to(ROOT)),
        "history": history,
    }


def plot_history(results: dict[str, dict]) -> None:
    figure, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    for name, result in results.items():
        epochs = [row["epoch"] for row in result["history"]]
        axes[0].plot(epochs, [row["train_loss"] for row in result["history"]], marker="o", label=name)
        axes[1].plot(epochs, [row["val_perplexity"] for row in result["history"]], marker="o", label=name)
    axes[0].set_title("Training Loss")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Cross-entropy")
    axes[1].set_title("Validation Perplexity")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Perplexity (lower is better)")
    for axis in axes:
        axis.grid(alpha=0.2)
        axis.legend()
    figure.tight_layout()
    figure.savefig(PROJECT_DIR / "training_curves.png", dpi=180)
    plt.close(figure)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--smoke-test", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    set_seed()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    epochs = 1 if args.smoke_test else args.epochs
    max_train_batches = 20 if args.smoke_test else None
    max_val_batches = 10 if args.smoke_test else None

    word_to_index, embedding_weights, embedding_source = load_project2_choices()
    tokenized = tokenize_corpus(remove_stopwords=True)
    chunks = make_chunks(tokenized, word_to_index, CHUNK_SIZE)
    rng = random.Random(SEED)
    rng.shuffle(chunks)
    split_index = int(len(chunks) * 0.9)
    train_dataset = NextTokenDataset(chunks[:split_index])
    val_dataset = NextTokenDataset(chunks[split_index:])

    print(f"Device: {device}")
    if device.type == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"Embedding source: {embedding_source}")
    print(f"Chunks: {len(chunks):,} (train={len(train_dataset):,}, val={len(val_dataset):,})")
    first_context, first_target = train_dataset[0]
    print(f"One sample: context={tuple(first_context.shape)}, target={tuple(first_target.shape)}")

    comparison: dict[str, dict] = {}
    for name, model_class in MODEL_CLASSES.items():
        set_seed()
        print(f"\n=== {name} ===")
        model = model_class(embedding_weights, HIDDEN_DIM, freeze_embedding=True)
        comparison[name] = train_one_model(
            name,
            model,
            train_dataset,
            val_dataset,
            device,
            epochs,
            max_train_batches,
            max_val_batches,
        )

    output = {
        "run_type": "smoke_test" if args.smoke_test else "portfolio_run",
        "device": str(device),
        "gpu": torch.cuda.get_device_name(0) if device.type == "cuda" else None,
        "epochs": epochs,
        "batch_size": BATCH_SIZE,
        "hidden_dim": HIDDEN_DIM,
        "learning_rate": LEARNING_RATE,
        "chunk_size": CHUNK_SIZE,
        "train_chunks": len(train_dataset),
        "val_chunks": len(val_dataset),
        "embedding_source": embedding_source,
        "models": comparison,
    }
    output_name = "smoke_results.json" if args.smoke_test else "results.json"
    (PROJECT_DIR / output_name).write_text(json.dumps(output, indent=2), encoding="utf-8")
    plot_history(comparison)
    winner = min(comparison, key=lambda name: comparison[name]["best_val_perplexity"])
    print(f"\nWinner by validation perplexity: {winner}")
    print(f"Saved {PROJECT_DIR / output_name}")


if __name__ == "__main__":
    main()
