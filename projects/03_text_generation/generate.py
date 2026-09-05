"""Load the generation model and decode text greedily or with temperature."""

from __future__ import annotations

import importlib.util
from pathlib import Path

import torch


PROJECT_DIR = Path(__file__).resolve().parent
ROOT = PROJECT_DIR.parents[1]
MODEL_FILE = ROOT / "projects" / "01_language_modeling_ladder" / "models.py"


def load_generator(device: torch.device) -> tuple[torch.nn.Module, dict[str, int], dict[int, str], dict]:
    checkpoint = torch.load(PROJECT_DIR / "artifacts" / "generator_model.pth", map_location="cpu")
    spec = importlib.util.spec_from_file_location("ladder_models", MODEL_FILE)
    models_module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(models_module)

    embedding_weights = checkpoint["model"]["embedding.weight"]
    model = models_module.MODEL_CLASSES[checkpoint["model_name"]](
        embedding_weights,
        checkpoint["hidden_dim"],
        freeze_embedding=False,
    )
    model.load_state_dict(checkpoint["model"])
    model.to(device).eval()
    index_to_word = {int(index): word for index, word in checkpoint["index_to_word"].items()}
    return model, checkpoint["word_to_index"], index_to_word, checkpoint


def encode_seed(seed_text: str, word_to_index: dict[str, int]) -> tuple[list[int], list[str]]:
    words = [word.strip(".,!?;:\"'()[]{}").lower() for word in seed_text.split()]
    words = [word for word in words if word]
    if not words:
        raise ValueError("Seed text must contain at least one word.")
    unknown_words = [word for word in words if word not in word_to_index]
    unknown = word_to_index["<UNK>"]
    return [word_to_index.get(word, unknown) for word in words], unknown_words


def generate_greedy(
    model: torch.nn.Module,
    seed_ids: list[int],
    index_to_word: dict[int, str],
    length: int,
    device: torch.device,
) -> str:
    generated = list(seed_ids)
    unknown_id = next(index for index, word in index_to_word.items() if word == "<UNK>")
    with torch.inference_mode():
        for _ in range(length):
            context = torch.tensor([generated], dtype=torch.long, device=device)
            logits = model(context)[0, -1]
            # <UNK> is useful for training rare words, but is not readable output.
            logits[unknown_id] = -torch.inf
            next_id = logits.argmax().item()
            generated.append(next_id)
    return " ".join(index_to_word[index] for index in generated)


def generate_with_temperature(
    model: torch.nn.Module,
    seed_ids: list[int],
    index_to_word: dict[int, str],
    length: int,
    temperature: float,
    device: torch.device,
    random_seed: int = 42,
) -> str:
    if temperature <= 0:
        raise ValueError("Temperature must be greater than zero.")
    generated = list(seed_ids)
    unknown_id = next(index for index, word in index_to_word.items() if word == "<UNK>")
    generator = torch.Generator(device=device).manual_seed(random_seed)
    with torch.inference_mode():
        for _ in range(length):
            context = torch.tensor([generated], dtype=torch.long, device=device)
            logits = model(context)[0, -1] / temperature
            logits[unknown_id] = -torch.inf
            probabilities = torch.softmax(logits, dim=-1)
            next_id = torch.multinomial(probabilities, 1, generator=generator).item()
            generated.append(next_id)
    return " ".join(index_to_word[index] for index in generated)
