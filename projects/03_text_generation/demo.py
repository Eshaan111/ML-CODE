"""Generate the interview demo at three temperatures and save the real text."""

from __future__ import annotations

import json
from pathlib import Path

import torch

from generate import (
    encode_seed,
    generate_greedy,
    generate_with_temperature,
    load_generator,
)


PROJECT_DIR = Path(__file__).resolve().parent


def diversity(text: str) -> dict[str, float | int]:
    words = text.split()
    return {
        "tokens": len(words),
        "unique_tokens": len(set(words)),
        "unique_ratio": round(len(set(words)) / len(words), 4),
    }


def main() -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, word_to_index, index_to_word, checkpoint = load_generator(device)
    seeds = ["harry looked at", "ron said to", "hermione opened the"]
    outputs: dict[str, dict] = {}

    print(f"Device: {device}")
    print(f"Model: {checkpoint['model_name']}")
    print(f"Held-out perplexity: {checkpoint['best_val_perplexity']:.2f}")
    for seed_number, seed in enumerate(seeds):
        seed_ids, unknown_words = encode_seed(seed, word_to_index)
        greedy = generate_greedy(model, seed_ids, index_to_word, 30, device)
        samples = {
            str(temperature): generate_with_temperature(
                model,
                seed_ids,
                index_to_word,
                30,
                temperature,
                device,
                random_seed=42 + seed_number,
            )
            for temperature in (0.5, 1.0, 1.5)
        }
        outputs[seed] = {
            "unknown_seed_words": unknown_words,
            "greedy": {"text": greedy, "diversity": diversity(greedy)},
            "temperatures": {
                temperature: {"text": text, "diversity": diversity(text)}
                for temperature, text in samples.items()
            },
        }
        print(f"\n=== Seed: {seed} ===")
        print(f"Greedy: {greedy}")
        for temperature, text in samples.items():
            print(f"Temperature {temperature}: {text}")

    result = {
        "device": str(device),
        "model_name": checkpoint["model_name"],
        "held_out_perplexity": checkpoint["best_val_perplexity"],
        "generated_length": 30,
        "outputs": outputs,
    }
    (PROJECT_DIR / "generations.json").write_text(
        json.dumps(result, indent=2), encoding="utf-8"
    )
    print(f"\nSaved {PROJECT_DIR / 'generations.json'}")


if __name__ == "__main__":
    main()
