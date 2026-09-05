# Recurrent NLP Learning Projects

This repository now contains three connected, CUDA-enabled NLP projects built from the Word2Vec, RNN, and LSTM experiments already present in the notebooks.

The projects are arranged as a learning ladder:

1. Learn how words become vectors and how to inspect those vectors.
2. Learn how recurrent models predict the next word and how RNN/LSTM implementations differ.
3. Learn how a trained language model generates text and how temperature changes its behavior.

Everything in the result pages comes from real runs on an NVIDIA GeForce RTX 4060 Laptop GPU. Expected outputs from the planning guides were not copied into the results.

## Projects

| Project | Main question | Main artifact |
|---|---|---|
| [01 — Language Modeling Ladder](projects/01_language_modeling_ladder/README.md) | How do a manual RNN, manual LSTM, and `nn.LSTM` compare? | Perplexity, speed, and parameter comparison |
| [02 — Word2Vec Evaluation](projects/02_word2vec_eval/README.md) | What did the CBOW model actually learn? | Neighbors, analogies, and t-SNE plot |
| [03 — Temperature Text Generation](projects/03_text_generation/README.md) | How does next-token prediction become generation? | Greedy and temperature-controlled samples |

The recommended learning order is **02 → 01 → 03**. Project 2 is numbered `02` because the original planning guide used that project numbering.

## Environment

The environment was built with:

- Python 3.11.9
- PyTorch 2.3.0
- CUDA runtime 12.1
- NVIDIA GeForce RTX 4060 Laptop GPU

Activate it from PowerShell:

```powershell
.\.venv\Scripts\Activate.ps1
```

To recreate the Python packages in a fresh Python 3.11 environment, use [requirements-cuda.txt](requirements-cuda.txt). Install the matching CUDA PyTorch wheels first using the command documented at the bottom of that file.

## Dataset

The scripts expect:

```text
datasets/hp_books/Book1.txt
...
datasets/hp_books/Book7.txt
```

The supplied `Harry-potter-TEXT.zip` was extracted there locally. The `datasets/` folder is intentionally ignored by Git.

With the project-local NLTK 3.8.1 tokenizer, the scripts measured:

- 62,821 sentence segments
- 586,163 stopword-free alphabetic tokens
- 1,145,487 alphabetic tokens with stopwords retained
- 5,000-word vocabularies

These measured values differ from the earlier draft guide's sentence estimate. The code reports what this exact environment produces.

## Run everything

Run commands from the repository root.

### Project 2: Word2Vec evaluation

```powershell
python projects\02_word2vec_eval\run.py
```

### Project 1: recurrent model comparison

Fast smoke test:

```powershell
python projects\01_language_modeling_ladder\run_comparison.py --smoke-test
```

Three-epoch portfolio run:

```powershell
python projects\01_language_modeling_ladder\run_comparison.py --epochs 3
```

### Project 3: generator training and demo

```powershell
python projects\03_text_generation\train_generator.py --epochs 5
python projects\03_text_generation\demo.py
```

## Results at a glance

### Word2Vec

- The three checkpoint files contain identical model tensors despite different file hashes.
- `output.weight` had stronger category cohesion than `embedding.weight` under the project's simple heuristic.
- Character neighbors were partially meaningful, but analogies were weak.
- This is an honest example of why a saved model should be evaluated instead of assumed to be good.

### Recurrent model comparison

| Model | Validation perplexity | Training time | Trainable parameters |
|---|---:|---:|---:|
| Simple RNN | **745.48** | 21.68 s | 667,912 |
| Custom LSTM | 798.72 | 52.91 s | 736,648 |
| `nn.LSTM` | 797.90 | **3.79 s** | 737,160 |

This was a short three-epoch comparison. The Simple RNN's early win does not prove it is generally superior to LSTMs. The major robust result is that PyTorch's fused LSTM was approximately 14 times faster than the Python-loop custom LSTM.

### Text generation

- The stopword-inclusive generator reached validation perplexity **113.08** after five epochs.
- 4,842 of 5,000 embedding rows were transferred from Word2Vec by matching words.
- Low temperature produced safer, more repetitive text.
- High temperature produced more diverse but less coherent text.

## How to study these projects

For each project:

1. Read its README before reading code.
2. Run the script once and watch the printed shapes.
3. Locate those shapes in the model's `forward` method.
4. Explain the result in your own words without looking at the README.
5. Change one controlled setting, such as epoch count or temperature, and predict what will happen before running it.

Do not memorize only the final numbers. Interviewers usually care more about whether you can explain the data flow, tensor shapes, loss, evaluation metric, limitations, and next experiment.
