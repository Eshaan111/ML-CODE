# Project 3 — Temperature-Controlled Text Generation

## The project in one sentence

This project retrains the Project 1 winner with a stopword-inclusive vocabulary, then generates text greedily and at temperatures `0.5`, `1.0`, and `1.5`.

Training is in [train_generator.py](train_generator.py), decoding is in [generate.py](generate.py), and the reproducible demo is in [demo.py](demo.py).

## What you should understand after this project

You should be able to explain:

- Why a next-token model can generate text autoregressively.
- The difference between teacher-forced training and inference.
- Why Project 1's stopword-free model was unsuitable for readable generation.
- How pretrained embeddings were transferred into a new vocabulary.
- The difference between greedy decoding and sampling.
- How temperature changes a probability distribution.
- Why lower perplexity does not guarantee perfect prose.

## 1. The stopword problem

Project 1 faithfully used:

```python
remove_stopwords=True
```

That excludes words such as:

```text
the, at, to, of, was, and
```

A language generator without these words cannot produce normal English. A seed such as `harry looked at` would map `at` to `<UNK>`.

Project 3 therefore builds a separate 5,000-word vocabulary with stopwords retained.

This is not a cosmetic change. Tokenization and vocabulary define what a model is capable of emitting.

## 2. Transferring Word2Vec embeddings by word

The old and new vocabularies assign different indices to words. Copying row 100 to row 100 would be wrong because those rows may represent different words.

The project transfers vectors by matching the word strings:

```python
for word, new_index in lm_word_to_index.items():
    if word in old_word_to_index:
        old_index = old_word_to_index[word]
        new_embeddings[new_index] = old_embeddings[old_index]
```

Actual transfer result:

```text
Transferred rows: 4,842 / 5,000
New random rows:    158 / 5,000
```

The entire embedding layer is then trainable. This allows:

- Transferred content-word vectors to adapt to next-token prediction.
- Newly introduced stopword vectors to learn from the language-model objective.

## 3. Training data

With stopwords retained, the project produced:

- 62,821 sentence segments.
- 22,909 chunks of 50 tokens.
- 20,618 training chunks.
- 2,291 validation chunks.

Each chunk still creates 49 next-token inputs and targets.

## 4. Why the Simple RNN was used

The Simple RNN had the best validation perplexity in Project 1's three-epoch comparison, so it became the generation architecture.

This is a project rule, not a claim that Simple RNNs are always superior. If a longer Project 1 run selects a different winner, `train_generator.py` will read the updated results and use that architecture.

## 5. Training results

| Epoch | Training loss | Validation loss | Validation perplexity |
|---:|---:|---:|---:|
| 1 | 5.9571 | 5.4137 | 224.46 |
| 2 | 5.1918 | 5.0275 | 152.54 |
| 3 | 4.9295 | 4.8726 | 130.66 |
| 4 | 4.7914 | 4.7855 | 119.77 |
| 5 | **4.7004** | **4.7281** | **113.08** |

Both training and validation loss fell consistently, showing that the model learned useful next-token statistics rather than only memorizing the training batches.

## 6. Autoregressive generation

During training, the model receives the real previous tokens. During generation, it must consume its own predictions.

The loop is:

```text
seed tokens
→ predict next-token distribution
→ choose one token
→ append it to the context
→ run the expanded context again
→ repeat
```

This is called autoregressive generation.

Errors can compound because a generated mistake becomes part of the next input.

## 7. Greedy decoding

Greedy decoding always chooses the highest-scoring token:

```python
next_token = logits.argmax()
```

Actual example:

```text
harry looked at the other side of the fire and the order of the phoenix
rowling the door of the phoenix rowling harry s face was still in the air...
```

It is locally confident but repetitive. Once the model enters a high-probability pattern, greedy decoding has no randomness to escape it.

## 8. Temperature sampling

The model produces logits, which are converted into probabilities with softmax.

Temperature changes the logits first:

```text
probabilities = softmax(logits / temperature)
```

### Temperature below 1

Dividing by `0.5` increases the magnitude of logit differences. The distribution becomes sharper, and already-likely words become even more likely.

Actual example:

```text
harry looked at him at the moment the door he had just got a little and
that s the hogwarts express the death eaters had just been a lot of it...
```

### Temperature equal to 1

The model samples from its original learned distribution.

```text
harry looked at him at malfoy as the end he felt her chest professor
mcgonagall kept that are he dreamed of the forbidden friends...
```

### Temperature above 1

Dividing by `1.5` reduces logit differences. The distribution becomes flatter, giving unlikely words a greater chance.

```text
harry looked at him at malfoy cheering the cost road whipping her chest
professor mcgonagall kept features forward he dreamed of galleons...
```

The high-temperature result is more diverse but less coherent.

## 9. Why `<UNK>` is masked during generation

`<UNK>` is a special training token representing all words outside the 5,000-word vocabulary. Because it represents many rare words, it can receive a high predicted probability.

Printing `<UNK>` repeatedly is not useful to a reader, so generation sets its logit to negative infinity before greedy selection or sampling:

```python
logits[unknown_id] = -torch.inf
```

This changes inference only. The model was still trained honestly with `<UNK>` targets.

## 10. Why the text contains no punctuation

The preprocessing retains only alphabetic tokens:

```python
token.isalpha()
```

Therefore commas, periods, quotation marks, and apostrophes are not vocabulary tokens. For example, `Harry's` becomes separate alphabetic pieces such as `harry` and `s`.

Adding punctuation-aware tokenization would be a natural next improvement.

## 11. Run the project

Train:

```powershell
.\.venv\Scripts\Activate.ps1
python projects\03_text_generation\train_generator.py --epochs 5
```

Generate the saved comparison:

```powershell
python projects\03_text_generation\demo.py
```

The outputs are stored in:

- `training_results.json`
- `generations.json`
- `artifacts/generator_model.pth` locally

## 12. Read the code in this order

In [train_generator.py](train_generator.py):

1. `NextTokenDataset`
2. `transferred_embeddings`
3. `evaluate`
4. `main`

In [generate.py](generate.py):

1. `load_generator`
2. `encode_seed`
3. `generate_greedy`
4. `generate_with_temperature`

Finally read [demo.py](demo.py) to see how the same model is exercised at several temperatures.

## 13. Limitations and next improvements

- The vocabulary contains only alphabetic tokens.
- Book headers and author names create patterns such as `phoenix rowling`.
- The model is small and trained for only five epochs.
- Greedy decoding is repetitive.
- Temperature alone cannot prevent all poor samples.
- The implementation recomputes the full context at every generation step for clarity.

Reasonable next experiments are:

- Remove book headers before training.
- Preserve punctuation.
- Train longer.
- Use the fused LSTM winner from a longer comparison.
- Add top-k or nucleus sampling.
- Cache recurrent hidden states during generation.

## 14. Interview explanation

> I converted my next-token model into an autoregressive generator. I first fixed a preprocessing mismatch: the benchmark vocabulary removed stopwords, so I built a stopword-inclusive vocabulary and transferred 4,842 pretrained vectors by matching words rather than indices. After five epochs the model reached validation perplexity 113.08. I implemented greedy decoding and temperature sampling, and showed that lower temperatures were safer and more repetitive while higher temperatures were more diverse but less coherent. I also masked the special unknown token during inference so it could not dominate readable output.

## Questions you should be ready to answer

- What is autoregressive generation?
- What is teacher forcing?
- Why did this project need a different vocabulary?
- Why must embeddings be transferred by word rather than row number?
- What happens mathematically when temperature decreases?
- Why is greedy decoding repetitive?
- Why mask `<UNK>` only at inference?
- Why can a model with improving perplexity still produce awkward text?

