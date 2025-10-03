# qa‑model

**Polish Extractive Question‑Answering + Token‑Mapping for Generative LLMs**

This repository provides a complete workflow for a Polish extractive QA model based on a fine‑tuned RoBERTa checkpoint,
together with tooling to bridge its token vocabulary to a set of modern generative models (Gemma‑3, LLaMA‑3, GPT‑OSS).
The token‑mapping enables static‑bias generation, where QA‑derived logits steer the output of a generator.

---

## Features

- **Polish extractive QA** fine‑tuned from `radlab/polish-roberta-large-v2`.
- **Pre‑computed token maps** from the QA tokenizer to various generative tokenizers.
- **Static bias processor** (`QAStaticBiasProcessor`) that adds a weighted bias derived from QA logits to a generator’s
  logits.
- End‑to‑end training script with optional Weights & Biases logging.
- Simple inference script using Hugging Face `pipeline`.
- Fully reproducible environment (Python 3.10, `virtualenv`).

---

## Installation

1. **Clone the repository**

```bash
git clone https://github.com/radlab-dev-group/qa-model.git
cd radlab-qa-model
```

2. **Create a virtual environment and install dependencies**

```bash
python3.10 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

3. **(Optional) Install additional Hugging Face models**

```bash
pip install transformers tqdm click
```

---

## Training the QA Model

The training script `qa-model-train.py` fine‑tunes the RoBERTa checkpoint on the Polish **POQUAD** dataset.
The model card is available in [HF_MODEL_CARD](qa-model/HF_MODEL_CARD.md).

```bash
python qa-model-train.py \
  --model-path sdadas/polish-roberta-large-v2 \
  --dataset-path clarin-pl/poquad \
  --output-dir ./models \
  --out-model-name polish-roberta-large-v2-qa \
  --max-context-size 512 \
  --store-artifacts-to-wb   # optional: push dataset & model to Weights & Biases
```

Key arguments:

| Argument                  | Description                                                        |
|---------------------------|--------------------------------------------------------------------|
| `--model-path`            | Base model checkpoint (default: `sdadas/polish-roberta-large-v2`). |
| `--dataset-path`          | HF dataset identifier (default: `clarin-pl/poquad`).               |
| `--output-dir`            | Directory where training artifacts will be saved.                  |
| `--out-model-name`        | Name of the final QA model.                                        |
| `--max-context-size`      | Maximum token length for the context (default: 512).               |
| `--store-artifacts-to-wb` | Store dataset and model in a W&B run.                              |

Training uses `Trainer` from 🤗 Transformers with mixed‑precision support,
checkpointing, and early‑stopping based on evaluation steps.

---

## Running Inference

A quick inference example is provided in `qa-model-inference.py`:

```bash
python qa-model-inference.py
```

The script loads the model from the Hugging Face Hub 
[radlab/polish-qa-v2](https://huggingface.co/radlab/polish-qa-v2)
and runs a single QA pair:

```python
from transformers import pipeline

model_path = "radlab/polish-qa-v2"
qa = pipeline("question-answering", model=model_path)

question = "Co będzie w budowanym obiekcie?"
context = """..."""
print(qa(question=question, context=context.replace("\n", " ")))
```

The output is a JSON‑like dict with `answer`, `score`, `start`, and `end`.

---

## Building Token‑Mapping Tables

The script `build_all_maps.py` creates JSON maps that translate QA token IDs to one or more IDs of a target generative
model.

```bash
python tokenizer/build_all_maps.py
```

- **Configuration** – edit the `ROBERTA_MODELS` and `GENAI_MODELS` lists inside the script to add or remove models.
- **Cache** – generated maps are stored under `cache/maps/`
  and metadata (hashes) under `cache/meta/maps_metadata.json`.
- **Force rebuild** – use `--force-rebuild` to ignore existing caches.

These maps are later consumed by the bias processor.

---

## License

This project is licensed under the Apache 2.0 License – see the [LICENSE](LICENSE) file for details.