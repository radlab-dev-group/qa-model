# qa‑model

**Polish Extractive Question‑Answering + Token‑Mapping for Generative LLMs**

This repository provides a complete workflow for a Polish extractive QA model based on a fine‑tuned RoBERTa checkpoint, 
together with tooling to bridge its token vocabulary to a set of modern generative models (Gemma‑3, LLaMA‑3, GPT‑OSS). 
The token‑mapping enables static‑bias generation, where QA‑derived logits steer the output of a generator.

---

## Features
- **Polish extractive QA** fine‑tuned from `radlab/polish-roberta-large-v2`.
- **Pre‑computed token maps** from the QA tokenizer to various generative tokenizers.
- **Static bias processor** (`QAStaticBiasProcessor`) that adds a weighted bias derived from QA logits to a generator’s logits.
- End‑to‑end training script with optional Weights & Biases logging.
- Simple inference script using Hugging Face `pipeline`.
- Fully reproducible environment (Python 3.10, `virtualenv`).

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

3. **(Optional) Install additional Hugging Face models**
```bash
pip install transformers tqdm click
```

---

## Training the QA Model

The training script `qa-model-train.py` fine‑tunes the RoBERTa checkpoint on the Polish **POQUAD** dataset.
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

Training uses `Trainer` from 🤗 Transformers with mixed‑precision support, 
checkpointing, and early‑stopping based on evaluation steps.

---

## Running Inference

A quick inference example is provided in `qa-model-inference.py`:
```
bash
python qa-model-inference.py
```
The script loads the model from the Hugging Face Hub (`radlab/polish-roberta-large-v2-qa`) and runs a single QA pair:
```python
from transformers import pipeline

model_path = "radlab/polish-roberta-large-v2-qa"
qa = pipeline("question-answering", model=model_path)

question = "Co będzie w budowanym obiekcie?"
context = """..."""
print(qa(question=question, context=context.replace("\n", " ")))
```
The output is a JSON‑like dict with `answer`, `score`, `start`, and `end`.

---

## Building Token‑Mapping Tables

The script `build_all_maps.py` creates JSON maps that translate QA token IDs to one or more IDs of a target generative model.
```bash
python tokenizer/builder/scrips/build_all_maps.py
```
- **Configuration** – edit the `ROBETTA_MODELS` and `GENAI_MODELS` lists inside the script to add or remove models.
- **Cache** – generated maps are stored under `tokenizer/cache/maps/` and metadata (hashes)
under `tokenizer/cache/meta/maps_metadata.json`.
- **Force rebuild** – use `--force-rebuild` to ignore existing caches.

These maps are later consumed by the bias processor.

---

## Static‑Bias Generation

`generation_with_precomputed_map.py` demonstrates how to:

1. Run a forward pass through the QA model to obtain logits.
2. Load the pre‑computed token map (`load_qa_to_gen_map`).
3. Build a `QAStaticBiasProcessor` that aggregates QA logits (mean by default) and distributes them onto the generator’s vocabulary.
4. Generate a response with a generative model (e.g., Gemma‑3) while the bias nudges the output toward QA‑relevant tokens.
```
python
from pathlib import Path
answer = generate_answer_with_static_bias(
    query="What are the opening hours?",
    passages=[...],
    qa_model=qa_model,
    qa_tokenizer=qa_tokenizer,
    gen_model=gen_model,
    gen_tokenizer=gen_tokenizer,
    map_path=Path("tokenizer/cache/maps/radlab-qa__google-gemma3.json")
)
print(answer)
```
Feel free to experiment with different aggregation functions (`torch.max`, `torch.median`) and scaling factors (`scale`).

---

## Contributing

Contributions are welcome! Please follow these steps:

1. Fork the repository.
2. Create a feature branch (`git checkout -b feature/your‑feature`).
3. Ensure code passes **PEP 8** (`autopep8`, `pylint`) and unit tests (if added).
4. Open a pull request with a clear description of the changes.

---

## License

The model weights are released under **CC‑BY‑4.0**.  
The source code in this repository is MIT‑licensed. See the `LICENSE` file for details.

---

*Happy coding! 🎉*
