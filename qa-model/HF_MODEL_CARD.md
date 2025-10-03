---
license: cc-by-4.0
datasets:
- clarin-pl/poquad
language:
- pl
library_name: transformers
---
# Model Card
Extractive Question-Answer model for polish.  Extractive means, that the most relevant 
chunk of the text is returned as answer from t he context for the given question.

## Model Details

- **Model name:** `radlab/polish-roberta-large-v2-qa`
- **Developed by:** [radlab.dev](https://radlab.dev)
- **Shared by:** [radlab.dev](https://radlab.dev)
- **Model type:** QA
- **Language(s) (NLP):** PL
- **License:** CC-BY-4.0
- **Finetuned from model:** [sdadas/polish-roberta-large-v2](https://huggingface.co/radlab/polish-roberta-large-v2-sts)
- **Maxiumum context size:*** 512 tokens

## Model Usage

Simple model usage with huggingface library:

```python
from transformers import pipeline

model_path = "radlab/polish-roberta-large-v2-qa"

question_answerer = pipeline(
  "question-answering",
  model=model_path
)

question = "Co będzie w budowanym obiekcie?"
context = """Pozwolenie na budowę zostało wydane w marcu. Pierwsze prace przygotowawcze
na terenie przy ul. Wojska Polskiego już się rozpoczęły.
Działkę ogrodzono, pojawił się również monitoring, a także kontenery
dla pracowników budowy. Na ten moment nie jest znana lista sklepów,
które pojawią się w nowym pasażu handlowym."""

print(
  question_answerer(
    question=question,
    context=context.replace("\n", " ")
  )
)
```

with the sample output:

```json
{
  'score': 0.3472374677658081, 
  'start': 259, 
  'end': 268, 
  'answer': ' sklepów,'
}
```
