**License**: CC‑BY‑4.0  
**Datasets**:  
- `clarin-pl/poquad`  

**Language(s)**: Polish (pl)  
**Library**: 🤗 Transformers  

### Model Overview  

- **Model name**: `radlab/polish-qa-v2-bnb`  
- **Developer**: [radlab.dev](https://radlab.dev)  
- **Model type**: Extractive Question‑Answering (QA)  
- **Base model**: `sdadas/polish-roberta-large-v2` (fine‑tuned for QA)  
- **Quantization**: 8‑bit inference‑only quantization via **bitsandbytes** (`load_in_8bit=True`, double‑quantization enabled, `qa_outputs` excluded from quantization)  
- **Maximum context size**: 512 tokens  

### Intended Use  

This model is designed for **extractive QA** on Polish text. Given a question and a context passage,
it returns the most relevant span of the context as the answer.

### Limitations  

- The model works best with contexts up to 512 tokens. Longer passages should be truncated or split.  
- 8‑bit quantization reduces memory usage and inference latency but may introduce a slight drop in accuracy 
compared with the full‑precision model.  
- Only suitable for inference; it cannot be further fine‑tuned while kept in 8‑bit mode.

### How to Use  

```python
from transformers import pipeline

model_path = "radlab/polish-qa-v2-bnb"

qa = pipeline(
    "question-answering",
    model=model_path,
)

question = "Co będzie w budowanym obiekcie?"
context = """Pozwolenie na budowę zostało wydane w marcu. Pierwsze prace przygotowawcze
na terenie przy ul. Wojska Polskiego już się rozpoczęły.
Działkę ogrodzono, pojawił się również monitoring, a także kontenery
dla pracowników budowy. Na ten moment nie jest znana lista sklepów,
które pojawią się w nowym pasażu handlowym."""

result = qa(
    question=question,
    context=context.replace("\n", " ")
)

print(result)
```


**Sample output**

```json
{
  "score": 0.32568359375,
  "start": 259,
  "end": 268,
  "answer": "sklepów,"
}
```


### Technical Details  

- **Quantization strategy**: `BitsAndBytesStrategy` (8‑bit, double‑quant, `qa_outputs` excluded).  
- **Loading code (for reference)**  

```python
from transformers import AutoConfig, BitsAndBytesConfig, AutoModelForQuestionAnswering

config = AutoConfig.from_pretrained(original_path)
bnb_cfg = BitsAndBytesConfig(
    load_in_8bit=True,
    bnb_8bit_use_double_quant=True,
    bnb_8bit_excluded_modules=["qa_outputs"],
)

model = AutoModelForQuestionAnswering.from_pretrained(
    original_path,
    config=config,
    quantization_config=bnb_cfg,
    device_map="auto",
)
```
