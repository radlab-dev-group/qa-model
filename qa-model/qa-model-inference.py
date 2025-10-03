from transformers import pipeline

model_path = "/mnt/data2/llms/models/radlab-open/qa/best_model"

question_answerer = pipeline("question-answering", model=model_path)

question = "Co będzie w budowanym obiekcie?"
context = """Pozwolenie na budowę zostało wydane w marcu. Pierwsze prace przygotowawcze
na terenie przy ul. Wojska Polskiego już się rozpoczęły.
Działkę ogrodzono, pojawił się również monitoring, a także kontenery
dla pracowników budowy. Na ten moment nie jest znana lista sklepów,
które pojawią się w nowym pasażu handlowym."""

print(question_answerer(question=question, context=context.replace("\n", " ")))
