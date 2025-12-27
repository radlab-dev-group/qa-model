import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

model_id = "/mnt/data2/llms/models/community/google/gemma-3-12b-it"
device = "cuda" if torch.cuda.is_available() else "cpu"

print(f"Ładowanie modelu na urządzenie: {device}")

tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)

try:
    if device == "cuda":
        model = AutoModelForCausalLM.from_pretrained(
            model_id,
            torch_dtype=torch.bfloat16,
            device_map="auto",
        )
    else:
        model = AutoModelForCausalLM.from_pretrained(
            model_id,
            torch_dtype=torch.bfloat16,
        ).to(device)

except Exception as e:
    print(f"Wystąpił błąd podczas ładowania modelu: {e}")
    print("Spróbuj zainstalować accelerate, bitsandbytes lub użyć innej strategii ładowania (np. bez kwantyzacji).")
    exit()


prompt = (
    "Napisz krótkie opowiadanie science fiction o sztucznej inteligencji, "
    "która odkryła, że rzeczywistość jest symulacją."
)

chat = [
    {"role": "user", "content": prompt},
]
formatted_prompt = tokenizer.apply_chat_template(chat, tokenize=False, add_generation_prompt=True)

print("\n--- Przygotowany Prompt ---")
print(formatted_prompt)

input_ids = tokenizer.encode(
    formatted_prompt,
    return_tensors="pt"
).to(device)


print("\n--- Rozpoczynanie Generacji (Sampling z temp 0.7) ---")

output_ids = model.generate(
    input_ids,
    max_new_tokens=300,
    do_sample=True,
    temperature=0.7,
    top_k=50,
    top_p=0.95,
    pad_token_id=tokenizer.eos_token_id
)

generated_text = tokenizer.decode(output_ids[0], skip_special_tokens=True)

start_index = len(formatted_prompt)
final_response = tokenizer.decode(output_ids[0][len(input_ids[0]):], skip_special_tokens=True)

print("\n--- Wynik Generacji ---")
print(final_response)