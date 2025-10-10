import torch
from transformers import LogitsProcessor, LogitsProcessorList


class QAStaticBiasProcessor(LogitsProcessor):
    """
    Dodaje bias wyliczony raz na podstawie pre‑zbudowanej mapy.
    """

    def __init__(
        self,
        qa_logits: torch.Tensor,  # (seq_len, vocab_qa)
        qa_to_gen_map: dict[int, list[int]],
        vocab_gen: int,
        agg_fn=torch.mean,
        scale: float = 0.8,
    ):
        """
        - `qa_logits` – logity z modelu QA (po forward‑passie).
        - `qa_to_gen_map` – słownik QA‑id → lista gen‑id.
        - `vocab_gen` – rozmiar słownika generatora (potrzebny do alokacji biasu).
        - `agg_fn` – funkcja agregująca logity wzdłuż wymiaru sekwencji
                     (np. mean, max, median).
        - `scale` – współczynnik α (0 < α ≤ 1).
        """
        self.bias = torch.zeros(
            vocab_gen, dtype=qa_logits.dtype, device=qa_logits.device
        )
        agg_per_token = agg_fn(qa_logits, dim=0)  # shape: (vocab_qa,)
        # If aggregation produced a scalar (0‑D), repeat it to match the
        # number of QA token ids so indexing works.
        if agg_per_token.dim() == 0:
            # największy QA‑id w mapie + 1, aby mieć wystarczającą długość
            max_qa_id = max(qa_to_gen_map.keys(), default=0)
            target_len = max_qa_id + 1
            agg_per_token = agg_per_token.repeat(target_len)

        # 3️⃣ Rozprowadzamy na tokeny generatora
        for qa_id, gen_ids in qa_to_gen_map.items():
            if not gen_ids:
                continue
            # Omiń QA‑id, które wykraczają poza rozmiar wektora
            if qa_id >= agg_per_token.size(0):
                continue
            value = agg_per_token[qa_id]  # skalar (już na właściwym urządzeniu)
            for gid in gen_ids:
                self.bias[gid] += value

        # 4️⃣ Skalujemy (α)
        if scale != 1.0:
            self.bias = self.bias * scale

    def __call__(
        self, input_ids: torch.Tensor, scores: torch.Tensor
    ) -> torch.Tensor:
        """
        `scores` – (batch, vocab_gen) surowe logity generatora.
        Dodajemy bias i zwracamy zmodyfikowane logity.
        """
        # Add bias
        modified_scores = scores + self.bias

        # ---- NEW: clamp logits to a reasonable range ----
        # Extremely large or small values can produce NaNs after softmax.
        # Clamp to [-1e4, 1e4]; adjust if needed.
        modified_scores = modified_scores.clamp(min=-1e4, max=1e4)

        # Ensure no NaNs / infinities remain after clamping.
        modified_scores = torch.nan_to_num(
            modified_scores, nan=0.0, posinf=0.0, neginf=0.0
        )

        return modified_scores


class StaticBiasProcessor(LogitsProcessor):
    def __init__(self, bias: torch.Tensor):
        super().__init__()
        self.bias = bias  # shape should be [vocab_size]

    def __call__(
        self, input_ids: torch.LongTensor, scores: torch.FloatTensor
    ) -> torch.FloatTensor:
        # scores shape: [batch, vocab_size]
        vocab_size = scores.size(-1)
        if self.bias.device != scores.device:
            self.bias = self.bias.to(scores.device)
        if self.bias.dtype != scores.dtype:
            self.bias = self.bias.to(dtype=scores.dtype)

        # Ensure bias length matches current vocab_size
        if self.bias.numel() != vocab_size:
            if self.bias.numel() > vocab_size:
                # Truncate extra entries (likely map built for larger vocab)
                self.bias = self.bias[:vocab_size]
            else:
                # Pad with zeros to match vocab (likely map built for smaller vocab)
                pad = torch.zeros(
                    vocab_size - self.bias.numel(),
                    device=self.bias.device,
                    dtype=self.bias.dtype,
                )
                self.bias = torch.cat([self.bias, pad], dim=0)

        return scores + self.bias


def _build_bias_vector(qa_to_gen_map, gen_tokenizer, device) -> torch.Tensor:
    # Create bias vector aligned to generator vocab
    vocab_size = getattr(gen_tokenizer, "vocab_size", None)
    if vocab_size is None:
        vocab_size = len(gen_tokenizer)

    bias = torch.zeros(vocab_size, dtype=torch.float32, device=device)

    # Fill from map; ignore indices outside vocab range
    for gen_id_str, val in qa_to_gen_map.items():
        try:
            gen_id = int(gen_id_str)
        except Exception:
            continue
        if 0 <= gen_id < vocab_size:
            bias[gen_id] = float(val)
        # else: silently skip or log if you have a logger

    return bias


def generate_answer_with_static_bias(
    query: str,
    passages: list[str],
    qa_model,
    qa_tokenizer,
    gen_model,
    gen_tokenizer,
    qa_to_gen_map: dict[int, list[int]],
    max_length: int = 200,
    temperature: float = 0.8,
    top_p: float = 0.9,
    device: torch.device = torch.device("cpu"),
) -> str:
    # ---------- QA forward ----------
    qa_input = qa_tokenizer(
        [query + " " + " ".join(passages)],
        return_tensors="pt",
        truncation=True,
        max_length=512,
    ).to(device)
    with torch.no_grad():
        qa_out = qa_model(**qa_input)
    print("qa_out=", qa_out)
    # #
    # # if hasattr(qa_out, "logits"):
    # #     qa_logits = qa_out.logits.squeeze(0)
    # # elif hasattr(qa_out, "start_logits"):
    # #     qa_logits = qa_out.start_logits.squeeze(0)
    # # else:
    # #     raise AttributeError(
    # #         "QA model output lacks 'logits' and 'start_logits' attributes."
    # #     )
    #
    # # ---------- Bias processor ----------
    # # bias_processor = QAStaticBiasProcessor(
    # #     qa_logits=qa_logits,
    # #     qa_to_gen_map=qa_to_gen_map,
    # #     vocab_gen=gen_tokenizer.vocab_size,
    # #     agg_fn=torch.mean,
    # #     scale=0.8,
    # # )
    # bias_vec = _build_bias_vector(qa_to_gen_map, gen_tokenizer, device)
    # logits_processor = LogitsProcessorList([StaticBiasProcessor(bias_vec)])
    #
    # # ---------- Generation ----------
    # gen_ids = gen_tokenizer.encode(query, return_tensors="pt").to(device)
    # out_ids = gen_model.generate(
    #     gen_ids,
    #     max_length=max_length,
    #     logits_processor=logits_processor,
    #     do_sample=True,
    #     temperature=temperature,
    #     top_p=top_p,
    # )
    # return gen_tokenizer.decode(out_ids[0], skip_special_tokens=True)
    return "===== generate_answer_with_static_bias ====="


def generate_without_bias(
    query: str,
    passages: list[str],
    gen_model,
    gen_tokenizer,
    max_length: int = 200,
    temperature: float = 0.8,
    top_p: float = 0.9,
    device: torch.device = torch.device("cpu"),
) -> str:
    """
    Generate text using the generator model **without** any bias processor.
    """
    # # Encode the query (the passages are ignored for the plain generator)
    # input_str = "".join([query + " " + " ".join(passages)])
    # gen_ids = gen_tokenizer.encode(input_str, return_tensors="pt").to(device)
    # out_ids = gen_model.generate(
    #     gen_ids,
    #     max_length=max_length,
    #     do_sample=True,
    #     temperature=temperature,
    #     top_p=top_p,
    # )
    # return gen_tokenizer.decode(out_ids[0], skip_special_tokens=True)
    return "===== generate_without_bias ====="
