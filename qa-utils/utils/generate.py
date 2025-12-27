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
            # ---------------------------------------------------------
            # NOTE: In some contexts `val` may be a list of generator IDs
            # (e.g., when `qa_to_gen_map` maps QA token IDs to a list of
            # generator token IDs). Converting a list directly to `float`
            # raises a TypeError. To make the function robust we handle
            # both scalar and list values:
            #   * If `val` is already a numeric scalar, use it directly.
            #   * If `val` is a list/tuple, use its length as a simple
            #     numeric bias (it can be replaced with another aggregation
            #     such as sum, mean, etc., depending on the use‑case).
            # ---------------------------------------------------------
            if isinstance(val, (list, tuple)):
                bias_value = float(len(val))
            else:
                bias_value = float(val)
            bias[gen_id] = bias_value

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
    use_logits_processor: bool = True,
) -> str:
    # # ---------- QA forward ----------
    # qa_input = qa_tokenizer(
    #     [query + " " + " ".join(passages)],
    #     return_tensors="pt",
    #     truncation=True,
    #     max_length=512,
    # ).to(device)
    # with torch.no_grad():
    #     qa_out = qa_model(**qa_input)
    #
    # if hasattr(qa_out, "logits"):
    #     qa_logits = qa_out.logits.squeeze(0)
    # elif hasattr(qa_out, "start_logits"):
    #     qa_logits = qa_out.start_logits.squeeze(0)
    # else:
    #     raise AttributeError(
    #         "QA model output lacks 'logits' and 'start_logits' attributes."
    #     )
    #
    # # ---------- Bias processor ----------
    # bias_processor = QAStaticBiasProcessor(
    #     qa_logits=qa_logits,
    #     qa_to_gen_map=qa_to_gen_map,
    #     vocab_gen=gen_tokenizer.vocab_size,
    #     agg_fn=torch.mean,
    #     scale=0.8,
    # )
    # bias_vec = _build_bias_vector(qa_to_gen_map, gen_tokenizer, device)
    #
    # # Build the logits processor list only when requested
    # if use_logits_processor:
    #     logits_processor = LogitsProcessorList(
    #         [bias_processor, StaticBiasProcessor(bias_vec)]
    #     )
    #     print("logits_processor=", logits_processor)
    # else:
    #     logits_processor = None
    #     print("logits_processor disabled")
    #
    # # ---------- Generation ----------
    # chat_input = {
    #     "role": "user",
    #     "content": query,
    # }
    #
    # chat_ids = gen_tokenizer.apply_chat_template(
    #     [chat_input],
    #     tokenize=True,
    #     add_generation_prompt=True,
    #     return_tensors="pt",
    # ).to(device)
    #
    # generate_kwargs = {
    #     "input_ids": chat_ids,
    #     "max_length": max_length,
    #     "do_sample": True,
    #     "temperature": temperature,
    #     "top_p": top_p,
    # }
    # if logits_processor is not None:
    #     generate_kwargs["logits_processor"] = logits_processor
    #
    # out_ids = gen_model.generate(**generate_kwargs)
    # return gen_tokenizer.decode(out_ids[0], skip_special_tokens=True)
    return "=== generate_answer_with_static_bias ==="


class ClampLogitsProcessor(LogitsProcessor):
    """
    Clamp logits to a safe range before the softmax.
    Prevents `inf`/`nan` values that break CUDA sampling.
    """

    def __init__(self, min_val: float = -1e4, max_val: float = 1e4):
        self.min_val = min_val
        self.max_val = max_val

    def __call__(
        self, input_ids: torch.Tensor, scores: torch.Tensor
    ) -> torch.Tensor:
        # Clamp and clean NaNs / infinities
        scores = scores.clamp(self.min_val, self.max_val)
        return torch.nan_to_num(scores, nan=0.0, posinf=0.0, neginf=0.0)


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
    # ---------- Generation ----------
    system_prompt = {
        "role": "system",
        "content": f"Na podstawie treści dostarczonych przez użytkownika odpowiedz na pytanie: {query}",
    }

    chat_input = {
        "role": "user",
        "content": " ".join(passages),
    }
    chat_ids = gen_tokenizer.apply_chat_template(
        [system_prompt, chat_input],
        tokenize=True,
        add_generation_prompt=True,
        return_tensors="pt",
        return_dict=True,
    ).to(device=device)

    print("chat_ids=", chat_ids)

    inputs = {
        # "max_length": max_length,
        "max_new_tokens": 1024,
        "do_sample": True,
        "temperature": temperature,
        "top_p": top_p,
    }
    input_len = chat_ids["input_ids"].shape[-1]

    # logits_processor = LogitsProcessorList(
    #     [ClampLogitsProcessor(min_val=-1e4, max_val=1e4)]
    # )
    # generate_kwargs["logits_processor"] = logits_processor
    #
    with torch.inference_mode():
        generation = gen_model.generate(**chat_ids,**inputs)
        generation = generation[0][input_len:]
        # generation = generation[0]
    return gen_tokenizer.decode(generation, skip_special_tokens=True)
    # return generation
