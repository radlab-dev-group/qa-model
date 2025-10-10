import torch
import argparse

from typing import Optional

from transformers import (
    AutoModelForQuestionAnswering,
    AutoTokenizer,
    AutoModelForCausalLM,
    BitsAndBytesConfig,
)

from utils.device import get_device
from utils.token_map_io import load_map
from utils.generate import generate_answer_with_static_bias, generate_without_bias


# ----------------------------------------------------------------------
# UI layer
class ConsoleUI:
    """Simple console UI for prompting and displaying results."""

    @staticmethod
    def print_header() -> None:
        print("\nModels loaded. Enter 'quit' at any prompt to exit.\n")

    @staticmethod
    def prompt_context() -> str:
        return input("Context (press Enter for empty): ").strip()

    @staticmethod
    def prompt_question() -> str:
        return input("Question: ").strip()

    @staticmethod
    def print_separator() -> None:
        print("-" * 60)

    @staticmethod
    def display_answers(with_bias: str, without_bias: str) -> None:
        ConsoleUI.print_separator()
        print("🟢 Answer WITH QA‑bias:")
        print(with_bias)
        ConsoleUI.print_separator()
        print("🔵 Answer WITHOUT QA‑bias:")
        print(without_bias)
        ConsoleUI.print_separator()


# ----------------------------------------------------------------------
# Model loading
class ModelLoader:
    """Loads QA and generator models together with their tokenizers."""

    def __init__(self, qa_path: str, gen_path: str, device: Optional[str] = None):
        self.qa_path = qa_path
        self.gen_path = gen_path
        self.device = device or get_device()

    def load(self):
        print("\nLoading models, this may take a while ...")
        # dtype = torch.float16 if self.device.type == "cuda" else torch.float32

        qa_model = AutoModelForQuestionAnswering.from_pretrained(self.qa_path)
        # qa_model = qa_model.to(self.device)
        qa_tokenizer = AutoTokenizer.from_pretrained(self.qa_path)

        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
        )
        gen_model = AutoModelForCausalLM.from_pretrained(
            self.gen_path,
            torch_dtype=torch.float16,
            quantization_config=bnb_config,
            device_map=self.device,
        )
        gen_model = gen_model.to(self.device)
        gen_tokenizer = AutoTokenizer.from_pretrained(self.gen_path)

        return qa_model, qa_tokenizer, gen_model, gen_tokenizer, self.device


# ----------------------------------------------------------------------
# Core application logic
class QAInteractiveApp:
    """Orchestrates the interactive loop, tying together UI, models and generation."""

    def __init__(
        self,
        qa_model_path: str,
        gen_model_path: str,
        cache_dir: str,
        qa_name: str,
        gen_name: str,
    ):
        self.qa_model_path = qa_model_path
        self.gen_model_path = gen_model_path
        self.cache_dir = cache_dir
        self.qa_name = qa_name
        self.gen_name = gen_name

        # Load token map
        self.qa_to_gen = load_map(self.cache_dir, self.qa_name, self.gen_name)

        # Load models/tokenizers
        loader = ModelLoader(self.qa_model_path, self.gen_model_path)
        (
            self.qa_model,
            self.qa_tokenizer,
            self.gen_model,
            self.gen_tokenizer,
            self.device,
        ) = loader.load()

    def run(self) -> None:
        ConsoleUI.print_header()
        while True:
            print("=" * 60)
            context = ConsoleUI.prompt_context()
            if context.lower() == "quit":
                break

            question = ConsoleUI.prompt_question()
            if question.lower() == "quit":
                break

            passages = [context] if context else []

            # Generation with static bias
            answer_with_bias = generate_answer_with_static_bias(
                query=question,
                passages=passages,
                qa_model=self.qa_model,
                qa_tokenizer=self.qa_tokenizer,
                gen_model=self.gen_model,
                gen_tokenizer=self.gen_tokenizer,
                qa_to_gen_map=self.qa_to_gen,
                device=self.device,
            )

            # Generation without bias
            answer_without_bias = generate_without_bias(
                query=question,
                passages=passages,
                gen_model=self.gen_model,
                gen_tokenizer=self.gen_tokenizer,
                device=self.device,
            )

            ConsoleUI.display_answers(answer_with_bias, answer_without_bias)

        print("Goodbye!")


# ----------------------------------------------------------------------
# Entry point ------------------------------------------------------------
def main() -> None:
    """
    CLI entry point.

    Arguments:
        --gen-model-path   Path or identifier of the generator (causal LM) model.
        --qa-model-path    Path or identifier of the QA model.
        --cache-dir        Directory containing the token‑map cache.
        --qa-name          Name of the QA tokenizer/model used in the map file.
        --gen-name         Name of the generator tokenizer/model used in the map file.
    """
    parser = argparse.ArgumentParser(
        description="Interactive QA‑biased generation (with and without bias)."
    )
    parser.add_argument(
        "--gen-model-path",
        type=str,
        required=True,
        help="Path or hub identifier of the generator model (e.g., meta-llama/Llama-2-7b-hf).",
    )
    parser.add_argument(
        "--qa-model-path",
        type=str,
        required=True,
        help="Path or hub identifier of the QA model (e.g., roberta-base).",
    )
    parser.add_argument(
        "--cache-dir",
        type=str,
        required=True,
        help="Directory where the token‑map JSON files are stored.",
    )
    parser.add_argument(
        "--qa-name",
        type=str,
        required=True,
        help="Name of the QA model/tokenizer used in the map filename.",
    )
    parser.add_argument(
        "--gen-name",
        type=str,
        required=True,
        help="Name of the generator model/tokenizer used in the map filename.",
    )
    args = parser.parse_args()

    app = QAInteractiveApp(
        qa_model_path=args.qa_model_path,
        gen_model_path=args.gen_model_path,
        cache_dir=args.cache_dir,
        qa_name=args.qa_name,
        gen_name=args.gen_name,
    )
    app.run()


if __name__ == "__main__":
    main()
# ... existing code ...
