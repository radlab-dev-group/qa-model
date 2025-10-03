from __future__ import annotations

import argparse
import sys
from pathlib import Path

from transformers import pipeline

default_model_path = "radlab/polish-qa-v2"
# default_model_path = "/mnt/data2/llms/models/radlab-open/qa/best_model"


def load_question_answerer(model_path: str):
    """Create a HuggingFace ``question‑answering`` pipeline."""
    return pipeline("question-answering", model=model_path)


def answer_question(
    question: str,
    context: str,
    question_answerer,
) -> dict:
    """
    Run the ``question‑answering`` pipeline.

    Parameters
    ----------
    question: str
        The question to be answered.
    context: str
        The passage containing the answer.
    question_answerer: Pipeline
        The HuggingFace pipeline created by :func:`load_question_answerer`.

    Returns
    -------
    dict
        The pipeline output (score, start, end, answer).
    """
    # Remove newline characters to match the original script’s behaviour.
    cleaned_context = context.replace("\n", " ")
    return question_answerer(question=question, context=cleaned_context)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse command‑line arguments."""
    parser = argparse.ArgumentParser(
        description="Simple QA inference script using a HuggingFace model."
    )
    parser.add_argument(
        "question",
        help="The question to ask the model.",
    )
    parser.add_argument(
        "context",
        nargs="?",
        help=(
            "The context string. If omitted, the script reads from STDIN. "
            "Use '-' to explicitly read from STDIN."
        ),
    )
    parser.add_argument(
        "--model",
        default=default_model_path,
        help="Path or identifier of the HuggingFace model.",
    )
    parser.add_argument(
        "--context-file",
        type=Path,
        help="Path to a file containing the context. Overrides the positional CONTEXT argument.",
    )
    return parser.parse_args(argv)


def main() -> None:
    """Entry point for the command‑line interface."""
    args = parse_args()

    qa = load_question_answerer(args.model)
    if args.context_file:
        context = args.context_file.read_text(encoding="utf-8")
    elif args.context is None or args.context == "-":
        context = sys.stdin.read()
    else:
        context = args.context

    if not context:
        sys.stderr.write("Error: No context provided.\n")
        sys.exit(1)

    result = answer_question(args.question, context, qa)
    print(result)


if __name__ == "__main__":
    main()
