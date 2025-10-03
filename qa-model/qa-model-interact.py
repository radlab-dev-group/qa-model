"""
Interactive QA script.

This module provides a simple command‑line interface for performing
question‑answering with a pre‑trained HuggingFace model.  The user is
prompted repeatedly for a context and a question; the model returns
the most likely answer span.  The script is intended for quick
experimentation and debugging.
"""

from transformers import pipeline


def main() -> None:
    """Run an interactive question‑answering session.

    The function creates a ``pipeline`` for the ``question‑answering``
    task, then enters an infinite loop prompting the user for a
    context and a question.  The entered text is stripped of newline
    characters before being passed to the pipeline.  The result is
    printed to standard output.
    """
    model_path = "radlab/polish-qa-v2"
    # model_path = "/mnt/data2/llms/models/radlab-open/qa/best_model"
    # model_path = "/mnt/data2/llms/models/radlab-open/qa/best_model/quantized/best_model-bitsandbytes"
    question_answerer = pipeline("question-answering", model=model_path)

    while True:
        print(50 * "=")
        context = input("Context:")
        question = input("Question:")
        print(50 * "-")
        print(
            question_answerer(
                question=question,
                context=context.replace("\n", " "),
            )
        )


if __name__ == "__main__":
    main()
