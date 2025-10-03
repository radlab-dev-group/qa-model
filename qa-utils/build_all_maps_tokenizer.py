"""
CLI entry point – delegates the heavy lifting to TokenMapBuilder.
"""

import click
from pathlib import Path

from builder.tokens_mapping_builder import TokenMapBuilder

# ------------------------------------------------------------------
# Configuration – model paths / identifiers
# ------------------------------------------------------------------
ROBERTA_MODELS = [("/mnt/data2/llms/models/radlab-open/qa/best_model", "radlab-qa")]

GENAI_MODELS = [
    ("/mnt/data2/llms/models/community/google/gemma-3-12b-it", "google-gemma3"),
    ("/mnt/data2/llms/models/community/openai/gpt-oss-20b", "openai-gpt-oss"),
    (
        "/mnt/data2/llms/models/community/llama-3/Meta-Llama-3.1-8B-Instruct",
        "meta-llama3.1",
    ),
    (
        "/mnt/data2/llms/models/community/llama-3/Meta-Llama-3.2-3B-Instruct",
        "meta-llama3.2",
    ),
    (
        "/mnt/data2/llms/models/community/llama-3/Llama-3.3-70B-Instruct",
        "meta-llama3.3",
    ),
]


# ------------------------------------------------------------------
# CLI
# ------------------------------------------------------------------
@click.command()
@click.option(
    "--force-rebuild",
    is_flag=True,
    help="Re‑build all maps, ignoring existing files and metadata.",
)
def main(force_rebuild: bool) -> None:
    """
    Run the full mapping build process.
    """
    # Resolve cache root relative to the project root
    cache_root = Path(__file__).parents[1] / "cache"

    builder = TokenMapBuilder(
        roberta_models=ROBERTA_MODELS,
        genai_models=GENAI_MODELS,
        cache_root=cache_root,
    )
    builder.build_all(force_rebuild=force_rebuild)


if __name__ == "__main__":
    main()
