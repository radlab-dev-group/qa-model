"""
build_model_quantization.py

A tiny command‑line wrapper around :class:`qa_utils.builder.quantizer.ModelQuantizer`.

Typical usage
-------------
    $ python -m qa_utils.builder.quantize_model \
        --model /path/to/roberta-large \
        --method gptq \
        --calibration /path/to/calibration.json \
        --force

Multiple methods can be supplied as a comma‑separated list:

    $ python -m qa_utils.builder.quantize_model \
        --model /path/to/bert-base-uncased \
        --method bitsandbytes,gptq,awq \
        --calibration /data/calib.json
"""

import argparse
import sys
from pathlib import Path
from typing import List, Optional

from builder.quant_model import ModelQuantizer


def _parse_methods(methods: str) -> List[str]:
    """
    Accept a comma‑separated string (or a single value) and return a clean list.
    Empty entries are ignored so ``"gptq,,awq"`` becomes ``["gptq","awq"]``.
    """
    return [m.strip().lower() for m in methods.split(",") if m.strip()]


def _existing_dir(path: str) -> Path:
    """Validate that *path* exists and is a directory."""
    p = Path(path).expanduser().resolve()
    if not p.is_dir():
        raise argparse.ArgumentTypeError(f"'{path}' is not a readable directory")
    return p


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Quantise a Hugging‑Face checkpoint with bits‑and‑bytes, GPTQ or AWQ.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    parser.add_argument(
        "-m",
        "--model",
        required=True,
        type=_existing_dir,
        help="Path to the model checkpoint (directory that contains config.json, pytorch_model.bin, …).",
    )

    parser.add_argument(
        "-t",
        "--method",
        required=True,
        type=_parse_methods,
        help=(
            "Quantisation method(s) to apply.  Supported values are "
            "'bitsandbytes', 'gptq' and 'awq'.  Supply several methods as a "
            "comma‑separated list (no spaces required)."
        ),
    )

    parser.add_argument(
        "-c",
        "--calibration",
        type=str,
        default=None,
        help=(
            "Path to a calibration dataset (any format accepted by the 🤗 "
            "`datasets` library). Required for GPTQ and AWQ; ignored for bits‑and‑bytes."
        ),
    )

    parser.add_argument(
        "-f",
        "--force",
        action="store_true",
        help="Overwrite an existing quantised checkpoint if it already exists.",
    )

    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Suppress the informational prints emitted by the quantiser.",
    )

    return parser


def main(argv: Optional[List[str]] = None) -> int:
    """
    Parse CLI arguments, instantiate :class:`ModelQuantizer` and run the requested
    quantisation(s).  Returns an exit‑code (0 = success, 1 = error).
    """
    parser = build_parser()
    args = parser.parse_args(argv)

    # --------------------------------------------------------------------- #
    #  Validate user‑provided values
    # --------------------------------------------------------------------- #
    model_path: Path = args.model
    methods: List[str] = args.method

    # The ModelQuantizer itself will raise a clear error if an unknown method
    # is requested, but we can give a nicer early‑exit message.
    known_methods = {"bitsandbytes", "gptq", "awq"}
    unknown = [m for m in methods if m not in known_methods]
    if unknown:
        sys.stderr.write(
            f"Error: unknown quantisation method(s): {', '.join(unknown)}\n"
            f"Supported methods are: {', '.join(sorted(known_methods))}\n"
        )
        return 1

    # --------------------------------------------------------------------- #
    #  Build the quantiser
    # --------------------------------------------------------------------- #
    quantizer = ModelQuantizer(
        model_path=str(model_path),
        quant_methods=methods,
        calibration_path=args.calibration,
    )

    # --------------------------------------------------------------------- #
    #  Run the job
    # --------------------------------------------------------------------- #
    # If the user asked for a single method we call ``quantize_one`` so the
    # ``--force`` flag has an effect.  For more than one method we delegate to
    # ``quantize_all`` (which internally loops over the list).
    if len(methods) == 1:
        quantizer.quantize_one(methods[0], force=args.force)
    else:
        quantizer.quantize_all(force=args.force)

    return 0


if __name__ == "__main__":
    sys.exit(main())
