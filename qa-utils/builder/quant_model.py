"""
A small, extensible library for quantizing Hugging Face models with popular
techniques (currently *bitsandbytes*).  The public entry point is the
:class:`ModelQuantizer` class.

Typical usage
-------------
    from quantizer import ModelQuantizer

    quantizer = ModelQuantizer(
        model_path="models/bert-base-uncased",
        quant_methods=["bitsandbytes"],
    )
    quantizer.quantize_all()
"""

import os
from pathlib import Path
from typing import List, Optional

from transformers import (
    AutoConfig,
    AutoModel,
    AutoModelForCausalLM,
    AutoModelForSeq2SeqLM,
    AutoModelForQuestionAnswering,
)

from .quant_strategies import (
    BitsAndBytesStrategy,
    GPTQStrategy,
    AWQStrategy,
    NotImplementedStrategy,
    QuantizationStrategy,
)


class ModelQuantizer:
    """
    High‑level helper that loads a Hugging Face model once and applies one
    or more quantization strategies to it.

    The quantized artefacts are stored under::

        <model_path>/quantized/<model_name>-<quant_method>/

    Parameters
    ----------
    model_path : str | pathlib.Path
        Path to the directory that holds the original model.
    quant_methods : list[str]
        List of quantization identifiers (e.g. ``["bitsandbytes", "gptq"]``).
        Unknown identifiers are treated as *not implemented* and will raise
        a clear error.
    calibration_path : str | None, optional
        Path to a calibration dataset used by GPTQ and AWQ.  If
        provided it is passed to the corresponding strategy
        constructors; otherwise those strategies will raise a
        ``ValueError`` when invoked.
    """

    # Mapping from method name → strategy class
    _STRATEGY_REGISTRY = {
        "bitsandbytes": BitsAndBytesStrategy,
        "gptq": GPTQStrategy,
        "awq": AWQStrategy,
    }

    def __init__(
        self,
        model_path: str,
        quant_methods: List[str],
        calibration_path: Optional[str] = None,
    ):
        self.model_path = Path(model_path).resolve()
        if not self.model_path.is_dir():
            raise FileNotFoundError(f"Model directory not found: {self.model_path}")

        self.model_name = self.model_path.name
        self.quant_methods = [m.lower() for m in quant_methods]
        self.quantized_root = self.model_path / "quantized"
        self.calibration_path = calibration_path
        self.original_model = self._load_model(str(self.model_path))

    def quantize_all(self, force: bool = False):
        """
        Apply every quantization method supplied at construction time.

        Parameters
        ----------
        force : bool, optional
            If ``True``, re‑quantize even when the target directory already
            exists.  Default is ``False`` (skip already‑quantized artefacts).
        """
        for method in self.quant_methods:
            self.quantize_one(method, force=force)

    def quantize_one(self, method: str, force: bool = False):
        """
        Quantize the model with a single *method* and store the result.

        Parameters
        ----------
        method : str
            Quantization identifier (case‑insensitive).
        force : bool, optional
            Overwrite existing output if ``True``.  Default ``False``.
        """
        method = method.lower()
        if method not in self.quant_methods:
            raise ValueError(f"Method '{method}' was not requested at init time.")

        print(f"🔧 Quantizing with method: {method}")

        strategy = self._get_strategy(method)
        try:
            quantized_model = strategy.apply(str(self.model_path))
        except NotImplementedError as nie:
            print(f"⚠️  Skipping {method}: {nie}")
            return
        except Exception as exc:
            print(f"❌  Quantization failed for {method}: {exc}")
            return

        out_dir = self.quantized_root / f"{self.model_name}-{method}"
        if out_dir.is_dir() and not force:
            print(f"✅  Skipping {method}: output already exists at {out_dir}")
            return

        print(f"💾 Saving quantized model to {out_dir}")
        self._save_model(quantized_model, out_dir)

        print(f"✅  Finished {method}\n")

    @staticmethod
    def _load_model(model_path: str):
        """
        Load a model (any architecture) from *model_path* using the most
        appropriate ``AutoModel`` class.

        Parameters
        ----------
        model_path : str
            Directory with the pretrained model.

        Returns
        -------
        transformers.PreTrainedModel
            The loaded model instance.
        """
        config = AutoConfig.from_pretrained(model_path)

        # Choose a loader based on the model type; fallback to generic AutoModel.
        if getattr(config, "model_type", None) in {
            "gpt2",
            "gpt_neox",
            "opt",
            "bloom",
            "falcon",
            "llama",
        }:
            return AutoModelForCausalLM.from_pretrained(model_path)
        if getattr(config, "model_type", None) in {
            "t5",
            "bart",
            "marian",
            "mbart",
        }:
            return AutoModelForSeq2SeqLM.from_pretrained(model_path)

        # QA models that expose a ``question_answering_head``
        if (
            getattr(config, "model_type", None)
            in {
                "bert",
                "roberta",
                "albert",
                "deberta",
                "deberta-v2",
                "electra",
                "xlnet",
                "xlm-roberta",
            }
            and getattr(config, "architectures", [])
            and any("ForQuestionAnswering" in arch for arch in config.architectures)
        ):
            return AutoModelForQuestionAnswering

        return AutoModel.from_pretrained(model_path)

    @staticmethod
    def _save_model(model, save_dir: Path):
        """
        Persist *model* to *save_dir*.

        Parameters
        ----------
        model : transformers.PreTrainedModel
            Model to be saved.
        save_dir : pathlib.Path
            Destination directory; created if it does not exist.
        """
        os.makedirs(save_dir, exist_ok=True)
        model.save_pretrained(str(save_dir))

    def _get_strategy(self, method: str) -> QuantizationStrategy:
        """
        Resolve *method* to a concrete strategy instance.

        For GPTQ/AWQ the optional ``calibration_path`` supplied at
        construction time is forwarded to the strategy constructor.
        """
        if method not in self._STRATEGY_REGISTRY:
            # Unknown but allowed – give a helpful placeholder error.
            return NotImplementedStrategy(method)

        StrategyCls = self._STRATEGY_REGISTRY[method]

        # GPTQ and AWQ require a calibration dataset; pass it if available.
        if method in {"gptq", "awq"}:
            return StrategyCls(calibration_path=self.calibration_path)

        return StrategyCls()
