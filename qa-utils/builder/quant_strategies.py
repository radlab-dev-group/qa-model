from pathlib import Path
from typing import Protocol, runtime_checkable, Optional

from transformers import (
    AutoConfig,
    BitsAndBytesConfig,
    AutoModelForQuestionAnswering,
)


@runtime_checkable
class QuantizationStrategy(Protocol):
    """
    Protocol that all quantization strategy classes must follow.

    Implementations receive the original model path and must return a
    *transformers* model instance that is ready to be saved.
    """

    def apply(self, original_path: str):
        """
        Apply the quantization to the model located at *original_path*.

        Parameters
        ----------
        original_path : str
            Path to the original, un‑quantized model directory.

        Returns
        -------
        transformers.PreTrainedModel
            The quantized model instance.
        """
        ...


class CalibrationDatasetMixin:
    """
    Mixin providing a reusable ``_load_calibration_dataset`` implementation
    for strategies that require a calibration dataset (e.g. GPTQ, AWQ).

    The mixin stores ``calibration_path`` and offers a single method to
    load the dataset using ``datasets.load_dataset``.  The logic is the
    same for all strategies, so it lives here to avoid duplication.
    """

    def __init__(self, calibration_path: Optional[str] = None):
        """
        Parameters
        ----------
        calibration_path : str | None, optional
            Path to a calibration dataset.  If ``None`` the strategy cannot run.
        """
        self.calibration_path = calibration_path

    def _load_calibration_dataset(self):
        """
        Load the calibration dataset using ``datasets.load_dataset``.
        The dataset must contain a column named ``input_ids`` (or any
        column that can be tokenised by the model).  The method returns
        the dataset object ready for the quantiser.
        """
        from datasets import load_dataset

        if not self.calibration_path:
            raise ValueError(
                "Calibration dataset is required for this quantisation strategy. "
                "Pass the path when constructing the strategy instance."
            )

        ext = Path(self.calibration_path).suffix.lower()
        if ext in {".json", ".jsonl"}:
            return load_dataset("json", data_files=self.calibration_path)["train"]
        if ext == ".csv":
            return load_dataset("csv", data_files=self.calibration_path)["train"]

        # Fallback – assume a HuggingFace repo identifier or generic format.
        return load_dataset(self.calibration_path)["train"]


class BitsAndBytesStrategy(QuantizationStrategy):
    """
    8‑bit quantization using the ``bitsandbytes`` integration
    provided by :mod:`transformers`.

    The strategy simply reloads the model with ``load_in_8bit=True`` and
    ``device_map="auto"``, which internally uses bitsandbytes.
    """

    def apply(self, original_path: str):
        """
        Load the model in 8‑bit mode.

        Parameters
        ----------
        original_path : str
            Directory containing the original model files.

        Returns
        -------
        transformers.PreTrainedModel
            The 8‑bit model.
        """
        config = AutoConfig.from_pretrained(original_path)
        bnb_cfg = BitsAndBytesConfig(
            load_in_8bit=True,
            bnb_8bit_use_double_quant=True,
            bnb_8bit_excluded_modules=["qa_outputs"],
        )

        return AutoModelForQuestionAnswering.from_pretrained(
            original_path,
            config=config,
            quantization_config=bnb_cfg,
            device_map="auto",
        )


class GPTQStrategy(QuantizationStrategy, CalibrationDatasetMixin):
    """
    This implementation relies on the *optimum* library (Intel
    optimisation stack).  If the library is not available a clear
    ``NotImplementedError`` is raised with installation instructions.

    The strategy expects a *calibration dataset* to be provided via the
    ``calibration_path`` argument.  The dataset should be a Hugging Face
    ``datasets`` compatible file (e.g. a JSON, CSV or a prepared
    ``Dataset`` saved with ``datasets``).  When the path is omitted a
    ``ValueError`` is raised – quantisation without calibration would
    produce poor results.
    """

    def apply(self, original_path: str):
        """
        Apply GPTQ quantisation and return a quantised model instance.

        Parameters
        ----------
        original_path : str
            Path to the original, un‑quantized model directory.

        Returns
        -------
        transformers.PreTrainedModel
            The GPTQ‑quantised model ready for ``save_pretrained``.

        Raises
        ------
        NotImplementedError
            If the required ``optimum`` backend is not installed.
        """
        try:
            # The Intel GPTQ implementation lives in optimum.intel.gptq
            from optimum.intel.gptq import GPTQQuantizer  # type: ignore
        except Exception as exc:
            raise NotImplementedError(
                "GPTQ quantisation requires the 'optimum' library with Intel "
                "GPTQ support. Install it via "
                "`pip install optimum[intel]` and retry."
            ) from exc

        calibration_dataset = self._load_calibration_dataset()
        quantizer = GPTQQuantizer.from_pretrained(original_path)
        quantizer.quantize(
            calibration_dataset,
            batch_size=8,
            num_calibration_steps=128,  # a modest number for quick runs
        )
        return quantizer.model


class AWQStrategy(QuantizationStrategy, CalibrationDatasetMixin):
    """
    The implementation currently targets the *optimum* AWQ backend
    (``optimum.onnxruntime``).  Like GPTQ, it needs a calibration
    dataset; otherwise a ``ValueError`` is raised.
    """

    def apply(self, original_path: str):
        """
        Apply AWQ quantisation and return the quantised model.

        Parameters
        ----------
        original_path : str
            Path to the original, un‑quantized model directory.

        Returns
        -------
        transformers.PreTrainedModel
            The AWQ‑quantised model.

        Raises
        ------
        NotImplementedError
            If the required AWQ backend is missing.
        """
        try:
            # The AWQ implementation lives in optimum.onnxruntime.quantization
            from optimum.onnxruntime import AWQQuantizer  # type: ignore
        except Exception as exc:
            raise NotImplementedError(
                "AWQ quantisation requires the 'optimum' library with the "
                "ONNX‑Runtime AWQ backend. Install it via "
                "`pip install optimum[onnxruntime]` and retry."
            ) from exc

        calibration_dataset = self._load_calibration_dataset()
        quantizer = AWQQuantizer.from_pretrained(original_path)
        quantizer.quantize(
            calibration_dataset,
            batch_size=8,
            num_calibration_steps=128,
        )
        return quantizer.model


class NotImplementedStrategy(QuantizationStrategy):
    """
    Placeholder for not‑yet‑implemented quantization methods.
    Raises :class:`NotImplementedError` when called.
    """

    def __init__(self, name: str):
        self.name = name

    def apply(self, original_path: str):
        raise NotImplementedError(
            f"{self.name} quantization is not implemented yet."
        )
