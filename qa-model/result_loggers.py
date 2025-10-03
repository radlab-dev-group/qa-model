"""
Result loggers module.

This module provides utilities for configuring Weights & Biases (WandB)
logging for the RadLab QA model training pipeline.  The primary class
`RadLabQAModelWandbConfig` encapsulates project‑level constants and any
future configuration options needed by the `WanDBHandler`.
"""


class WandbConfig:
    PREFIX_RUN = None
    BASE_RUN_NAME = None
    PROJECT_NAME = None
    PROJECT_TAGS = None


class RadLabQAModelWandbConfig(WandbConfig):
    """
    Configuration holder for WandB logging.

    Attributes
    ----------
    PROJECT_NAME : str
        The WandB project name under which runs are grouped.
    BASE_RUN_NAME : str
        Base name used for each run; can be combined with timestamps or
        other identifiers by the caller.
    """

    PREFIX_RUN = ""
    BASE_RUN_NAME = "QAMODEL"
    PROJECT_NAME = "radlab-qa-model"
    PROJECT_TAGS = ["QA", "LLM", "RadLab", "HuggingFace"]
