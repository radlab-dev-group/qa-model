import torch


def get_device() -> torch.device:
    """Returns available GPU or CPU."""
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")
