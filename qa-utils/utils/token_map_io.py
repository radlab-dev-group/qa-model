import json
from pathlib import Path
from typing import Dict, List


def load_map(path: str, roberta_name: str, gen_name: str) -> Dict[int, List[int]]:
    """
    Load a pre‑computed token map from ``roberta_name__gen_name.json``.
    Returns a dictionary mapping integers to lists of integers (keys are ``int``,
    not ``str``).

    Raises
    ------
    FileNotFoundError
        If the map file does not exist – in that case run
        ``scripts/build_all_maps_tokenizer.py`` first.
    """
    file_name = f"{roberta_name}__{gen_name}.json"
    if "cache" in path:
        map_path = Path(path) / "maps" / file_name
    else:
        map_path = Path(path) / "cache" / "maps" / file_name

    raw = json.loads(map_path.read_text())
    return {int(k): v for k, v in raw.items()}


def get_available_pairs(path: str) -> List[tuple[str, str]]:
    """
    Scan the ``cache/maps`` directory and return a list of available
    (roberta_name, gen_name) pairs.
    """
    if "cache" not in path:
        maps_dir = Path(path) / "cache" / "maps"
    else:
        maps_dir = Path(path) / "maps"

    pairs = []
    for p in maps_dir.glob("*.json"):
        name = p.stem
        roberta, gen = name.split("__")
        pairs.append((roberta, gen))
    return pairs
