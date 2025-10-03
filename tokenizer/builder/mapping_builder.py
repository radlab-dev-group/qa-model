"""
Utilities for constructing and caching token‑mapping tables between
RoBERTa‑style QA tokenizers and generative‑AI tokenizers.

The module defines :class:`TokenMapBuilder`, which can generate a
JSON‑serialisable mapping from QA token IDs to one or more generator
token IDs, cache the results on disk, and keep metadata about the
hashes of the tokenizers used to create each map.  This enables fast
reuse of mappings across runs while automatically rebuilding them
when the underlying tokenizers change.
"""

import json
import hashlib
from pathlib import Path
from typing import List, Dict, Tuple

from tqdm import tqdm
from transformers import AutoTokenizer


class TokenMapBuilder:
    """
    Initialise a :class:`TokenMapBuilder`.

    Parameters
    ----------
    roberta_models : List[Tuple[str, str]]
        A list of ``(model_path, model_name)`` tuples for the QA
        (RoBERTa‑style) tokenizers. ``model_path`` is passed to
        ``AutoTokenizer.from_pretrained`` and ``model_name`` is used
        to compose the cache key.

    genai_models : List[Tuple[str, str]]
        A list of ``(model_path, model_name)`` tuples for the
        generative‑AI tokenizers, interpreted in the same way as
        ``roberta_models``.

    cache_root : Path | str, optional
        Root directory for cached mapping files and metadata.  By
        default it is set to ``<project_root>/cache``.  The directory
        structure will contain a ``maps`` sub‑directory for the JSON
        maps and a ``meta`` sub‑directory for the metadata file.
    """

    def __init__(
        self,
        roberta_models: List[Tuple[str, str]],
        genai_models: List[Tuple[str, str]],
        cache_root: Path | str = Path(__file__).parents[2] / "cache",
    ) -> None:
        self.roberta_models = roberta_models
        self.genai_models = genai_models
        self.cache_root = Path(cache_root)
        self.maps_dir = self.cache_root / "maps"
        self.meta_path = self.cache_root / "meta" / "maps_metadata.json"

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def build_all(self, force_rebuild: bool = False) -> None:
        """
        Build or refresh token‑mapping tables for all configured model pairs.

        The method iterates over every (RoBERTa, generator) model pair,
        loads their tokenizers, computes hashes of their vocabularies, and
        decides whether a new mapping needs to be generated based on:
        * ``force_rebuild`` flag,
        * absence of an existing map file,
        * mismatch between stored hashes and current tokenizer hashes.

        New mappings are written to ``self.maps_dir`` and the accompanying
        metadata file is updated accordingly.

        Parameters
        ----------
        force_rebuild : bool, optional
            If ``True``, all maps are regenerated regardless of existing
            metadata.  Defaults to ``False``.
        """
        metadata = self._load_metadata(self.meta_path)

        for roberta_path, roberta_name in self.roberta_models:
            qa_tokenizer = AutoTokenizer.from_pretrained(roberta_path, use_fast=True)
            qa_hash = self._hash_tokenizer(qa_tokenizer)

            for gen_path, gen_name in self.genai_models:
                gen_tokenizer = AutoTokenizer.from_pretrained(
                    gen_path, use_fast=True
                )
                gen_hash = self._hash_tokenizer(gen_tokenizer)

                meta_key = f"{roberta_name}__{gen_name}"
                existing = metadata.get(meta_key)

                need_rebuild = force_rebuild
                if not need_rebuild and existing:
                    need_rebuild = (
                        existing["qa_hash"] != qa_hash
                        or existing["gen_hash"] != gen_hash
                    )

                out_file = self.maps_dir / f"{meta_key}.json"

                if need_rebuild or not out_file.is_file():
                    print(f"🔨 Building map: {meta_key}")
                    mapping = self._build_single_map(qa_tokenizer, gen_tokenizer)
                    self._save_map(mapping, out_file)

                    metadata[meta_key] = {"qa_hash": qa_hash, "gen_hash": gen_hash}
                    self._save_metadata(self.meta_path, metadata)
                else:
                    print(f"✅ Map exists and is up‑to‑date: {meta_key}")

        print("\n✅ All maps are ready!")

    # ------------------------------------------------------------------
    # Helper methods – unchanged from the original script (only renamed)
    # ------------------------------------------------------------------
    @staticmethod
    def _hash_tokenizer(tokenizer: AutoTokenizer) -> str:
        """
        Compute a stable SHA‑256 hash of a tokenizer's vocabulary.

        The hash is derived from a JSON‑encoded representation of the
        tokenizer's ``vocab`` dictionary with keys sorted, ensuring that
        the same vocabulary always yields the same hash regardless of
        dictionary ordering.  This hash is used to detect when a cached
        token‑mapping table is out‑of‑date.

        Parameters
        ----------
        tokenizer : AutoTokenizer
            The tokenizer whose vocabulary should be hashed.

        Returns
        -------
        str
            Hexadecimal SHA‑256 digest of the vocabulary.
        """
        vocab_bytes = json.dumps(tokenizer.get_vocab(), sort_keys=True).encode()
        return hashlib.sha256(vocab_bytes).hexdigest()

    @staticmethod
    def _qa_token_to_word(token: str) -> str:
        """
        Convert a RoBERTa‑style token into its plain word form.

        RoBERTa tokenization prefixes tokens that begin a new word
        with the special character ``Ġ`` and uses the ``##`` prefix for
        sub‑word continuation.  This helper strips those prefixes so
        that the resulting word can be encoded by another tokenizer.

        Parameters
        ----------
        token : str
            The token string produced by a RoBERTa tokenizer.

        Returns
        -------
        str
            The underlying word without RoBERTa‑specific prefixes.
        """
        if token.startswith("Ġ"):
            return token[1:]
        if token.startswith("##"):
            return token[2:]
        return token

    @staticmethod
    def _build_single_map(
        qa_tokenizer: AutoTokenizer,
        gen_tokenizer: AutoTokenizer,
    ) -> Dict[int, List[int]]:
        """
        Build a token‑mapping dictionary for a single QA‑generator pair.

        The function iterates over every token ID in the QA tokenizer,
        converts the token to its underlying word form (handling RoBERTa
        prefixes such as ``Ġ`` or ``##``), and then attempts to encode that
        word with the generator tokenizer.

        Parameters
        ----------
        qa_tokenizer : AutoTokenizer
            The tokenizer used for the QA (RoBERTa‑style) model.
        gen_tokenizer : AutoTokenizer
            The tokenizer used for the generative‑AI model.

        Returns
        -------
        Dict[int, List[int]]
            A mapping where each key is a QA token ID and the value is a
            list of generator token IDs that represent the same word.  Tokens
            that cannot be represented by the generator are omitted.
        """
        mapping: Dict[int, List[int]] = {}
        for qa_id in tqdm(
            range(len(qa_tokenizer)), desc="building map", leave=False
        ):
            token_str = qa_tokenizer.convert_ids_to_tokens(qa_id)
            word = TokenMapBuilder._qa_token_to_word(token_str)

            # Skip words that the generator cannot represent
            gen_ids = gen_tokenizer.encode(word, add_special_tokens=False)
            if gen_ids:
                mapping[qa_id] = gen_ids
        return mapping

    @staticmethod
    def _save_map(mapping: Dict[int, List[int]], out_path: Path) -> None:
        """
        Serialize a token‑mapping dictionary to a JSON file.

        The mapping keys (token IDs) are converted to strings because
        JSON object keys must be strings.  The resulting file is written
        with UTF‑8 encoding and human‑readable indentation.

        Parameters
        ----------
        mapping : Dict[int, List[int]]
            Mapping from QA token IDs to lists of generator token IDs.
        out_path : Path
            Destination file path where the JSON representation will be
            saved.  Parent directories are created if they do not exist.
        """
        json_ready = {str(k): v for k, v in mapping.items()}
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(json_ready, ensure_ascii=False, indent=2))

    @staticmethod
    def _load_metadata(meta_path: Path) -> Dict[str, Dict[str, str]]:
        """
        Load the mapping metadata from ``maps_metadata.json``.

        If the file exists, its JSON content is parsed and returned.
        Otherwise an empty dictionary is returned, allowing the caller
        to treat missing metadata as “no maps have been generated yet”.

        Parameters
        ----------
        meta_path : Path
            Path to the metadata JSON file.

        Returns
        -------
        Dict[str, Dict[str, str]]
            A dictionary mapping ``<roberta_name>__<gen_name>`` keys to a
            sub‑dictionary containing ``qa_hash`` and ``gen_hash`` entries.
        """
        if meta_path.is_file():
            return json.loads(meta_path.read_text())
        return {}

    @staticmethod
    def _save_metadata(meta_path: Path, data: Dict) -> None:
        """
        Persist mapping‑metadata to disk.

        The metadata dictionary stores hashes of the QA and generator
        tokenizers for each map, enabling the builder to detect when a
        cached map is stale.  The function ensures that the target
        directory exists before writing a pretty‑printed JSON file.

        Parameters
        ----------
        meta_path : Path
            Full path to the ``maps_metadata.json`` file.
        data : Dict
            The metadata dictionary to be saved.
        """
        meta_path.parent.mkdir(parents=True, exist_ok=True)
        meta_path.write_text(json.dumps(data, indent=2, sort_keys=True))
