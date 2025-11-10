import re
import unicodedata
from typing import Iterable, List

_WS_RE = re.compile(r"\s+")


def normalize_description(text: str) -> str:
    """
    Lightweight normalization for descriptions:
    - Unicode NFKC
    - Collapse whitespace
    - Strip
    Note: We do NOT lowercase to preserve cased models like mBERT.
    """
    if text is None:
        return ""
    norm = unicodedata.normalize("NFKC", text)
    norm = _WS_RE.sub(" ", norm)
    return norm.strip()


def normalize_batch(texts: Iterable[str]) -> List[str]:
    return [normalize_description(t) for t in texts]

