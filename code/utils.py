from collections.abc import Callable
import re
import json
import math
from typing import List, Set


STOPWORDS_FILE = "data/stopwords_english.json"


def identity(x: float) -> float:
    return x

def sigmoid(x: float) -> float:
    return 1 / (1 + pow(math.e, -x))

def clip(x: float) -> float:
    return min(max(x, 0.0), 1.0)

def remove_tex_symbols(text: str) -> str:
    # Remove common TeX symbols
    symbols = ["\\", "{", "}", "^", "_", "~", "%", "&", "#", "$"]
    for sym in symbols:
        text = text.replace(sym, " ")
    return text

def remove_tex(text: str) -> str:
    # \begin{...} and \end{...} and everything between them
    result = re.sub(r"\\begin\{.*?\}.*?\\end\{.*?\}", " ", text, flags=re.DOTALL)
    # Display math expressions \[...\] or $$...$$
    result = re.sub(r"\\\[.*?\\\]|\$\$.*?\$\$", " ", result, flags=re.DOTALL)
    # Inline math expressions $...$
    result = re.sub(r"\$.*?\$", " ", result, flags=re.DOTALL)

    return result

def stem_words(words: List[str], stem: Callable[[str], str]) -> None:
    words[:] = [stem(word) for word in words]

def filter_words(words: List[str], filter: Set[str], contains: bool) -> None:
    if contains:
        words[:] = [word for word in words if any(f in word for f in filter)]
    else:
        words[:] = [word for word in words if word not in filter]

def load_stopwords() -> Set[str]:
    with open(STOPWORDS_FILE, "r", encoding="utf-8") as f:
        stopwords = json.load(f)
    return set(stopwords)