from typing import List, Set
from collections.abc import Callable
import re
import json


STOPWORDS_FILE = "data/stopwords_english.json"


def bound_target(value: float) -> float:
    return min(max(value, 0.0), 1.0)

def remove_latex(text: str) -> str:
    # Inline math expressions $...$
    result = re.sub(r"\$.*?\$", " ", text)
    # Display math expressions \[...\] or $$...$$
    result = re.sub(r"\\\[.*?\\\]", " ", result, flags=re.DOTALL)
    # \begin{...} and \end{...} and everything between them
    result = re.sub(r"\\begin\{.*?\}.*?\\end\{.*?\}", " ", result, flags=re.DOTALL)

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