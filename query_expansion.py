"""Task 7 — lightweight query expansion (synonyms / acronyms)."""

import re
from typing import Dict

_EXPANSIONS: Dict[str, str] = {
    "sgd": "stochastic gradient descent optimization",
    "cnn": "convolutional neural network",
    "rnn": "recurrent neural network",
    "lstm": "long short-term memory",
    "gan": "generative adversarial network",
    "relu": "rectified linear unit",
    "mlp": "multilayer perceptron feedforward",
    "bp": "backpropagation",
}


def expand_query(question: str) -> str:
    """
    Append clarifying phrases for common DL acronyms / short forms.
    Keeps original wording so vector search still matches paraphrases.
    """
    q = question.strip()
    if not q:
        return q

    lowered = q.lower()
    extras: list[str] = []
    for key, phrase in _EXPANSIONS.items():
        if re.search(rf"\b{re.escape(key)}\b", lowered):
            extras.append(phrase)

    if not extras:
        return q

    return f"{q} {' '.join(extras)}"
