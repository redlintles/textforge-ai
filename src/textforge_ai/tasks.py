"""Text formats of the two tasks the single model learns (multitask, by prefix).

    generate: <prompt>                                   -> <pipeline>
    verify: <prompt> || pipeline: ... || input: ... || output: ...  -> match | mismatch: ...

Input/output texts are truncated so a verify example fits the model's context
(512 tokens for CodeT5). The *head* of the text is kept: most edits are visible there.
"""

from __future__ import annotations

import random

from .verify_data import examples_for

GENERATE = "generate: "
VERIFY = "verify: "
MAX_TEXT_CHARS = 400


def _clip(text: str, limit: int = MAX_TEXT_CHARS) -> str:
    return text if len(text) <= limit else text[:limit] + " …"


def generate_source(prompt: str) -> str:
    return GENERATE + prompt


def verify_source(prompt: str, pipeline: str, input_text: str, output_text: str) -> str:
    return (
        f"{VERIFY}{prompt} || pipeline: {pipeline} || "
        f"input: {_clip(input_text)} || output: {_clip(output_text)}"
    )


def generate_pairs(examples: list[dict]) -> list[tuple[str, str]]:
    return [(generate_source(e["prompt"]), e["pipeline"]) for e in examples]


def verify_pairs(examples: list[dict], seed: int = 0, negatives: int = 2) -> list[tuple[str, str]]:
    rng = random.Random(seed)
    pairs = []
    for example in examples:
        for v in examples_for(example, rng, negatives):
            pairs.append((verify_source(v.prompt, v.pipeline, v.input, v.output), v.target))
    return pairs


def training_pairs(examples: list[dict], seed: int = 0, verify: bool = True) -> list[tuple[str, str]]:
    """All (source, target) pairs, shuffled, for one multitask training run."""
    pairs = generate_pairs(examples) + (verify_pairs(examples, seed) if verify else [])
    random.Random(seed).shuffle(pairs)
    return pairs
