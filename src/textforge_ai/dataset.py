"""Build, split and persist synthetic datasets as JSONL."""

from __future__ import annotations

import json
import random
from collections.abc import Iterable, Iterator
from pathlib import Path

from .synth import Example, try_make_example


def generate(n: int, seed: int = 0, n_probes: int = 6, max_attempts_factor: int = 20) -> list[dict]:
    """Generate ``n`` verified, de-duplicated examples (deterministic for a seed)."""
    rng = random.Random(seed)
    seen: set[tuple[str, str]] = set()
    examples: list[dict] = []
    attempts = 0
    while len(examples) < n:
        attempts += 1
        if attempts > n * max_attempts_factor:
            raise RuntimeError(f"only {len(examples)}/{n} examples after {attempts} attempts")
        example: Example | None = try_make_example(rng, f"s{seed}-{attempts}", n_probes)
        if example is None:
            continue
        key = (example.prompt, example.pipeline)
        if key in seen:
            continue
        seen.add(key)
        examples.append(example.to_dict())
    return examples


def split_by_pipeline(
    examples: list[dict], fractions: tuple[float, float, float] = (0.9, 0.05, 0.05), seed: int = 0
) -> dict[str, list[dict]]:
    """Split so that a given pipeline appears in only ONE split.

    Otherwise the test set would contain pipelines the model memorised in training
    (only the wording would differ) and the score would overstate generalisation.
    """
    pipelines = sorted({e["pipeline"] for e in examples})
    random.Random(seed).shuffle(pipelines)
    n_train = int(len(pipelines) * fractions[0])
    n_val = int(len(pipelines) * fractions[1])
    owner = {}
    for index, pipeline in enumerate(pipelines):
        owner[pipeline] = "train" if index < n_train else "val" if index < n_train + n_val else "test"
    splits: dict[str, list[dict]] = {"train": [], "val": [], "test": []}
    for example in examples:
        splits[owner[example["pipeline"]]].append(example)
    return splits


def write_jsonl(path: str | Path, rows: Iterable[dict]) -> int:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
            count += 1
    return count


def read_jsonl(path: str | Path) -> Iterator[dict]:
    with Path(path).open(encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                yield json.loads(line)
