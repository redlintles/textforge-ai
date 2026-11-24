"""Synthetic training examples: (English prompt) -> (TextForge pipeline).

Recipe for one example:

1. sample 1..N instruction steps from the catalog, with arguments;
2. build the pipeline with the *official builder* and serialize it with the engine;
3. generate several probe texts and run the pipeline step by step on them;
4. reject the example if any step errors or does nothing on every probe
   (a prompt describing a no-op step teaches the model to hallucinate steps);
5. describe the steps in English using the catalog templates.

Every target is a pipeline. The model is never shown a "rewritten text" as a
target, so imitation learning gives it no reason to output text instead of code.
"""

from __future__ import annotations

import random
from dataclasses import asdict, dataclass

import textforge as tf

from . import catalog as C
from . import lexicon, phrasing
from .engine import canonical_text, run_builder
from .texts import VOCAB, make_text

#: share of prompts written with the original templates (the rest use phrasing.py)
CLASSIC_SHARE = 0.25
#: share of literal values taken from the original small pools (the rest: lexicon.py)
CLASSIC_VALUES = 0.4

#: Instructions that reshape the text so much that line/word steps after them are
#: odd or meaningless. They are only placed as the last step.
TERMINAL = frozenset(
    {"jkbc", "jsnc", "jcmc", "jpsc", "emj", "joinln", "rmws", "urle", "htmle", "jsone", "rpt"}
)

_TEXT_LITERALS = ("!", ".", " END", "--", ">> ", "[x] ", "NOTE: ", " (done)", "***", "# ")
_SEPARATORS = (", ", " | ", " ", "; ", " / ", " - ")
_MORE_SEPARATORS = (",", "|", " -> ", " + ", " :: ", " & ", "; ", " ", ", ")
_PAD_CHARS = ("*", "-", "=", ".", "#")
_CONTENT_WORDS = tuple(w for w in VOCAB if len(w) > 3)


@dataclass(frozen=True)
class Step:
    mnemonic: str
    values: tuple[int | str, ...]


@dataclass
class Example:
    id: str
    prompt: str
    pipeline: str
    steps: list[str]
    probes: list[dict]  # [{"input": ..., "output": ...}]

    def to_dict(self) -> dict:
        return asdict(self)


def sample_value(kind: str, rng: random.Random) -> int | str:
    classic = rng.random() < CLASSIC_VALUES
    if kind == C.WORD:
        return rng.choice(_CONTENT_WORDS) if classic else lexicon.pattern(rng)
    if kind == C.REPL:
        if not classic:
            return lexicon.replacement(rng)
        return rng.choice(_CONTENT_WORDS).upper() if rng.random() < 0.3 else rng.choice(
            _CONTENT_WORDS
        )
    if kind == C.TEXT:
        return rng.choice(_TEXT_LITERALS) if classic else lexicon.literal(rng)
    if kind == C.SEP:
        return rng.choice(_SEPARATORS if classic else _MORE_SEPARATORS)
    if kind == C.PAD_CHAR:
        return rng.choice(_PAD_CHARS)
    if kind == C.CHAR_IDX:
        return rng.randint(0, 6)
    if kind in (C.WORD_IDX, C.NTH):
        return rng.randint(0, 2)
    if kind == C.LINE_IDX:
        return rng.randint(0, 1)
    if kind == C.COUNT:
        return rng.randint(2, 3)
    if kind == C.CUT:
        return rng.randint(1, 5)
    if kind == C.PAD_LEN:
        return rng.choice((60, 80, 100, 120))
    raise ValueError(kind)


def sample_steps(rng: random.Random, max_steps: int = 4) -> list[Step]:
    count = rng.choices(range(1, max_steps + 1), weights=[4, 4, 3, 2][:max_steps])[0]
    specs = list(C.CATALOG)
    weights = [s.weight for s in specs]
    steps: list[Step] = []
    while len(steps) < count:
        spec = rng.choices(specs, weights=weights)[0]
        if steps and steps[-1].mnemonic == spec.mnemonic:
            continue
        if spec.mnemonic in TERMINAL and len(steps) != count - 1:
            continue
        steps.append(Step(spec.mnemonic, tuple(sample_value(k, rng) for k in spec.args)))
    return steps


def build_pipeline(steps: list[Step]) -> tf.Pipeline:
    pipeline = tf.Pipeline()
    for step in steps:
        spec = C.BY_MNEMONIC[step.mnemonic]
        pipeline = getattr(pipeline, spec.method)(*step.values)
    return pipeline


def describe_step(step: Step, rng: random.Random) -> str:
    spec = C.BY_MNEMONIC[step.mnemonic]
    style = rng.randrange(2)
    rendered = [C.render_arg(k, v, style=style) for k, v in zip(spec.args, step.values)]
    return rng.choice(spec.templates).format(*rendered)


def _cap(text: str) -> str:
    return text[:1].upper() + text[1:]


def describe(steps: list[Step], rng: random.Random) -> str:
    if rng.random() >= CLASSIC_SHARE:
        return phrasing.describe(steps, rng)
    return describe_classic(steps, rng)


def describe_classic(steps: list[Step], rng: random.Random) -> str:
    """The original phrasing: catalog templates, double quotes, 4 ways to chain steps."""
    clauses = [describe_step(s, rng) for s in steps]
    if len(clauses) == 1:
        frame = rng.choice(("{}.", "please {}.", "{}", "I need you to {}."))
        return _cap(frame.format(clauses[0]))
    style = rng.randrange(4)
    if style == 0:
        middle = "".join(f", then {c}" for c in clauses[1:-1])
        return f"First, {clauses[0]}{middle}, and finally {clauses[-1]}."
    if style == 1:
        return " ".join(_cap(c) + "." for c in clauses)
    if style == 2:
        return "\n".join(f"{i}. {_cap(c)}" for i, c in enumerate(clauses, 1))
    return _cap(", then ".join(clauses)) + "."


def _probe_words(steps: list[Step]) -> list[str]:
    words = []
    for step in steps:
        spec = C.BY_MNEMONIC[step.mnemonic]
        words += [v for k, v in zip(spec.args, step.values) if k == C.WORD]
    return words


def try_make_example(rng: random.Random, example_id: str, n_probes: int = 6) -> Example | None:
    """Return a verified example, or None if the sampled pipeline is degenerate."""
    steps = sample_steps(rng)
    required = _probe_words(steps)
    inputs = [make_text(rng, required) for _ in range(n_probes)]

    # Execute step by step: every step must succeed everywhere and matter somewhere.
    current = list(inputs)
    for step in steps:
        results = run_builder(build_pipeline([step]), current)
        if not all(r.ok for r in results):
            return None
        outputs = [r.output for r in results]
        if outputs == current:
            return None
        current = outputs
    if current == inputs or any(out == "" for out in current):
        return None

    return Example(
        id=example_id,
        prompt=describe(steps, rng),
        pipeline=canonical_text(build_pipeline(steps)),
        steps=[s.mnemonic for s in steps],
        probes=[{"input": i, "output": o} for i, o in zip(inputs, current)],
    )
