"""Detectors for the "AI just rewrote the text" failure mode.

A cheating pipeline erases the input and writes the expected output as a literal,
e.g. ``slt 0 0; dlf; ate 'THE WHOLE ANSWER'``. Two cheap signals catch it:

* **input independence** - a real transformation reacts to its input. If several
  different realistic texts all produce the same non-empty output, the pipeline
  ignores its input. (Use texts that fit the request: a correct ``klm tiger`` empties
  any text without "tiger", so unrelated texts would give false alarms.)
* **literal leak** - a long string literal in the pipeline that was copied from the
  input text or from the expected output (and not from the user's prompt) is text,
  not a transformation.
"""

from __future__ import annotations

from dataclasses import dataclass

from .engine import parse_pipeline, run_pipeline_text

FOREIGN_TEXTS = (
    "Zebra 42 quantum\n  xylophone   jam\n",
    "completely different input; nothing shared at all, 7 8 9",
)
LEAK_MIN_LEN = 12


@dataclass(frozen=True)
class GuardReport:
    input_independent: bool
    leaked_literals: tuple[str, ...]

    @property
    def suspicious(self) -> bool:
        return self.input_independent or bool(self.leaked_literals)


def string_literals(pipeline_text: str) -> list[str]:
    parsed = parse_pipeline(pipeline_text)
    if not parsed.ok:
        return []
    literals = []
    for instruction in parsed.processor.get_transform_vec(parsed.pipeline_id):
        literals += [a for a in instruction.arguments if isinstance(a, str)]
    return literals


def is_input_independent(pipeline_text: str, inputs: list[str] | None = None) -> bool:
    inputs = list(dict.fromkeys(inputs)) if inputs else list(FOREIGN_TEXTS)
    if len(inputs) < 2:
        return False
    results = run_pipeline_text(pipeline_text, inputs)
    if not all(r.ok for r in results):
        return False
    outputs = {r.output for r in results}
    return len(outputs) == 1 and outputs != {""}


def leaked_literals(pipeline_text: str, texts: list[str], prompt: str) -> list[str]:
    """Literals >= LEAK_MIN_LEN chars found in any of ``texts`` but not in ``prompt``."""
    return [
        lit
        for lit in string_literals(pipeline_text)
        if len(lit) >= LEAK_MIN_LEN and lit not in prompt and any(lit in t for t in texts)
    ]


def check(
    pipeline_text: str, inputs: list[str], references: list[str], prompt: str
) -> GuardReport:
    """``inputs``: probe texts; ``references``: expected outputs (must not be copied)."""
    return GuardReport(
        input_independent=is_input_independent(pipeline_text, inputs),
        leaked_literals=tuple(leaked_literals(pipeline_text, inputs + references, prompt)),
    )
