"""Score predicted pipelines against a dataset.

The headline metric is **probe accuracy**: a prediction counts only if it produces
the expected output on *every* probe text. Probes are different texts, so a
pipeline that hard-codes one answer fails the others - that is what makes
"replace the text with the answer" a losing strategy, even before the guards.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import asdict, dataclass

from . import guards
from .engine import canonicalize, run_pipeline_text

Predictor = Callable[[dict], str]


@dataclass
class ExampleScore:
    id: str
    valid: bool  # parses with the real TextForge parser
    executes: bool  # runs without error on all probes
    probe_hits: int
    probe_total: int
    exact: bool  # same canonical pipeline as the reference
    suspicious: bool  # caught by guards.py
    error: dict | None = None

    @property
    def correct(self) -> bool:
        return self.executes and self.probe_hits == self.probe_total and not self.suspicious


def score_one(example: dict, predicted: str) -> ExampleScore:
    probes = example["probes"]
    results = run_pipeline_text(predicted, [p["input"] for p in probes])
    first_error = next((r.error.to_dict() for r in results if not r.ok), None)
    valid = not (first_error and first_error["stage"] in ("persistence", "parse", "python"))
    canonical = canonicalize(predicted) if valid else None
    inputs, references = [p["input"] for p in probes], [p["output"] for p in probes]
    report = guards.check(predicted, inputs, references, example["prompt"]) if valid else None
    return ExampleScore(
        id=example["id"],
        valid=valid and canonical is not None,
        executes=all(r.ok for r in results),
        probe_hits=sum(r.ok and r.output == p["output"] for r, p in zip(results, probes)),
        probe_total=len(probes),
        exact=canonical == example["pipeline"],
        suspicious=bool(report and report.suspicious),
        error=first_error,
    )


def summarize(scores: list[ExampleScore]) -> dict:
    n = len(scores) or 1
    return {
        "examples": len(scores),
        "valid_rate": sum(s.valid for s in scores) / n,
        "exec_rate": sum(s.executes for s in scores) / n,
        "probe_accuracy": sum(s.correct for s in scores) / n,
        "exact_match": sum(s.exact for s in scores) / n,
        "suspicious_rate": sum(s.suspicious for s in scores) / n,
    }


def evaluate(examples: list[dict], predictions: dict[str, str]) -> tuple[dict, list[dict]]:
    """``predictions`` maps example id -> predicted pipeline text."""
    scores = [score_one(e, predictions.get(e["id"], "")) for e in examples]
    return summarize(scores), [asdict(s) | {"correct": s.correct} for s in scores]


def evaluate_predictor(examples: list[dict], predictor: Predictor) -> tuple[dict, list[dict]]:
    return evaluate(examples, {e["id"]: predictor(e) for e in examples})
