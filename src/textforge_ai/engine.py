"""Thin wrapper around the ``textforgers`` binding (import name ``textforge``).

Everything the AI produces is *pipeline text* (the ``.tft`` format). This module is
the single place that turns that text into something executable, runs it, and
reports failures in a structured, model-friendly shape.
"""

from __future__ import annotations

import os
import tempfile
from dataclasses import dataclass, field

import textforge as tf

#: Separator between instructions in a canonical pipeline. The TextForge parser
#: requires one instruction per line.
LINE_SEP = "\n"


@dataclass(frozen=True)
class EngineError:
    """Structured view of a ``TextForgeError`` (useful later for self-correction)."""

    stage: str
    kind: str
    message: str
    hint: str | None = None

    @classmethod
    def from_exception(cls, exc: Exception) -> EngineError:
        if isinstance(exc, tf.TextForgeError):
            return cls(stage=exc.stage, kind=exc.kind, message=exc.message, hint=exc.hint)
        return cls(stage="python", kind=type(exc).__name__, message=str(exc))

    def to_dict(self) -> dict:
        return {"stage": self.stage, "kind": self.kind, "message": self.message, "hint": self.hint}


@dataclass
class ParseResult:
    ok: bool
    processor: tf.Processor | None = None
    pipeline_id: str | None = None
    error: EngineError | None = None


@dataclass
class RunResult:
    ok: bool
    output: str | None = None
    error: EngineError | None = None
    extra: dict = field(default_factory=dict)


def parse_pipeline(text: str) -> ParseResult:
    """Parse ``.tft`` pipeline text. Never raises for bad pipelines."""
    fd, path = tempfile.mkstemp(suffix=".tft", text=True)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            handle.write(text if text.endswith("\n") else text + "\n")
        processor = tf.Processor()
        try:
            pipeline_id = processor.get_pipeline_from_file(path)
        except Exception as exc:  # noqa: BLE001 - report every failure as data
            return ParseResult(ok=False, error=EngineError.from_exception(exc))
        return ParseResult(ok=True, processor=processor, pipeline_id=pipeline_id)
    finally:
        os.unlink(path)


def run_pipeline_text(text: str, inputs: list[str]) -> list[RunResult]:
    """Parse once, then run against every input. Parse errors fail every input."""
    parsed = parse_pipeline(text)
    if not parsed.ok:
        return [RunResult(ok=False, error=parsed.error) for _ in inputs]
    results = []
    for item in inputs:
        try:
            out = parsed.processor.process_all(parsed.pipeline_id, item)
            results.append(RunResult(ok=True, output=out))
        except Exception as exc:  # noqa: BLE001
            results.append(RunResult(ok=False, error=EngineError.from_exception(exc)))
    return results


def canonical_text(pipeline: tf.Pipeline) -> str:
    """Serialize a builder ``Pipeline`` exactly as the engine writes ``.tft`` files.

    Using the engine's own serializer means quoting rules are never re-implemented
    (and never drift) on the Python side.
    """
    processor = tf.Processor()
    pipeline_id = processor.register(pipeline)
    lines = [line.strip() for line in processor.get_text_transform_vec(pipeline_id)]
    return LINE_SEP.join(lines)


def canonicalize(text: str) -> str | None:
    """Normalize pipeline text by round-tripping it through the engine (None if invalid)."""
    parsed = parse_pipeline(text)
    if not parsed.ok:
        return None
    lines = parsed.processor.get_text_transform_vec(parsed.pipeline_id)
    return LINE_SEP.join(line.strip() for line in lines)


def run_builder(pipeline: tf.Pipeline, inputs: list[str]) -> list[RunResult]:
    """Run a builder ``Pipeline`` directly (used by the data generator)."""
    compiled = pipeline.compile()
    results = []
    for item in inputs:
        try:
            results.append(RunResult(ok=True, output=compiled(item)))
        except Exception as exc:  # noqa: BLE001
            results.append(RunResult(ok=False, error=EngineError.from_exception(exc)))
    return results
