"""Training data for the *verifier* task: does this pipeline do what was asked?

Negatives are **mutations** of the reference pipeline (wrong argument, neighbour
instruction, dropped/added step, swapped order). The label comes from **execution**:
a mutation that yields the same output on every probe is equivalent, so it is a
``match``. The verifier therefore learns to judge results, not to memorise syntax.

For a mismatch, the probe shown to the verifier is one where the outputs actually
differ; otherwise we would ask it to spot an error that is invisible in its input.
"""

from __future__ import annotations

import random
from dataclasses import dataclass

import textforge as tf

from . import catalog as C
from .engine import LINE_SEP, parse_pipeline, run_pipeline_text
from .synth import sample_value

MATCH = "match"
MISMATCH = "mismatch"

#: Instructions that are easy to confuse with each other (same argument shape).
NEIGHBOURS: tuple[tuple[str, ...], ...] = (
    ("raw", "rfw", "rlw"),
    ("tua", "tla"),
    ("klm", "dlm"),
    ("tbs", "tls", "trs"),
    ("cfw", "clw"),
    ("tucw", "tlcw", "cts"),
    ("atb", "ate"),
    ("apl", "pln"),
    ("dlf", "dll"),
    ("rtl", "rtr"),
    ("padl", "padr"),
    ("jkbc", "jsnc", "jcmc", "jpsc"),
)
_NEIGHBOUR_OF = {m: group for group in NEIGHBOURS for m in group}


@dataclass(frozen=True)
class VerifyExample:
    prompt: str
    pipeline: str
    input: str
    output: str
    label: str  # "match" or "mismatch"
    target: str  # what the model must write, e.g. "mismatch: step 2"
    mutation: str  # "none" for the reference pipeline


def instructions_of(pipeline_text: str) -> list[tf.Instruction]:
    parsed = parse_pipeline(pipeline_text)
    if not parsed.ok:
        raise ValueError(f"invalid pipeline: {parsed.error}")
    return list(parsed.processor.get_transform_vec(parsed.pipeline_id))


def text_of(instructions: list[tf.Instruction]) -> str:
    processor = tf.Processor()
    pipeline_id = processor.add_transform(instructions)
    return LINE_SEP.join(line.strip() for line in processor.get_text_transform_vec(pipeline_id))


def _changed_argument(instruction: tf.Instruction, rng: random.Random) -> tf.Instruction | None:
    spec = C.BY_MNEMONIC.get(instruction.name)
    args = list(instruction.arguments)
    if spec is None or not args or len(spec.args) != len(args):
        return None
    index = rng.randrange(len(args))
    old = args[index]
    for _ in range(10):
        new = sample_value(spec.args[index], rng)
        if isinstance(old, int) and not isinstance(new, int):
            new = int(new) if str(new).isdigit() else old
        if str(new) != str(old):
            args[index] = new
            return tf.Instruction(instruction.name, *args)
    return None


def mutate(
    instructions: list[tf.Instruction], rng: random.Random
) -> tuple[list[tf.Instruction], str, int | None] | None:
    """Return (mutated, kind, step index or None for 'missing step')."""
    kinds = ["argument", "neighbour", "add"]
    if len(instructions) > 1:
        kinds += ["drop", "swap"]
    kind = rng.choice(kinds)
    steps = list(instructions)
    i = rng.randrange(len(steps))
    if kind == "argument":
        changed = _changed_argument(steps[i], rng)
        if changed is None:
            return None
        steps[i] = changed
        return steps, kind, i
    if kind == "neighbour":
        group = _NEIGHBOUR_OF.get(steps[i].name)
        if not group:
            return None
        other = rng.choice([m for m in group if m != steps[i].name])
        steps[i] = tf.Instruction(other, *steps[i].arguments)
        return steps, kind, i
    if kind == "add":
        spec = rng.choice(C.CATALOG)
        new = tf.Instruction(spec.mnemonic, *(sample_value(k, rng) for k in spec.args))
        position = rng.randrange(len(steps) + 1)
        steps.insert(position, new)
        return steps, kind, position
    if kind == "drop":
        del steps[i]
        return steps, kind, None
    # swap two adjacent steps
    j = i + 1 if i + 1 < len(steps) else i - 1
    steps[i], steps[j] = steps[j], steps[i]
    return steps, kind, min(i, j)


def mismatch_target(step: int | None, instructions: list[tf.Instruction]) -> str:
    if step is None:
        return f"{MISMATCH}: missing step"
    return f"{MISMATCH}: step {step + 1} ({instructions[step].name})"


def examples_for(example: dict, rng: random.Random, negatives: int = 2) -> list[VerifyExample]:
    """One positive (the reference) plus up to ``negatives`` execution-labelled mutations."""
    probes = example["probes"]
    inputs = [p["input"] for p in probes]
    expected = [p["output"] for p in probes]
    changed = [k for k, p in enumerate(probes) if p["input"] != p["output"]]
    shown = rng.choice(changed or range(len(probes)))
    out = [
        VerifyExample(example["prompt"], example["pipeline"], inputs[shown], expected[shown],
                      MATCH, MATCH, "none")
    ]
    reference = instructions_of(example["pipeline"])
    tries = 0
    while len(out) < negatives + 1 and tries < negatives * 6:
        tries += 1
        mutation = mutate(reference, rng)
        if mutation is None:
            continue
        steps, kind, index = mutation
        if not steps:
            continue
        try:
            candidate = text_of(steps)
        except Exception:  # noqa: BLE001, S112 - invalid mutation, just skip it
            continue
        results = run_pipeline_text(candidate, inputs)
        if not all(r.ok for r in results):
            continue  # runtime errors are caught by the engine, not the verifier
        outputs = [r.output for r in results]
        if outputs == expected:
            label, target = MATCH, MATCH  # equivalent pipeline: a correct answer
            k = shown
        else:
            label, target = MISMATCH, mismatch_target(index, steps)
            k = rng.choice([n for n in range(len(probes)) if outputs[n] != expected[n]])
        out.append(VerifyExample(example["prompt"], candidate, inputs[k], outputs[k],
                                 label, target, kind))
    return out
