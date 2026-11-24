import random

from textforge_ai import catalog as C
from textforge_ai.engine import run_pipeline_text
from textforge_ai.tasks import GENERATE, VERIFY, training_pairs, verify_source
from textforge_ai.verify_data import (
    MATCH,
    MISMATCH,
    NEIGHBOURS,
    examples_for,
    instructions_of,
    text_of,
)


def test_neighbours_are_real_and_shape_compatible():
    for group in NEIGHBOURS:
        shapes = {C.BY_MNEMONIC[m].args for m in group}
        assert len(shapes) == 1, group


def test_instruction_roundtrip():
    text = "raw apple pear;\nrcw 'a b' X 2;"
    assert text_of(instructions_of(text)) == text


def test_reference_is_always_a_match(small_dataset):
    rng = random.Random(0)
    for ex in small_dataset[:40]:
        first = examples_for(ex, rng)[0]
        assert first.label == MATCH and first.mutation == "none"
        assert first.pipeline == ex["pipeline"]


def test_labels_agree_with_execution(small_dataset):
    rng = random.Random(1)
    for ex in small_dataset[:60]:
        expected = {p["input"]: p["output"] for p in ex["probes"]}
        for v in examples_for(ex, rng):
            (run,) = run_pipeline_text(v.pipeline, [v.input])
            assert run.ok and run.output == v.output
            if v.label == MISMATCH:
                # the error must be visible in the example the verifier is shown
                assert v.output != expected[v.input]
                assert v.target.startswith("mismatch: ")


def test_both_labels_present(small_dataset):
    rng = random.Random(2)
    labels = {v.label for ex in small_dataset for v in examples_for(ex, rng)}
    assert labels == {MATCH, MISMATCH}


def test_task_prefixes_and_truncation(small_dataset):
    pairs = training_pairs(small_dataset[:20])
    assert {s.split(":")[0] + ": " for s, _ in pairs} == {GENERATE, VERIFY}
    long = verify_source("p", "tua;", "x" * 5000, "y" * 5000)
    assert len(long) < 1000
