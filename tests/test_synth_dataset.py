from textforge_ai import dataset
from textforge_ai.engine import run_pipeline_text
from textforge_ai.synth import TERMINAL


def test_generation_is_deterministic():
    assert dataset.generate(20, seed=7) == dataset.generate(20, seed=7)


def test_every_example_reproduces_its_probes(small_dataset):
    for ex in small_dataset:
        results = run_pipeline_text(ex["pipeline"], [p["input"] for p in ex["probes"]])
        assert [r.output for r in results] == [p["output"] for p in ex["probes"]]


def test_examples_are_never_identity(small_dataset):
    for ex in small_dataset:
        assert any(p["input"] != p["output"] for p in ex["probes"])


def test_targets_are_pipelines_not_text(small_dataset):
    # the core guarantee: a target is always a short program, never rewritten text
    for ex in small_dataset:
        assert len(ex["pipeline"].splitlines()) == len(ex["steps"])
        assert all(line.endswith(";") for line in ex["pipeline"].splitlines())


def test_terminal_steps_only_last(small_dataset):
    for ex in small_dataset:
        assert not any(s in TERMINAL for s in ex["steps"][:-1])


def test_split_never_shares_pipelines(small_dataset):
    splits = dataset.split_by_pipeline(small_dataset, seed=1)
    owners = [{e["pipeline"] for e in rows} for rows in splits.values()]
    assert sum(len(o) for o in owners) == len(set().union(*owners))
    assert sum(len(r) for r in splits.values()) == len(small_dataset)


def test_jsonl_roundtrip(tmp_path, small_dataset):
    path = tmp_path / "x.jsonl"
    dataset.write_jsonl(path, small_dataset)
    assert list(dataset.read_jsonl(path)) == small_dataset
