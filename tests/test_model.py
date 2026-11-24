"""Training/inference tests with the offline tiny model. Skipped without torch."""

import json

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("transformers")

from textforge_ai import dataset
from textforge_ai.cli import main
from textforge_ai.infer import Forge
from textforge_ai.train import TrainConfig, train


@pytest.fixture(scope="module")
def overfit_run(tmp_path_factory):
    """Train the tiny model on 8 examples until it memorises them (val == train)."""
    d = tmp_path_factory.mktemp("overfit")
    examples = dataset.generate(8, seed=9)
    dataset.write_jsonl(d / "train.jsonl", examples)
    dataset.write_jsonl(d / "val.jsonl", examples)
    result = train(TrainConfig(data=str(d), out=str(d / "run"), tiny=True, epochs=40,
                               batch_size=8, lr=3e-3, warmup=0.02, log_every=10_000,
                               eval_examples=8, no_verify=True))
    return d, examples, result


def test_training_loop_can_memorise(overfit_run):
    # classic sanity check: if a model cannot overfit 8 examples, the loop is broken
    _, _, result = overfit_run
    assert result["best_gen_probe_accuracy"] == 1.0
    assert result["history"][0]["gen_probe_accuracy"] < 1.0


def test_checkpoints_and_info_saved(overfit_run):
    d, _, _ = overfit_run
    for name in ("best", "last"):
        info = json.loads((d / "run" / name / "train_info.json").read_text())
        assert info["config"]["tiny"] is True and info["history"]


def test_reloaded_model_generates_and_solves(overfit_run):
    d, examples, _ = overfit_run
    forge = Forge.from_dir(str(d / "run" / "best"), device="cpu")
    ex = examples[0]
    assert ex["pipeline"] in forge.generate(ex["prompt"], n=4)
    probe = next(p for p in ex["probes"] if p["input"] != p["output"])
    result = forge.solve(ex["prompt"], probe["input"], expected=probe["output"])
    assert result.best is not None and result.best.output == probe["output"]
    assert result.best.verdict.method == "expected-output"


def test_model_verdict_is_a_probability(overfit_run):
    d, examples, _ = overfit_run
    forge = Forge.from_dir(str(d / "run" / "best"), device="cpu")
    ex, probe = examples[0], examples[0]["probes"][0]
    verdict = forge.verify(ex["prompt"], ex["pipeline"], probe["input"], probe["output"])
    assert verdict.label in ("match", "mismatch") and 0.5 <= verdict.confidence <= 1.0
    assert verdict.method == "model"


def test_multitask_training_runs(tmp_path):
    examples = dataset.generate(30, seed=4)
    dataset.write_jsonl(tmp_path / "train.jsonl", examples)
    dataset.write_jsonl(tmp_path / "val.jsonl", examples[:5])
    result = train(TrainConfig(data=str(tmp_path), out=str(tmp_path / "r"), tiny=True,
                               epochs=1, batch_size=16, eval_examples=5, log_every=10_000))
    assert 0.0 <= result["history"][0]["verify_accuracy"] <= 1.0


def test_cli_solve_and_verify(overfit_run, tmp_path, capsys):
    d, examples, _ = overfit_run
    ex = examples[0]
    probe = next(p for p in ex["probes"] if p["input"] != p["output"])
    (tmp_path / "in.txt").write_text(probe["input"])
    (tmp_path / "exp.txt").write_text(probe["output"])
    (tmp_path / "prompt.txt").write_text(ex["prompt"])
    model = str(d / "run" / "best")
    args = ["--model", model, "--input", str(tmp_path / "in.txt"),
            "--prompt-file", str(tmp_path / "prompt.txt"), "--device", "cpu"]
    assert main(["solve", *args, "--expected", str(tmp_path / "exp.txt"),
                 "--save-pipeline", str(tmp_path / "out.tft")]) == 0
    assert (tmp_path / "out.tft").exists()
    capsys.readouterr()
    assert main(["verify", *args, "--pipeline", str(tmp_path / "out.tft"),
                 "--expected", str(tmp_path / "exp.txt")]) == 0
    assert json.loads(capsys.readouterr().out)["label"] == "match"
