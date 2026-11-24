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



def test_cli_ask(overfit_run, tmp_path, capsys):
    d, examples, _ = overfit_run
    ex = examples[0]
    probe = next(p for p in ex["probes"] if p["input"] != p["output"])
    (tmp_path / "in.txt").write_text(probe["input"])
    (tmp_path / "exp.txt").write_text(probe["output"])
    model = str(d / "run" / "best")
    code = main(["ask", str(tmp_path / "in.txt"), ex["prompt"], "--model", model,
                 "--device", "cpu", "-e", str(tmp_path / "exp.txt"),
                 "-o", str(tmp_path / "out.txt"), "-p", str(tmp_path / "out.tft")])
    assert code == 0
    assert (tmp_path / "out.txt").read_text() == probe["output"]
    assert "## pipeline" in capsys.readouterr().out


def test_cli_ask_explains_when_nothing_is_generated(overfit_run, tmp_path, capsys, monkeypatch):
    from textforge_ai.infer import Forge

    d, _, _ = overfit_run
    (tmp_path / "in.txt").write_text("abc")
    monkeypatch.setattr(Forge, "generate", lambda self, prompt, n=4: [])
    code = main(["ask", str(tmp_path / "in.txt"), "faça algo", "--model",
                 str(d / "run" / "best"), "--device", "cpu"])
    assert code == 2
    assert "English" in capsys.readouterr().err


def test_ask_has_default_model(monkeypatch):
    from textforge_ai.cli import DEFAULT_MODEL_DIR, V2_MODEL_DIR, build_parser

    monkeypatch.delenv("TFAI_MODEL", raising=False)
    args = build_parser().parse_args(["ask", "x.txt"])
    assert args.model in (str(DEFAULT_MODEL_DIR), str(V2_MODEL_DIR))
    assert args.prompt is None

def _params(path):
    from transformers import AutoModelForSeq2SeqLM

    return {k: v.clone() for k, v in AutoModelForSeq2SeqLM.from_pretrained(path).state_dict().items()}


def test_resume_continues_exactly(tmp_path):
    """Stopping mid-epoch and resuming must give the same weights as never stopping."""
    examples = dataset.generate(40, seed=11)
    dataset.write_jsonl(tmp_path / "train.jsonl", examples)
    dataset.write_jsonl(tmp_path / "val.jsonl", examples[:4])
    common = {"data": str(tmp_path), "tiny": True, "epochs": 2, "batch_size": 8, "eval_examples": 4,
                  "log_every": 10_000, "save_every": 3, "no_verify": True}

    straight = train(TrainConfig(out=str(tmp_path / "a"), **common))

    # stop in the middle of epoch 1 (40 examples / 8 = 5 steps per epoch)
    first = train(TrainConfig(out=str(tmp_path / "b"), max_steps=2, **common))
    assert first["stopped"] and first["step"] == 2
    assert (tmp_path / "b" / "checkpoint" / "trainer_state.pt").exists()
    # resume; run settings come from the checkpoint even if passed differently
    resumed = train(TrainConfig(out=str(tmp_path / "b"), resume=True,
                                **{**common, "epochs": 99}))
    assert not resumed["stopped"] and resumed["step"] == straight["step"] == 10

    a, b = _params(tmp_path / "a" / "last"), _params(tmp_path / "b" / "last")
    assert all(torch.allclose(a[k], b[k], atol=1e-6) for k in a)
    assert [h["step"] for h in resumed["history"]] == [h["step"] for h in straight["history"]]


def test_resume_after_epoch_end_skips_finished_eval(tmp_path):
    examples = dataset.generate(16, seed=12)
    dataset.write_jsonl(tmp_path / "train.jsonl", examples)
    dataset.write_jsonl(tmp_path / "val.jsonl", examples[:4])
    common = {"data": str(tmp_path), "out": str(tmp_path / "r"), "tiny": True, "epochs": 2,
                  "batch_size": 8, "eval_examples": 4, "log_every": 10_000, "no_verify": True}
    train(TrainConfig(max_steps=2, **common))  # exactly one full epoch, then stop
    # max_steps hits on the epoch's last batch: the eval of epoch 1 has not run yet
    result = train(TrainConfig(resume=True, **common))
    assert [h["epoch"] for h in result["history"]] == [1, 2]


def test_resume_without_checkpoint_fails_clearly(tmp_path):
    with pytest.raises(FileNotFoundError):
        train(TrainConfig(out=str(tmp_path / "none"), resume=True, tiny=True))


def test_cli_resume_flags(tmp_path):
    examples = dataset.generate(16, seed=13)
    dataset.write_jsonl(tmp_path / "train.jsonl", examples)
    dataset.write_jsonl(tmp_path / "val.jsonl", examples[:4])
    base = ["train", "--tiny", "--data", str(tmp_path), "--out", str(tmp_path / "c"),
            "--epochs", "1", "--batch-size", "8", "--eval-examples", "4", "--no-verify"]
    assert main([*base, "--max-steps", "1"]) == 0
    assert main(["train", "--resume", "--out", str(tmp_path / "c")]) == 0
    info = json.loads((tmp_path / "c" / "last" / "train_info.json").read_text())
    assert info["history"][-1]["step"] == 2
