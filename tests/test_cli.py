import json

from textforge_ai import dataset
from textforge_ai.cli import main


def test_gen_data_and_baseline(tmp_path, capsys):
    out = tmp_path / "data"
    assert main(["gen-data", "--n", "60", "--seed", "3", "--out", str(out)]) == 0
    assert sum(1 for _ in dataset.read_jsonl(out / "train.jsonl")) > 0
    capsys.readouterr()
    assert main(["baseline", "oracle", "--data", str(out / "train.jsonl")]) == 0
    assert json.loads(capsys.readouterr().out)["probe_accuracy"] == 1.0


def test_eval_predictions_file(tmp_path, capsys, small_dataset):
    data, pred = tmp_path / "d.jsonl", tmp_path / "p.jsonl"
    dataset.write_jsonl(data, small_dataset[:10])
    dataset.write_jsonl(pred, [{"id": e["id"], "pipeline": e["pipeline"]} for e in small_dataset[:10]])
    assert main(["eval", "--data", str(data), "--pred", str(pred)]) == 0
    assert json.loads(capsys.readouterr().out)["exact_match"] == 1.0


def test_run_command(tmp_path, capsys):
    (tmp_path / "p.tft").write_text("tbs;\ntua;\n")
    (tmp_path / "in.txt").write_text("  hi there \n")
    assert main(["run", str(tmp_path / "p.tft"), str(tmp_path / "in.txt")]) == 0
    assert capsys.readouterr().out == "HI THERE"
