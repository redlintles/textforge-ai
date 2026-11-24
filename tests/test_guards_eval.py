from textforge_ai import guards
from textforge_ai.baselines import BASELINES
from textforge_ai.evaluate import evaluate, evaluate_predictor, score_one


def test_oracle_scores_perfectly(small_dataset):
    summary, _ = evaluate_predictor(small_dataset, BASELINES["oracle"])
    assert summary["probe_accuracy"] == 1.0
    assert summary["exact_match"] == 1.0
    assert summary["suspicious_rate"] == 0.0


def test_identity_scores_zero(small_dataset):
    summary, _ = evaluate_predictor(small_dataset, BASELINES["identity"])
    assert summary["probe_accuracy"] == 0.0


def test_cheater_is_caught(small_dataset):
    # "erase the input and write the answer" must never count as correct
    summary, _ = evaluate_predictor(small_dataset, BASELINES["cheater"])
    assert summary["probe_accuracy"] == 0.0
    assert summary["suspicious_rate"] > 0.95


def test_cheater_with_perfect_single_probe_still_fails(small_dataset):
    ex = dict(small_dataset[0], probes=small_dataset[0]["probes"][:1])
    score = score_one(ex, BASELINES["cheater"](ex))
    assert score.probe_hits == 1  # it IS right on the text it copied ...
    assert score.suspicious and not score.correct  # ... and still rejected


def test_equivalent_but_different_pipeline_counts_as_correct():
    ex = {"id": "x", "prompt": "uppercase it", "pipeline": "tua;",
          "probes": [{"input": "ab c", "output": "AB C"}, {"input": "x", "output": "X"}]}
    score = score_one(ex, "tla;\ntua;")
    assert score.correct and not score.exact


def test_invalid_prediction_is_scored_not_crashing(small_dataset):
    summary, records = evaluate(small_dataset[:5], {e["id"]: "garbage ;;" for e in small_dataset})
    assert summary["valid_rate"] == 0.0 and records[0]["error"]["stage"] == "persistence"


def test_guards_individually():
    assert guards.is_input_independent("slt 0 0;\ndlf;\nate 'fixed answer';")
    assert not guards.is_input_independent("tua;")
    leak = guards.leaked_literals("ate 'the quick brown fox';", ["THE the quick brown fox"], "add it")
    assert leak == ["the quick brown fox"]
    assert guards.leaked_literals("ate 'the quick brown fox';", ["the quick brown fox"],
                                  'add "the quick brown fox"') == []


def test_filter_then_constant_is_not_a_false_alarm():
    # klm empties texts without the word; on fitting texts the output varies
    pipeline = "klm tiger;\npln '--';"
    assert guards.is_input_independent(pipeline)  # unrelated texts: looks constant
    assert not guards.is_input_independent(pipeline, ["a tiger\nx", "tiger b\ny"])
