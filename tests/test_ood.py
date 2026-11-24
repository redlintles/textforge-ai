from textforge_ai import dataset, ood
from textforge_ai.engine import run_pipeline_text


def _norm(text: str) -> str:
    return " ".join(text.lower().replace("'", '"').split()).rstrip(".!?")


def test_ood_cases_build_and_reproduce():
    rows = ood.build_examples()
    assert len(rows) == len(ood.OOD_CASES) >= 80
    for row in rows:
        results = run_pipeline_text(row["pipeline"], [p["input"] for p in row["probes"]])
        assert [r.output for r in results] == [p["output"] for p in row["probes"]]
        assert any(p["input"] != p["output"] for p in row["probes"])


def test_oracle_scores_100_and_report_lists_failures():
    rows = ood.build_examples()
    perfect = ood.report(rows, [r["pipeline"] for r in rows])
    assert perfect["probe_accuracy"] == 1.0 and not perfect["failures"]
    broken = ood.report(rows, ["tua;"] * len(rows))
    assert broken["probe_accuracy"] < 0.1
    assert "expected:" in ood.format_report(broken)


def test_ood_prompts_never_appear_in_generated_data():
    # the OOD set is only a fair test if training never contains its exact requests.
    # "format" cases are template phrases with odd casing/spacing on purpose, so for
    # them only the exact string must be absent.
    generated = [e["prompt"] for e in dataset.generate(3000, seed=11)]
    exact = {p.strip() for p in generated}
    normalized = {_norm(p) for p in generated}
    for tag, prompt, _ in ood.OOD_CASES:
        assert prompt.strip() not in exact, prompt
        if tag != "format":
            assert _norm(prompt) not in normalized, prompt
