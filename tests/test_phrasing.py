import random
import re

import pytest

from textforge_ai import catalog as C
from textforge_ai import dataset, phrasing
from textforge_ai.synth import Step, sample_steps


@pytest.mark.parametrize("mnemonic", sorted(phrasing.EXTRA))
def test_extra_templates_use_every_argument(mnemonic):
    spec = C.BY_MNEMONIC[mnemonic]
    for template in phrasing.EXTRA[mnemonic]:
        used = {int(i) for i in re.findall(r"\{[in]?(\d)\}", template)}
        assert used == set(range(len(spec.args))), template


def test_every_catalog_instruction_has_extra_phrasings():
    assert set(phrasing.EXTRA) == set(C.BY_MNEMONIC)


def test_index_placeholders_are_zero_and_one_based():
    style = phrasing.Style("double", False, None, False, "keep", False, False)
    rng = random.Random(0)
    texts = {phrasing.clause(Step("ins", (2, "x")), style, rng) for _ in range(200)}
    assert 'insert "x" after index 2' in texts  # 0-based, like TextForge
    assert 'insert "x" at index 3' in texts  # after the first 3 characters
    assert 'add "x" after the 3rd char' in texts


def test_quote_styles():
    def q(kind, value):
        return phrasing.quote_value(value, phrasing.Style(kind, 0, None, 0, "keep", 0, 0))

    assert q("single", "banana") == "'banana'"
    assert q("none", "banana") == "banana"
    assert q("none", "two words") == '"two words"'  # bare only for single tokens
    assert q("single", "it's") == '"it\'s"'
    assert q("smart", "x") == "“x”"


def test_third_person():
    assert phrasing.third_person("remove all spaces") == "removes all spaces"
    assert phrasing.third_person("only replace the first") == "only replaces the first"
    assert phrasing.third_person("push it") == "pushes it"


def test_noise_never_touches_literal_values():
    # casing/typos apply to the template only; the requested values must survive
    rng = random.Random(3)
    for _ in range(400):
        steps = sample_steps(rng)
        prompt = phrasing.describe(steps, rng)
        for step in steps:
            spec = C.BY_MNEMONIC[step.mnemonic]
            for kind, value in zip(spec.args, step.values):
                if kind in (C.WORD, C.REPL) or (kind == C.TEXT and value not in phrasing.NAMED_VALUES):
                    assert str(value) in prompt, (value, prompt)


def test_generated_prompts_are_varied():
    prompts = [e["prompt"] for e in dataset.generate(400, seed=4)]
    assert any("'" in p for p in prompts) and any('"' in p for p in prompts)
    assert any("pipeline" in p.lower() for p in prompts)
    assert any(p == p.lower() for p in prompts)
    assert len({p.split()[0].lower() for p in prompts}) > 40  # many different openings
