import random
import re

import pytest

from textforge_ai import catalog as C
from textforge_ai.engine import canonical_text
from textforge_ai.synth import Step, build_pipeline, describe_step, sample_value
from textforge_ai.texts import VOCAB


@pytest.mark.parametrize("spec", C.CATALOG, ids=lambda s: s.mnemonic)
def test_builder_method_emits_declared_mnemonic(spec):
    rng = random.Random(0)
    step = Step(spec.mnemonic, tuple(sample_value(k, rng) for k in spec.args))
    text = canonical_text(build_pipeline([step]))
    assert text.split()[0].rstrip(";") == spec.mnemonic


@pytest.mark.parametrize("spec", C.CATALOG, ids=lambda s: s.mnemonic)
def test_templates_use_every_argument(spec):
    for template in spec.templates:
        used = {int(i) for i in re.findall(r"\{(\d)\}", template)}
        assert used == set(range(len(spec.args))), template


def test_mnemonics_unique_and_args_known():
    assert len(C.BY_MNEMONIC) == len(C.CATALOG)
    for spec in C.CATALOG:
        assert set(spec.args) <= set(C.ARG_KINDS)


def test_ordinals_are_one_based_for_humans():
    assert C.render_arg(C.WORD_IDX, 0) == "first"
    assert C.render_arg(C.WORD_IDX, 1, style=1) == "2nd"
    assert C.ordinal(11, words=False) == "11th" and C.ordinal(22, words=False) == "22nd"


def test_vocab_has_no_word_inside_another():
    # patterns are regexes; "cat" inside "category" would make descriptions lie
    for a in VOCAB:
        for b in VOCAB:
            if a != b and len(a) > 3:
                assert a not in b, (a, b)


def test_describe_step_quotes_strings():
    text = describe_step(Step("raw", ("apple", "pear")), random.Random(0))
    assert '"apple"' in text and '"pear"' in text
