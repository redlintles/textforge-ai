import textforge as tf

from textforge_ai import engine


def test_parse_and_run_multiline_pipeline():
    (result,) = engine.run_pipeline_text("tbs;\nraw banana orange;\ntua;", ["  banana!  "])
    assert result.ok and result.output == "ORANGE!"


def test_unknown_instruction_is_reported_not_raised():
    (result,) = engine.run_pipeline_text("foo 1;", ["abc"])
    assert not result.ok
    assert result.error.stage == "persistence"
    assert result.error.kind == "TokenNotFound"


def test_runtime_error_is_structured():
    (result,) = engine.run_pipeline_text("dls 99;", ["abc"])
    assert not result.ok and result.error.stage == "execution"
    assert result.error.kind == "IndexOutOfRange"
    assert result.error.hint  # the engine explains how indices work


def test_canonical_text_roundtrips():
    text = engine.canonical_text(tf.Pipeline().replace_all_with("a b", "c d").add_to_end("!"))
    assert text == "raw 'a b' 'c d';\nate !;"
    assert engine.canonicalize(text) == text


def test_canonicalize_normalizes_quoting():
    assert engine.canonicalize('ate "x y";') == "ate 'x y';"
    assert engine.canonicalize("not a thing;") is None
