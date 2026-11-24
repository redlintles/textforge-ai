"""Hand-written, out-of-distribution (OOD) requests: the honest generalisation test.

The synthetic dataset (synth.py) and its validation split come from the same
templates, so a high score there only means "learned the templates". These requests
were written by hand, in the way people actually type, *before* the generator was
made more varied (docs/05-generalizacao.md), and are never used for training
(``tests/test_ood.py`` checks that no generated prompt equals one of them).

Each case is ``(tag, prompt, steps)``; steps are ``(mnemonic, *args)`` with
TextForge's own 0-based indices. The expected pipeline is built with the official
builder, and probes are generated and executed exactly like synthetic examples, so
``model-eval`` / ``ood-eval`` score them with the same probe metric.

Tags (what the case stresses):
  paraphrase - same meaning, wording not in the templates
  format     - quotes, casing, typos, missing punctuation
  literal    - values never seen in training (names, other languages, numbers)
  compose    - several steps joined in free-form ways
  semantic   - needs world knowledge ("twice", "a period")
"""

from __future__ import annotations

import random

from .engine import canonical_text, run_builder
from .synth import Step, build_pipeline
from .texts import make_text

OOD_CASES: tuple[tuple[str, str, tuple[tuple, ...]], ...] = (
    # --- the user's own first requests --------------------------------------------
    ("compose", ("remove all whitespaces, append 'banana' to the input and prepend 'laranja' "
     "to the beginning of the text, and finally insert 'terry lee' after index 2"),
     (("rmws",), ("ate", "banana"), ("atb", "laranja"), ("ins", 2, "terry lee"))),
    ("compose", ("Create a pipeline that removes all the whitespace from a file, adds 'banana' "
     "at the end and 'laranja' at the start"),
     (("rmws",), ("ate", "banana"), ("atb", "laranja"))),
    # --- paraphrase -------------------------------------------------------------------
    ("paraphrase", "make everything upper case", (("tua",),)),
    ("paraphrase", "turn the whole thing into lowercase letters", (("tla",),)),
    ("paraphrase", "change dog to puppy everywhere", (("raw", "dog", "puppy"),)),
    ("paraphrase", "replace only the first 'river' by 'lake'", (("rfw", "river", "lake"),)),
    ("paraphrase", "the last occurrence of 'tiger' should become 'lion'",
     (("rlw", "tiger", "lion"),)),
    ("paraphrase", "replace the second 'apple' with 'mango'", (("rnw", "apple", "mango", 1),)),
    ("paraphrase", "get rid of every occurrence of 'never'", (("rmp", "never"),)),
    ("paraphrase", "delete the word 'slowly' everywhere", (("rmp", "slowly"),)),
    ("paraphrase", "drop the first line", (("dln", 0),)),
    ("paraphrase", "remove line 2", (("dln", 1),)),
    ("paraphrase", "get rid of the blank lines", (("dle",),)),
    ("paraphrase", "only keep lines that mention 'eagle'", (("klm", "eagle"),)),
    ("paraphrase", "throw away any line containing 'winter'", (("dlm", "winter"),)),
    ("paraphrase", "join every line using ' | '", (("joinln", " | "),)),
    ("paraphrase", "trim the spaces at both ends of the input", (("tbs",),)),
    ("paraphrase", "strip leading spaces", (("tls",),)),
    ("paraphrase", "remove trailing whitespace", (("trs",),)),
    ("paraphrase", "capitalize the very first word", (("cfw",),)),
    ("paraphrase", "make the last word capitalized", (("clw",),)),
    ("paraphrase", "capitalize word number 3", (("cts", 2),)),
    ("paraphrase", "put the 2nd word in all caps", (("tucw", 1),)),
    ("paraphrase", "make the character at index 4 uppercase", (("tucs", 4),)),
    ("paraphrase", "delete the character at index 3", (("dls", 3),)),
    ("paraphrase", "chop off the last char", (("dll",),)),
    ("paraphrase", "cut the first 3 characters", (("dlb", 3),)),
    ("paraphrase", "reverse the string", (("rev",),)),
    ("paraphrase", "write the input in reverse", (("rev",),)),
    ("paraphrase", "move every character 2 positions to the left", (("rtl", 2),)),
    ("paraphrase", "url encode the input", (("urle",),)),
    ("paraphrase", "escape html characters", (("htmle",),)),
    ("paraphrase", "make it snake case", (("jsnc",),)),
    ("paraphrase", "turn it into camel case", (("jcmc",),)),
    ("paraphrase", "convert the words to kebab case", (("jkbc",),)),
    ("paraphrase", "add '!!!' at the end", (("ate", "!!!"),)),
    ("paraphrase", "put 'Hello, ' in front of the text", (("atb", "Hello, "),)),
    ("paraphrase", "insert 'xyz' after the 5th char", (("ins", 4, "xyz"),)),
    ("paraphrase", "add a line saying 'THE END' at the end", (("apl", "THE END"),)),
    ("paraphrase", "add the header line '# Notes' at the top", (("pln", "# Notes"),)),
    ("paraphrase", "pad the left side with '0' up to 80 chars", (("padl", "0", 80),)),
    ("paraphrase", "keep only the 'fox' matches, separated by ' - '",
     (("emj", "fox", " - "),)),
    # --- format: quotes, casing, typos, punctuation ------------------------------------
    ("format", "Remove All Whitespace", (("rmws",),)),
    ("format", "UPPERCASE EVERYTHING", (("tua",),)),
    ("format", "replace every “orange” with “lemon”",
     (("raw", "orange", "lemon"),)),
    ("format", "replace `table` with `desk`", (("raw", "table", "desk"),)),
    ("format", "replace all occurrences of fox with wolf", (("raw", "fox", "wolf"),)),
    ("format", "repalce every 'bridge' with 'tunnel'", (("raw", "bridge", "tunnel"),)),
    ("format", "delte the empty lines", (("dle",),)),
    ("format", "reverse the order of lines pls", (("revln",),)),
    ("format", "lowercase   the   text", (("tla",),)),
    # --- literal: values never seen in training ---------------------------------------
    ("literal", "append 'Terry Lee' to the end", (("ate", "Terry Lee"),)),
    ("literal", "prepend 'Olá ' to the text", (("atb", "Olá "),)),
    ("literal", "replace 'city' with 'São Paulo'", (("raw", "city", "São Paulo"),)),
    ("literal", "add '2026-11-24' at the beginning", (("atb", "2026-11-24"),)),
    ("literal", "insert ' -- ' after the 1st character", (("ins", 0, " -- "),)),
    ("literal", "append 'zqx'", (("ate", "zqx"),)),
    ("literal", "prepend 'laranja'", (("atb", "laranja"),)),
    ("literal", "append the word 'goodbye' to the text", (("ate", "goodbye"),)),
    ("literal", "replace 'rabbit' with 'Kaninchen'", (("raw", "rabbit", "Kaninchen"),)),
    # --- compose: several steps, free-form joins --------------------------------------
    ("compose", ("lowercase everything, then replace 'dog' with 'puppy' and finally remove "
     "empty lines"), (("tla",), ("raw", "dog", "puppy"), ("dle",))),
    ("compose", "Could you trim the text and then capitalize the first word?",
     (("tbs",), ("cfw",))),
    ("compose", "first remove duplicate lines. after that, reverse their order",
     (("uniqln",), ("revln",))),
    ("compose", "replace 'summer' with 'autumn' and 'winter' with 'spring'",
     (("raw", "summer", "autumn"), ("raw", "winter", "spring"))),
    ("compose", "delete the lines with 'never' and uppercase the rest",
     (("dlm", "never"), ("tua",))),
    ("compose", "add '>> ' before the text and ' <<' after it",
     (("atb", ">> "), ("ate", " <<"))),
    ("compose", "i want a pipeline that deletes the first char and the last char",
     (("dlf",), ("dll",))),
    ("compose", "1) trim both sides 2) lowercase 3) remove the 2nd line",
     (("tbs",), ("tla",), ("dln", 1))),
    ("compose", "- delete empty lines\n- reverse the lines\n- join them with '; '",
     (("dle",), ("revln",), ("joinln", "; "))),
    ("compose", "Please make the text all lowercase; then delete each line that has 'river'",
     (("tla",), ("dlm", "river"))),
    ("compose", "uppercase the 3rd word then capitalize the last one", (("tucw", 2), ("clw",))),
    ("compose", "remove spaces at the beginning and at the end, then remove the last character",
     (("tbs",), ("dll",))),
    ("compose", "swap 'happy' for 'sad', delete line 1, and append 'fim'",
     (("raw", "happy", "sad"), ("dln", 0), ("ate", "fim"))),
    ("compose", "write a pipeline that converts the text to uppercase and removes every 'THE'",
     (("tua",), ("rmp", "THE"))),
    ("compose", "reverse the lines and then the whole text", (("revln",), ("rev",))),
    ("compose", ("build a pipeline which strips the trailing spaces, puts '[x] ' at the start "
     "and makes the first word uppercase"), (("trs",), ("atb", "[x] "), ("tucw", 0))),
    # --- semantic: needs a bit of world knowledge --------------------------------------
    ("semantic", "repeat the text twice", (("rpt", 2),)),
    ("semantic", "make the 1st word uppercase and add a period at the end",
     (("tucw", 0), ("ate", "."))),
    ("semantic", "add an exclamation mark to the end", (("ate", "!"),)),
    ("semantic", "join all the lines with a space", (("joinln", " "),)),
)

#: words a probe must contain for a step to have something to act on
_PATTERN_ARG = {"raw": 0, "rfw": 0, "rlw": 0, "rnw": 0, "rcw": 0, "rmp": 0, "emj": 0,
                "klm": 0, "dlm": 0}


def _required_words(steps: list[Step]) -> list[str]:
    words = []
    for step in steps:
        if step.mnemonic in _PATTERN_ARG:
            words.append(str(step.values[_PATTERN_ARG[step.mnemonic]]).lower())
    return words


def build_examples(n_probes: int = 6, seed: int = 0) -> list[dict]:
    """Turn OOD_CASES into dataset rows (same schema as synthetic examples)."""
    rng = random.Random(seed)
    rows = []
    for index, (tag, prompt, raw_steps) in enumerate(OOD_CASES):
        steps = [Step(s[0], tuple(s[1:])) for s in raw_steps]
        required = _required_words(steps)
        pipeline = build_pipeline(steps)
        for _attempt in range(200):
            inputs = [make_text(rng, required) for _ in range(n_probes)]
            if "htmle" in raw_steps[0]:  # probes need something to escape
                inputs = [f"{t} <b>{t[:4]} & co</b>" for t in inputs]
            results = run_builder(pipeline, inputs)
            if all(r.ok for r in results) and any(
                r.output != i for r, i in zip(results, inputs)
            ) and all(r.output for r in results):
                break
        else:  # pragma: no cover - a case that can never act is a bug in the case
            raise ValueError(f"OOD case {index} never changes its probes: {prompt!r}")
        rows.append({
            "id": f"ood-{index}",
            "tag": tag,
            "prompt": prompt,
            "pipeline": canonical_text(pipeline),
            "steps": [s.mnemonic for s in steps],
            "probes": [{"input": i, "output": r.output} for i, r in zip(inputs, results)],
        })
    return rows


def report(rows: list[dict], predictions: list[str]) -> dict:
    """Probe accuracy overall and per tag, plus every failure (prompt, expected, got)."""
    from .evaluate import evaluate

    summary, records = evaluate(rows, {r["id"]: p for r, p in zip(rows, predictions)})
    by_tag: dict[str, list[int]] = {}
    failures = []
    for row, record, predicted in zip(rows, records, predictions):
        hits = by_tag.setdefault(row["tag"], [0, 0])
        hits[0] += record["correct"]
        hits[1] += 1
        if not record["correct"]:
            failures.append({"tag": row["tag"], "prompt": row["prompt"],
                             "expected": row["pipeline"], "got": predicted,
                             "valid": record["valid"]})
    return {
        "probe_accuracy": summary["probe_accuracy"],
        "valid_rate": summary["valid_rate"],
        "by_tag": {tag: {"correct": c, "total": t, "accuracy": c / t}
                   for tag, (c, t) in sorted(by_tag.items())},
        "failures": failures,
    }


def format_report(result: dict, max_failures: int = 20) -> str:
    lines = [(f"OOD probe accuracy: {result['probe_accuracy']:.0%}   "
             f"(valid pipelines: {result['valid_rate']:.0%})"), ""]
    for tag, r in result["by_tag"].items():
        lines.append(f"  {tag:<11} {r['correct']:>3}/{r['total']:<3} {r['accuracy']:.0%}")
    shown = result["failures"][:max_failures]
    if shown:
        lines += ["", f"failures ({len(result['failures'])}, showing {len(shown)}):"]
        for f in shown:
            got = f["got"].replace("\n", " ") if f["got"] else "(empty)"
            mark = "" if f["valid"] else "  [invalid]"
            lines += [f"- [{f['tag']}] {f['prompt']!r}",
                      f"    expected: {f['expected'].replace(chr(10), ' ')}",
                      f"    got:      {got}{mark}"]
    return "\n".join(lines)
