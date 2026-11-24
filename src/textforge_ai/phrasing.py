"""Varied English for the synthetic prompts (phase 6, docs/05-generalizacao.md).

The first model learned 2-3 fixed phrases per instruction and failed on anything
else (wrong quotes, "the input" instead of "the text", "after index 2", other ways
to chain steps). This module widens the *surface* of the prompts without touching
their meaning:

* many more templates per instruction (``EXTRA``), including 0-based ``index N``
  and 1-based ``number N`` forms (placeholders ``{i0}`` and ``{n0}``);
* quoting styles: "double", 'single', none, “smart”, `backticks`;
* names for common values ("a space", "a comma", "a period");
* synonyms for "the text" (input, string, content, file);
* many ways to chain steps (lists, "and", "then", "after that", "a pipeline that ...");
* noise: lowercase/UPPERCASE, missing punctuation, double spaces, small typos.

Literals (quoted values) are never altered by the noise: changing them would change
the meaning of the request.
"""

from __future__ import annotations

import random
import re
from dataclasses import dataclass

from . import catalog as C

# --- extra templates ---------------------------------------------------------------
# {0}, {1}: rendered argument; {iK}: raw 0-based value; {nK}: value + 1 (1-based).
EXTRA: dict[str, tuple[str, ...]] = {
    "tbs": ("trim the text", "trim both sides", "trim whitespace from both ends of the text",
            "strip the text", "remove leading and trailing spaces",
            "clean up the spaces around the text", "remove the surrounding whitespace"),
    "tls": ("trim the left side", "remove leading whitespace", "strip the spaces at the start",
            "trim the beginning of the text", "remove the whitespace before the text",
            "left-trim the text"),
    "trs": ("trim the right side", "remove trailing spaces", "strip the whitespace at the end",
            "trim the end of the text", "remove the whitespace after the text",
            "right-trim the text"),
    "rmws": ("remove all the spaces", "delete all whitespace", "remove every space and newline",
             "strip all whitespace from the text", "remove whitespaces",
             "get rid of all whitespace", "remove all spaces from the text",
             "remove all whitespace from the text", "delete the whitespace"),
    "tua": ("make the text uppercase", "convert the text to upper case",
            "put everything in capital letters", "uppercase everything",
            "change the whole text to uppercase", "turn the text into all caps",
            "make it all caps", "convert to uppercase", "capitalize every letter"),
    "tla": ("make the text lowercase", "convert the text to lower case",
            "put everything in lowercase", "lowercase everything",
            "change the whole text to lowercase", "turn the text into small letters",
            "make it all lowercase", "convert to lowercase"),
    "cfw": ("capitalize the first word of the text",
            "uppercase the first letter of the first word", "make the first word title case",
            "capitalize the opening word"),
    "clw": ("capitalize the final word", "uppercase the first letter of the last word",
            "make the last word title case", "capitalize the last word of the text"),
    "cts": ("capitalize word {n0}", "capitalize the word at index {i0}",
            "make word number {n0} start with a capital letter", "title-case the {0} word"),
    "tucw": ("make the {0} word uppercase", "uppercase word {n0}",
             "put the {0} word in capital letters", "convert word number {n0} to uppercase",
             "uppercase the word at index {i0}"),
    "tlcw": ("make the {0} word lowercase", "lowercase word {n0}",
             "convert word number {n0} to lowercase", "lowercase the word at index {i0}"),
    "tucs": ("uppercase character {n0}", "capitalize the character at index {i0}",
             "convert the {0} letter to uppercase", "uppercase the char at index {i0}"),
    "jkbc": ("turn the text into kebab-case", "kebab-case the text", "use kebab case",
             "join the words with dashes", "convert to kebab case"),
    "jsnc": ("turn the text into snake_case", "snake-case the text", "use snake case",
             "join the words with underscores", "convert to snake case"),
    "jcmc": ("turn the text into camelCase", "camel-case the text", "use camel case",
             "make the text camelCase", "convert to camel case"),
    "jpsc": ("turn the text into PascalCase", "use pascal case", "make the text PascalCase",
             "pascal-case the text", "convert to pascal case"),
    "ate": ("append {0}", "add {0} at the end", "add {0} at the end of the text",
            "append {0} to the end of the text", "append {0} to the input",
            "put {0} after the text", "add {0} to the end of the text", "insert {0} at the end",
            "concatenate {0} at the end", "add the suffix {0}", "end the text with {0}",
            "add {0} to the end of the input", "append {0} at the end"),
    "atb": ("prepend {0}", "add {0} at the start", "add {0} to the beginning of the text",
            "prepend {0} to the beginning of the text", "prepend {0} to the input",
            "put {0} before the text", "insert {0} at the beginning", "add the prefix {0}",
            "start the text with {0}", "put {0} at the front", "add {0} at the beginning",
            "prepend {0} to the start of the input"),
    "apl": ("add a new line {0} at the end", "append the line {0}",
            "add a final line with {0}", "add {0} as the last line",
            "write {0} on a new line at the end"),
    "pln": ("add a line {0} at the beginning", "prepend the line {0}",
            "add {0} as the first line", "write {0} on a new line at the top",
            "insert {0} as a header line"),
    "ins": ("insert {1} after index {i0}", "insert {1} at index {n0}",
            "insert {1} after character {n0}", "add {1} after the {0} char",
            "put {1} after the {0} letter", "insert {1} after the first {n0} characters"),
    "padl": ("left-pad the text with {0} to {1} characters",
             "pad the start with {0} until the length is {1}",
             "fill the left side with {0} up to {1} characters"),
    "padr": ("right-pad the text with {0} to {1} characters",
             "pad the end with {0} until the length is {1}",
             "fill the right side with {0} up to {1} characters"),
    "rpt": ("repeat the text {0} times", "repeat it {0} times", "make {0} copies of the text",
            "write the text {0} times in a row"),
    "raw": ("replace {0} with {1}", "replace all {0} with {1}", "change {0} to {1}",
            "change every {0} into {1}", "substitute {1} for every {0}",
            "replace each occurrence of {0} with {1}", "turn every {0} into {1}",
            "replace {0} by {1}", "replace all instances of {0} with {1}",
            "change all {0} to {1}"),
    "rfw": ("replace the first occurrence of {0} with {1}", "change the first {0} to {1}",
            "replace just the first {0} with {1}", "only replace the first {0} with {1}",
            "turn the first {0} into {1}"),
    "rlw": ("replace the last occurrence of {0} with {1}", "change the last {0} to {1}",
            "replace just the last {0} with {1}", "only replace the last {0} with {1}",
            "turn the last {0} into {1}"),
    "rnw": ("replace occurrence number {n2} of {0} with {1}",
            "change the {2} occurrence of {0} to {1}", "replace the {2} {0} by {1}"),
    "rcw": ("replace the first {2} {0} with {1}", "change the first {2} occurrences of {0} to {1}",
            "replace {0} with {1} only {2} times", "replace at most {2} {0} with {1}"),
    "rmp": ("remove {0}", "delete {0}", "remove all {0}", "delete every {0}",
            "remove all occurrences of {0}", "erase every {0}", "erase all instances of {0}",
            "take out every {0}", "get rid of {0}"),
    "emj": ("extract all {0} and join them with {1}", "keep only {0}, separated by {1}",
            "list every {0} joined with {1}", "collect all matches of {0} separated by {1}"),
    "dlf": ("remove the first character", "delete the first char", "drop the first character",
            "cut the first character", "remove the leading character"),
    "dll": ("remove the last character", "delete the last char", "drop the last character",
            "cut the final character", "remove the trailing character"),
    "dls": ("delete character {n0}", "remove the character at index {i0}",
            "delete the char at index {i0}", "drop the {0} char", "remove the {0} letter"),
    "dlb": ("delete the first {0} characters", "remove the first {0} chars",
            "cut off the first {0} characters", "skip the first {0} characters",
            "drop {0} characters from the start"),
    "rev": ("reverse it", "invert the text", "flip the text backwards",
            "spell the text backwards", "reverse the input", "reverse the characters"),
    "rtl": ("rotate left by {0}", "rotate the text {0} characters left",
            "shift everything left by {0} characters", "rotate the string to the left by {0}"),
    "rtr": ("rotate right by {0}", "rotate the text {0} characters right",
            "shift everything right by {0} characters", "rotate the string to the right by {0}"),
    "dle": ("remove empty lines", "delete blank lines", "remove all the empty lines",
            "drop blank lines", "get rid of empty lines", "remove lines that are empty"),
    "uniqln": ("remove the duplicate lines", "delete repeated lines", "deduplicate the lines",
               "keep each line only once", "remove repeated lines"),
    "revln": ("reverse the lines", "reverse the line order", "invert the order of the lines",
              "flip the order of the lines", "put the last line first and so on"),
    "dln": ("delete line {n0}", "remove line number {n0}", "delete the line at index {i0}",
            "drop the {0} line", "remove the {0} line of the text"),
    "klm": ("keep the lines with {0}", "keep only lines containing {0}",
            "filter the lines to those with {0}", "remove every line that does not contain {0}",
            "only keep the lines that have {0}"),
    "dlm": ("remove lines containing {0}", "delete the lines with {0}",
            "remove every line with {0}", "delete any line that contains {0}",
            "filter out the lines with {0}"),
    "joinln": ("join the lines with {0}", "merge all lines using {0}",
               "combine the lines into one, separated by {0}",
               "put all lines on one line separated by {0}",
               "join the lines using {0} as separator"),
    "urle": ("url encode the text", "encode the text for a URL", "percent encode the text",
             "make the text URL-safe"),
    "htmle": ("escape the html", "html escape the text", "escape html special characters",
              "encode the text as HTML entities"),
    "jsone": ("escape the text as JSON", "json escape the text", "make the text a json string",
              "encode the text as a JSON string"),
}

#: how people name some common values instead of quoting them
NAMED_VALUES: dict[str, tuple[str, ...]] = {
    " ": ("a space", "a single space", "spaces"),
    ", ": ("a comma and a space", "comma plus space"),
    "; ": ("a semicolon and a space",),
    ".": ("a period", "a dot", "a full stop"),
    "!": ("an exclamation mark", "an exclamation point"),
    "-": ("a dash", "a hyphen"),
    "*": ("an asterisk", "a star"),
    "#": ("a hash sign",),
    "=": ("an equals sign",),
}
_COUNT_WORDS = {1: "one", 2: "two", 3: "three", 4: "four", 5: "five"}

_TEXT_NOUNS = ("the text", "the input", "the string", "the content", "the file",
               "the input text", "the whole text")

# --- prompt framing ---------------------------------------------------------------
_SINGLE_FRAMES = ("{}", "please {}", "{} please", "can you {}?", "could you {}?",
                  "I want to {}", "I need to {}", "I'd like to {}", "help me {}", "now {}",
                  "just {}", "{} for me", "I need you to {}", "let's {}")
_PIPELINE_FRAMES = ("create a pipeline that {}", "write a pipeline that {}",
                    "make a pipeline that {}", "I need a pipeline that {}",
                    "a pipeline that {}", "generate a pipeline which {}",
                    "build a textforge pipeline that {}", "write a textforge pipeline that {}",
                    "I want a pipeline which {}", "I need a pipeline which {}")
_GAP_CONNECTORS = (", then ", " then ", " and then ", ", and then ", " and ", ", ", "; ",
                   ". Then ", ". After that, ", ". Next, ", ", after that ", ". ",
                   ", next ", " & ")
_LAST_CONNECTORS = (" and ", ", and ", ", and finally ", " and finally ", ". Finally, ",
                    ", then finally ", ", and lastly ", ". Lastly, ", " and at the end ")
_LIST_MARKERS = ("{}. ", "{}) ", "- ", "* ", "step {}: ", "{} - ")

_ADVERBS = ("only", "just")


@dataclass
class Style:
    """Choices that stay the same for one whole prompt."""

    quote: str  # double, single, none, smart, backtick
    words: bool  # ordinals/counts as words ("third", "two") or digits ("3rd", "2")
    noun: str | None  # replacement for "the text", or None to keep it
    named: bool  # say "a space" instead of " " when possible
    case: str  # keep, lower, upper
    typo: bool
    double_space: bool

    @classmethod
    def sample(cls, rng: random.Random) -> Style:
        quote = rng.choices(("double", "single", "none", "smart", "backtick"),
                            weights=(40, 38, 12, 5, 5))[0]
        return cls(
            quote=quote,
            words=rng.random() < 0.5,
            noun=rng.choice(_TEXT_NOUNS) if rng.random() < 0.45 else None,
            named=rng.random() < 0.35,
            case=rng.choices(("keep", "lower", "upper"), weights=(84, 13, 3))[0],
            typo=rng.random() < 0.05,
            double_space=rng.random() < 0.03,
        )


_BARE_OK = re.compile(r"^[A-Za-z0-9À-ſ_-]+$")


def quote_value(value: str, style: Style) -> str:
    q = style.quote
    if q == "none" and not _BARE_OK.match(value):
        q = "double"
    if q == "single" and "'" in value:
        q = "double"
    return {"double": f'"{value}"', "single": f"'{value}'", "none": value,
            "smart": f"“{value}”", "backtick": f"`{value}`"}[q]


def render(kind: str, value: int | str, style: Style, rng: random.Random) -> str:
    if kind in (C.WORD, C.REPL, C.TEXT, C.SEP, C.PAD_CHAR):
        value = str(value)
        if kind in (C.TEXT, C.SEP, C.PAD_CHAR) and style.named and value in NAMED_VALUES:
            return rng.choice(NAMED_VALUES[value])
        return quote_value(value, style)
    if kind in (C.CHAR_IDX, C.WORD_IDX, C.LINE_IDX, C.NTH):
        return C.ordinal(int(value) + 1, words=style.words)
    if kind in (C.COUNT, C.CUT) and style.words and int(value) in _COUNT_WORDS:
        return _COUNT_WORDS[int(value)]
    return str(value)


def third_person(clause: str) -> str:
    """'remove all spaces' -> 'removes all spaces' (for 'a pipeline that ...')."""
    words = clause.split(" ")
    i = 1 if len(words) > 1 and words[0] in _ADVERBS else 0
    verb = words[i]
    if verb.endswith(("sh", "ch", "x", "s", "z")):
        words[i] = verb + "es"
    elif verb.endswith("y") and len(verb) > 2 and verb[-2] not in "aeiou":
        words[i] = verb[:-1] + "ies"
    else:
        words[i] = verb + "s"
    return " ".join(words)


def _typo(text: str, rng: random.Random) -> str:
    """One small typo in a plain word (never inside a {placeholder})."""
    words = [(m.start(), m.group()) for m in re.finditer(r"(?<![{\w])[a-z]{5,}(?![\w}])", text)]
    if not words:
        return text
    start, word = rng.choice(words)
    i = rng.randrange(1, len(word) - 1)
    kind = rng.randrange(3)
    if kind == 0:
        new = word[:i] + word[i + 1] + word[i] + word[i + 2:]  # swap
    elif kind == 1:
        new = word[:i] + word[i + 1:]  # drop
    else:
        new = word[:i] + word[i] + word[i:]  # double
    return text[:start] + new + text[start + len(word):]


def apply_case(text: str, style: Style) -> str:
    if style.case == "lower":
        return text.lower()
    if style.case == "upper":  # keep {n0}/{i0} placeholders intact
        return re.sub(r"\{([IN]?\d)\}", lambda m: m.group().lower(), text.upper())
    return text


def clause(step, style: Style, rng: random.Random, *, conjugate: bool = False,
           typo: bool = False) -> str:
    spec = C.BY_MNEMONIC[step.mnemonic]
    pool = spec.templates + EXTRA.get(step.mnemonic, ())
    template = rng.choice(pool)
    if style.noun and "the text" in template:
        template = template.replace("the text", style.noun, 1)
    if conjugate:
        template = third_person(template)
    if typo:
        template = _typo(template, rng)
    # Noise only touches the template; values are rendered afterwards, untouched.
    template = apply_case(template, style)
    if style.double_space:
        template = template.replace(" ", "  ", 1)
    rendered = [render(k, v, style, rng) for k, v in zip(spec.args, step.values)]
    extra = {}
    for index, value in enumerate(step.values):
        if isinstance(value, int):
            extra[f"i{index}"] = str(value)
            extra[f"n{index}"] = str(value + 1)
    return template.format(*rendered, **extra)


def _cap(text: str, style: Style) -> str:
    return text if style.case == "lower" else text[:1].upper() + text[1:]


def describe(steps, rng: random.Random) -> str:
    """A varied English request for ``steps`` (see module docstring)."""
    style = Style.sample(rng)
    pipeline_frame = rng.random() < 0.2
    typo_at = rng.randrange(len(steps)) if style.typo else -1
    clauses = [clause(s, style, rng, conjugate=pipeline_frame, typo=i == typo_at)
               for i, s in enumerate(steps)]
    t = lambda s: apply_case(s, style)

    if pipeline_frame:
        body = _join_inline(clauses, rng, t)
        text = t(rng.choice(_PIPELINE_FRAMES)).replace("{}", body, 1)
        return _cap(text, style) + rng.choice(("", ".", "."))

    if len(clauses) == 1:
        frame = t(rng.choice(_SINGLE_FRAMES))
        text = _cap(frame.replace("{}", clauses[0], 1), style)
        return text if text.endswith("?") else text + rng.choice(("", ".", ".", "!"))

    layout = rng.choices(("inline", "list", "first"), weights=(60, 20, 20))[0]
    if layout == "list":
        marker = rng.choice(_LIST_MARKERS)
        sep = rng.choice(("\n", "\n", " "))
        end = rng.choice(("", "."))
        return sep.join(t(marker.format(i)) + _cap(c, style) + end
                        for i, c in enumerate(clauses, 1))
    if layout == "first":
        head = t(rng.choice(("first ", "first, ", "start by ", "to begin, ")))
        if head.startswith(("start by", "START BY")):
            head = t("first ")
        body = head + _join_inline(clauses, rng, t)
        return _cap(body, style) + rng.choice(("", "."))
    frame = t(rng.choice(("{}", "{}", "please {}", "can you {}?", "I want to {}")))
    text = _cap(frame.replace("{}", _join_inline(clauses, rng, t), 1), style)
    return text if text.endswith("?") else text + rng.choice(("", ".", "."))


def _join_inline(clauses: list[str], rng: random.Random, t) -> str:
    if len(clauses) == 1:
        return clauses[0]
    text = clauses[0]
    for index, c in enumerate(clauses[1:], 1):
        last = index == len(clauses) - 1
        connector = t(rng.choice(_LAST_CONNECTORS if last and rng.random() < 0.6
                                 else _GAP_CONNECTORS))
        if connector.rstrip().endswith((".", ":")) and c:
            c = c[:1].upper() + c[1:] if t("a") == "a" else c
        text += connector + c
    return text
