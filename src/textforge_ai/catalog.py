"""Catalog of the TextForge instructions the model is allowed to use (v1 subset).

Each :class:`InstructionSpec` says:

* which builder method creates it (so the engine, not us, decides the syntax);
* the *kind* of every argument (how to sample it and how to say it in English);
* several English templates. ``{0}``, ``{1}`` ... are the arguments already rendered
  for humans (quoted strings, 1-based ordinals, ...).

Important convention: humans count from 1 ("the second word"), TextForge counts
from 0. The conversion happens here, in :func:`render_arg`, so the model has to
learn it - exactly like it will have to with real prompts.
"""

from __future__ import annotations

from dataclasses import dataclass

# --- argument kinds -------------------------------------------------------------
# Value domains are defined in synth.py; here we only define how they *read*.
WORD = "word"  # literal pattern taken from the text vocabulary
REPL = "repl"  # replacement word
TEXT = "text"  # short literal to add (may contain spaces/punctuation)
SEP = "sep"  # separator used to join things
CHAR_IDX = "char_idx"  # 0-based character index, said as an ordinal
WORD_IDX = "word_idx"  # 0-based word index, said as an ordinal
LINE_IDX = "line_idx"  # 0-based line index, said as an ordinal
NTH = "nth"  # 0-based occurrence index, said as an ordinal
COUNT = "count"  # plain cardinal number (times, how many)
CUT = "cut"  # number of leading characters to drop (cardinal)
PAD_CHAR = "pad_char"
PAD_LEN = "pad_len"

ARG_KINDS = (WORD, REPL, TEXT, SEP, CHAR_IDX, WORD_IDX, LINE_IDX, NTH, COUNT, CUT, PAD_CHAR, PAD_LEN)

_ORDINAL_WORDS = {
    1: "first", 2: "second", 3: "third", 4: "fourth", 5: "fifth",
    6: "sixth", 7: "seventh", 8: "eighth", 9: "ninth", 10: "tenth",
}


def ordinal(n: int, *, words: bool = True) -> str:
    """1 -> 'first' (or '1st'). ``n`` is 1-based."""
    if words and n in _ORDINAL_WORDS:
        return _ORDINAL_WORDS[n]
    if 10 <= n % 100 <= 20:
        suffix = "th"
    else:
        suffix = {1: "st", 2: "nd", 3: "rd"}.get(n % 10, "th")
    return f"{n}{suffix}"


def quote(text: str) -> str:
    return f'"{text}"'


def render_arg(kind: str, value: int | str, *, style: int = 0) -> str:
    """Render one argument the way a person would write it in a prompt."""
    if kind in (WORD, REPL, TEXT, SEP, PAD_CHAR):
        return quote(str(value))
    if kind in (CHAR_IDX, WORD_IDX, LINE_IDX, NTH):
        return ordinal(int(value) + 1, words=style % 2 == 0)
    if kind in (COUNT, CUT, PAD_LEN):
        return str(value)
    raise ValueError(f"unknown argument kind {kind!r}")


@dataclass(frozen=True)
class InstructionSpec:
    mnemonic: str
    method: str
    args: tuple[str, ...]
    templates: tuple[str, ...]
    #: sample weight; rarely useful instructions can be down-weighted
    weight: float = 1.0


def _s(mnemonic, method, args, *templates, weight=1.0):
    return InstructionSpec(mnemonic, method, tuple(args), tuple(templates), weight)


CATALOG: tuple[InstructionSpec, ...] = (
    # --- whitespace -------------------------------------------------------------
    _s("tbs", "trim_both_sides", [],
       "trim the whitespace on both ends", "strip leading and trailing whitespace",
       "remove the spaces at the start and end of the text"),
    _s("tls", "trim_left_side", [],
       "trim the leading whitespace", "strip whitespace from the start of the text",
       "remove the spaces at the beginning"),
    _s("trs", "trim_right_side", [],
       "trim the trailing whitespace", "strip whitespace from the end of the text",
       "remove the spaces at the end"),
    _s("rmws", "remove_whitespace", [],
       "remove all whitespace", "delete every whitespace character",
       "strip out all spaces and line breaks", weight=0.5),
    # --- case -----------------------------------------------------------------
    _s("tua", "to_uppercase_all", [],
       "convert everything to uppercase", "make the whole text uppercase", "uppercase the text"),
    _s("tla", "to_lowercase_all", [],
       "convert everything to lowercase", "make the whole text lowercase", "lowercase the text"),
    _s("cfw", "capitalize_first_word", [],
       "capitalize the first word", "make the first word start with a capital letter"),
    _s("clw", "capitalize_last_word", [],
       "capitalize the last word", "make the last word start with a capital letter"),
    _s("cts", "capitalize_single_word", [WORD_IDX],
       "capitalize the {0} word", "make the {0} word start with a capital letter"),
    _s("tucw", "to_uppercase_word", [WORD_IDX],
       "uppercase the {0} word", "make the {0} word all caps"),
    _s("tlcw", "to_lowercase_word", [WORD_IDX],
       "lowercase the {0} word", "make the {0} word all lowercase"),
    _s("tucs", "to_uppercase_single", [CHAR_IDX],
       "uppercase the {0} character", "make the {0} character a capital letter", weight=0.5),
    _s("jkbc", "join_to_kebab_case", [],
       "convert the text to kebab-case", "join the words with hyphens in kebab case", weight=0.6),
    _s("jsnc", "join_to_snake_case", [],
       "convert the text to snake_case", "join the words with underscores in snake case",
       weight=0.6),
    _s("jcmc", "join_to_camel_case", [],
       "convert the text to camelCase", "join the words in camel case", weight=0.6),
    _s("jpsc", "join_to_pascal_case", [],
       "convert the text to PascalCase", "join the words in pascal case", weight=0.6),
    # --- adding text ----------------------------------------------------------
    _s("ate", "add_to_end", [TEXT],
       "add {0} to the end", "append {0} to the text", "put {0} at the very end"),
    _s("atb", "add_to_beginning", [TEXT],
       "add {0} to the beginning", "prepend {0} to the text", "put {0} at the very start"),
    _s("apl", "append_line", [TEXT],
       "append a new line containing {0}", "add a line {0} at the bottom"),
    _s("pln", "prepend_line", [TEXT],
       "insert a line {0} at the top", "add a new first line containing {0}"),
    _s("ins", "insert", [CHAR_IDX, TEXT],
       "insert {1} after the {0} character", "put {1} right after the {0} character",
       weight=0.5),
    _s("padl", "pad_left", [PAD_CHAR, PAD_LEN],
       "pad the text on the left with {0} until it is {1} characters long",
       "left-pad with {0} to a length of {1}", weight=0.4),
    _s("padr", "pad_right", [PAD_CHAR, PAD_LEN],
       "pad the text on the right with {0} until it is {1} characters long",
       "right-pad with {0} to a length of {1}", weight=0.4),
    _s("rpt", "repeat", [COUNT],
       "repeat the whole text {0} times", "duplicate the text so it appears {0} times",
       weight=0.4),
    # --- replacing / removing patterns ---------------------------------------
    _s("raw", "replace_all_with", [WORD, REPL],
       "replace every {0} with {1}", "change all occurrences of {0} to {1}",
       "swap each {0} for {1}", weight=2.0),
    _s("rfw", "replace_first_with", [WORD, REPL],
       "replace the first {0} with {1}", "change only the first occurrence of {0} to {1}",
       weight=1.2),
    _s("rlw", "replace_last_with", [WORD, REPL],
       "replace the last {0} with {1}", "change only the last occurrence of {0} to {1}",
       weight=1.2),
    _s("rnw", "replace_nth_with", [WORD, REPL, NTH],
       "replace the {2} occurrence of {0} with {1}", "change the {2} {0} to {1}", weight=0.8),
    _s("rcw", "replace_count_with", [WORD, REPL, COUNT],
       "replace the first {2} occurrences of {0} with {1}",
       "change up to {2} occurrences of {0} to {1}", weight=0.8),
    _s("rmp", "remove_pattern", [WORD],
       "remove every {0}", "delete all occurrences of {0}", "strip out {0} everywhere"),
    _s("emj", "extract_matches_joined", [WORD, SEP],
       "keep only the occurrences of {0}, joined by {1}",
       "extract every {0} and join them with {1}", weight=0.5),
    # --- characters -------------------------------------------------------------
    _s("dlf", "delete_first", [],
       "delete the first character", "remove the very first character"),
    _s("dll", "delete_last", [],
       "delete the last character", "remove the very last character"),
    _s("dls", "delete_single", [CHAR_IDX],
       "delete the {0} character", "remove the {0} character", weight=0.6),
    _s("dlb", "delete_before", [CUT],
       "remove the first {0} characters", "drop the first {0} characters", weight=0.6),
    _s("rev", "to_reverse", [],
       "reverse the text", "write the text backwards", weight=0.6),
    _s("rtl", "rotate_left", [COUNT],
       "rotate the text left by {0} characters", "shift the text {0} characters to the left",
       weight=0.3),
    _s("rtr", "rotate_right", [COUNT],
       "rotate the text right by {0} characters", "shift the text {0} characters to the right",
       weight=0.3),
    # --- lines ------------------------------------------------------------------
    _s("dle", "delete_empty_lines", [],
       "delete the empty lines", "remove all blank lines"),
    _s("uniqln", "unique_lines", [],
       "remove duplicate lines", "keep only the first copy of each repeated line"),
    _s("revln", "reverse_lines", [],
       "reverse the order of the lines", "put the lines in reverse order"),
    _s("dln", "delete_line", [LINE_IDX],
       "delete the {0} line", "remove the {0} line"),
    _s("klm", "keep_lines_matching", [WORD],
       "keep only the lines that contain {0}", "drop every line without {0}"),
    _s("dlm", "delete_lines_matching", [WORD],
       "delete every line that contains {0}", "remove the lines mentioning {0}"),
    _s("joinln", "join_lines", [SEP],
       "join all lines with {0}", "merge the lines into one, separated by {0}"),
    # --- encodings ----------------------------------------------------------------
    _s("urle", "to_url_encoded", [],
       "url-encode the text", "percent-encode the text for a URL", weight=0.4),
    _s("htmle", "to_html_escaped", [],
       "escape the text for HTML", "html-escape the text", weight=0.4),
    _s("jsone", "to_json_escaped", [],
       "turn the text into a JSON string", "json-escape the text", weight=0.3),
)

BY_MNEMONIC: dict[str, InstructionSpec] = {spec.mnemonic: spec for spec in CATALOG}
