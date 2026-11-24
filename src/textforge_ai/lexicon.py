"""Literal values beyond the probe vocabulary, so the model learns to *copy* values.

The first dataset used 10 fixed literals for ``ate``/``atb`` and 64 words for
patterns. The model learned those strings by heart and garbled anything else
("laranja" -> "laranaja", "mango" -> "morango"). Sampling values from a large,
varied pool (plus random pseudo-words) forces it to copy from the request instead.
"""

from __future__ import annotations

import random

from .texts import VOCAB

_WORDS = (
    "cat", "wolf", "lion", "puppy", "kitten", "mango", "lemon", "grape", "cherry", "peach",
    "pear", "plum", "melon", "coffee", "tea", "bread", "cheese", "pizza", "pasta", "soup",
    "lake", "sea", "hill", "valley", "island", "beach", "storm", "rain", "snow", "cloud",
    "sun", "moon", "star", "planet", "rocket", "robot", "laptop", "phone", "screen", "mouse",
    "keyboard", "server", "client", "error", "warning", "info", "debug", "user", "admin",
    "guest", "hello", "goodbye", "thanks", "welcome", "yes", "no", "maybe", "ok", "done",
    "todo", "fixme", "draft", "final", "version", "release", "update", "monday", "friday",
    "january", "july", "red", "blue", "green", "black", "white", "pink", "gold", "big",
    "small", "old", "new", "fast", "slow", "hot", "cold", "dark", "light", "music", "movie",
    "game", "school", "office", "house", "street", "road", "car", "train", "plane", "boat",
    "doctor", "teacher", "student", "friend", "family", "spring", "autumn", "fall", "desk",
    "tunnel", "cake", "sugar", "salt", "water", "fire", "earth", "wind", "stone", "glass",
    # other languages (people mix them in)
    "laranja", "banana", "abacaxi", "casa", "gato", "cachorro", "obrigado", "olá",
    "coração", "açúcar", "pão", "maçã", "são", "hola",
    "gracias", "niño", "bonjour", "merci", "café", "danke", "straße",
    "Kaninchen", "ciao", "amore",
)
_FIRST = ("terry", "maria", "joão", "ana", "pedro", "lucas", "julia", "alice", "bob",
          "carlos", "emma", "noah", "li", "yuki", "omar", "sofia", "jo", "rafael")
_LAST = ("lee", "silva", "souza", "smith", "santos", "oliveira", "costa", "brown", "garcia",
         "kim", "nguyen", "ferreira", "lima", "jones")
_SYMBOLS = ("!!!", "...", "?!", "->", "=>", "::", "[ok]", "(c)", "#tag", "@user", "<<", ">>",
            "~", "|", "**", "--->", "+1", "[done]", "(!)", ":)", "...!")
_PHRASES = ("the end", "thank you", "see you soon", "good morning", "to be continued",
            "read more", "all rights reserved", "end of file", "best regards", "new version",
            "hello world", "fim do texto")
_CONSONANTS = "bcdfghjklmnprstvwxyz"
_VOWELS = "aeiou"


def pseudo_word(rng: random.Random) -> str:
    """A pronounceable nonsense word ("zorbel", "kafi"): cannot be memorised."""
    n = rng.randint(2, 4)
    word = "".join(rng.choice(_CONSONANTS) + rng.choice(_VOWELS) for _ in range(n))
    if rng.random() < 0.3:
        word += rng.choice(_CONSONANTS)
    return word


def _case(word: str, rng: random.Random) -> str:
    r = rng.random()
    return word.upper() if r < 0.15 else word.capitalize() if r < 0.4 else word


def name(rng: random.Random) -> str:
    return f"{_case(rng.choice(_FIRST), rng)} {_case(rng.choice(_LAST), rng)}"


def number(rng: random.Random) -> str:
    kind = rng.randrange(3)
    if kind == 0:
        return str(rng.randint(0, 9999))
    if kind == 1:
        return f"{rng.randint(2000, 2030)}-{rng.randint(1, 12):02d}-{rng.randint(1, 28):02d}"
    return f"v{rng.randint(0, 9)}.{rng.randint(0, 20)}"


def literal(rng: random.Random) -> str:
    """A value to *add* to the text (ate, atb, apl, pln, ins)."""
    kind = rng.choices(("word", "pseudo", "name", "phrase", "symbol", "number"),
                       weights=(30, 20, 15, 10, 15, 10))[0]
    value = {
        "word": lambda: _case(rng.choice(_WORDS), rng),
        "pseudo": lambda: _case(pseudo_word(rng), rng),
        "name": lambda: name(rng),
        "phrase": lambda: _case(rng.choice(_PHRASES), rng),
        "symbol": lambda: rng.choice(_SYMBOLS),
        "number": lambda: number(rng),
    }[kind]()
    if kind in ("word", "pseudo", "name", "phrase", "number") and rng.random() < 0.2:
        value = rng.choice((" ", ": ", "- ", "[", "(")) + value if rng.random() < 0.5 \
            else value + rng.choice((" ", "!", ".", ":", "]", ")"))
    return value


def replacement(rng: random.Random) -> str:
    """A value that *replaces* a pattern (no regex-special characters)."""
    kind = rng.choices(("word", "pseudo", "name", "number"), weights=(55, 25, 12, 8))[0]
    if kind == "word":
        return _case(rng.choice(_WORDS), rng)
    if kind == "pseudo":
        return _case(pseudo_word(rng), rng)
    if kind == "name":
        return name(rng)
    return str(rng.randint(0, 999))


def _safe_pattern(word: str) -> bool:
    # patterns are regexes and must not hide inside (or contain) probe words,
    # otherwise "replace every X" would also change other words
    return word.isalpha() and word.isascii() and len(word) >= 3 and not any(
        word in v or v in word for v in VOCAB if len(v) > 2
    )


PATTERN_WORDS: tuple[str, ...] = tuple(w for w in _WORDS if _safe_pattern(w))


def pattern(rng: random.Random) -> str:
    """A word to search for; it is injected into the probes by the generator."""
    if rng.random() < 0.7:
        return rng.choice(PATTERN_WORDS)
    for _ in range(20):
        word = pseudo_word(rng)
        if _safe_pattern(word):
            return word
    return rng.choice(PATTERN_WORDS)
