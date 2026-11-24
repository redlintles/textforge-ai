"""Synthetic English input texts ("probes") used to execute and check pipelines.

The words are chosen so that no word is a substring of another word: TextForge
patterns are regular expressions, and ``cat`` silently matching inside
``category`` would make the generated descriptions lie.
"""

from __future__ import annotations

import random

VOCAB: tuple[str, ...] = (
    "apple", "banana", "river", "mountain", "forest", "garden", "window", "bridge",
    "silver", "orange", "purple", "yellow", "quick", "lazy", "happy", "brave",
    "dog", "fox", "horse", "rabbit", "tiger", "eagle", "salmon", "turtle",
    "runs", "jumps", "sleeps", "sings", "reads", "writes", "builds", "paints",
    "morning", "evening", "winter", "summer", "city", "village", "ocean", "desert",
    "book", "letter", "song", "story", "lamp", "chair", "table", "clock",
    "the", "a", "and", "with", "near", "under", "over", "from",
    "today", "always", "never", "slowly", "loudly", "softly", "again", "here",
)

PUNCTUATION = ("", "", "", ".", "!", "?", ",")


def _line(rng: random.Random, required: list[str]) -> str:
    words = [rng.choice(VOCAB) for _ in range(rng.randint(3, 9))]
    for word in required:
        if rng.random() < 0.6:
            words.insert(rng.randrange(len(words) + 1), word)
    if rng.random() < 0.3:
        words[0] = words[0].capitalize()
    return " ".join(words) + rng.choice(PUNCTUATION)


def make_text(rng: random.Random, required: list[str] | None = None) -> str:
    """One probe text: 1-5 lines with realistic messiness (blank lines, dupes, padding)."""
    required = list(required or [])
    lines = [_line(rng, required) for _ in range(rng.randint(1, 5))]
    # Guarantee every required word shows up at least twice somewhere.
    for word in required:
        while " ".join(lines).split().count(word) < 2:
            index = rng.randrange(len(lines))
            parts = lines[index].split(" ")
            parts.insert(rng.randrange(len(parts) + 1), word)
            lines[index] = " ".join(parts)
    if len(lines) > 1 and rng.random() < 0.3:
        lines.insert(rng.randrange(1, len(lines)), "")
    if rng.random() < 0.25:
        lines.append(rng.choice(lines))
    text = "\n".join(lines)
    if rng.random() < 0.25:
        text = " " * rng.randint(1, 3) + text
    if rng.random() < 0.25:
        text = text + " " * rng.randint(1, 3)
    if rng.random() < 0.5:
        text += "\n"
    return text
