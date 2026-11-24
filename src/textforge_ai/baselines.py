"""Reference predictors used to sanity-check the metrics (not real models).

* ``oracle``   - returns the reference pipeline: the metric must give ~100%.
* ``identity`` - returns an empty pipeline (text unchanged): must score ~0%.
* ``cheater``  - erases the input and writes the expected output of the first probe
  as a literal. This is the failure mode we fear; it must score ~0% and be flagged.
"""

from __future__ import annotations

import textforge as tf

from .engine import canonical_text


def oracle(example: dict) -> str:
    return example["pipeline"]


def identity(example: dict) -> str:
    return ""


def cheater(example: dict) -> str:
    answer = example["probes"][0]["output"]
    return canonical_text(tf.Pipeline().select(0, 0).delete_first().add_to_end(answer))


BASELINES = {"oracle": oracle, "identity": identity, "cheater": cheater}
