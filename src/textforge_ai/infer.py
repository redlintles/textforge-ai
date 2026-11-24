"""Inference: generate pipelines, validate them, pick the best.

``Forge.solve`` implements the loop described in docs/01-abordagem.md:

1. **generate** N candidate pipelines (beam search);
2. **execute** each on the user's text with the real engine (errors are discarded);
3. **validate**:
   - with an expected output: exact comparison (deterministic, no model involved);
   - without it: the model's own *verify* task estimates P(match).
"""

from __future__ import annotations

from dataclasses import dataclass, field

from . import guards
from .engine import canonicalize, run_pipeline_text
from .model import MAX_TARGET_TOKENS, encode_batch, load, pick_device
from .tasks import generate_source, verify_source
from .verify_data import MATCH, MISMATCH


@dataclass
class Verdict:
    label: str  # "match" / "mismatch"
    confidence: float  # model's probability for `label` (0.5 .. 1.0)
    detail: str  # e.g. "mismatch: step 2 (rfw)"
    method: str  # "expected-output" (exact) or "model" (estimate)


@dataclass
class Candidate:
    pipeline: str
    ok: bool
    output: str | None = None
    error: dict | None = None
    verdict: Verdict | None = None
    suspicious: bool = False


@dataclass
class SolveResult:
    prompt: str
    best: Candidate | None
    candidates: list[Candidate] = field(default_factory=list)


class Forge:
    def __init__(self, tokenizer, model, device="auto"):
        self.tokenizer = tokenizer
        self.device = pick_device(device) if isinstance(device, str) else device
        self.model = model.to(self.device).eval()
        # Tokenize labels exactly like training targets (CodeT5 prepends <s>), minus the
        # final </s>, so "mismatch: step 2" still counts as the "mismatch" prefix.
        self._label_ids = {}
        for label in (MATCH, MISMATCH):
            ids = tokenizer(text_target=label)["input_ids"]
            if ids and ids[-1] == tokenizer.eos_token_id:
                ids = ids[:-1]
            self._label_ids[label] = ids

    @classmethod
    def from_dir(cls, path: str, device: str = "auto") -> Forge:
        tokenizer, model = load(path)
        return cls(tokenizer, model, device)

    # -- generation --------------------------------------------------------------
    def generate(self, prompt: str, n: int = 4) -> list[str]:
        """Up to ``n`` distinct, syntactically valid pipelines, best first."""
        import torch

        batch = encode_batch(self.tokenizer, [generate_source(prompt)], device=self.device)
        beams = max(n, 4)
        with torch.no_grad():
            output = self.model.generate(**batch, max_new_tokens=MAX_TARGET_TOKENS,
                                         num_beams=beams, num_return_sequences=beams)
        seen, result = set(), []
        for text in self.tokenizer.batch_decode(output, skip_special_tokens=True):
            canonical = canonicalize(text.strip())
            if canonical and canonical not in seen:
                seen.add(canonical)
                result.append(canonical)
        return result[:n]

    def generate_batch(self, prompts: list[str]) -> list[str]:
        """Greedy, one pipeline per prompt (fast; used for evaluation)."""
        import torch

        batch = encode_batch(self.tokenizer, [generate_source(p) for p in prompts],
                             device=self.device)
        with torch.no_grad():
            output = self.model.generate(**batch, max_new_tokens=MAX_TARGET_TOKENS, num_beams=1)
        return [t.strip() for t in self.tokenizer.batch_decode(output, skip_special_tokens=True)]

    # -- verification ------------------------------------------------------------
    def _label_logprobs(self, source: str) -> dict[str, float]:
        import torch

        batch = encode_batch(self.tokenizer, [source], device=self.device)
        scores = {}
        for label, ids in self._label_ids.items():
            labels = torch.tensor([ids], device=self.device)
            with torch.no_grad():
                logits = self.model(**batch, labels=labels).logits
            logp = torch.log_softmax(logits, dim=-1)[0, torch.arange(len(ids)), labels[0]]
            scores[label] = float(logp.sum())
        return scores

    def verify(self, prompt: str, pipeline: str, input_text: str, output_text: str) -> Verdict:
        import math

        import torch

        source = verify_source(prompt, pipeline, input_text, output_text)
        scores = self._label_logprobs(source)
        top = max(scores.values())
        probs = {k: math.exp(v - top) for k, v in scores.items()}
        total = sum(probs.values())
        label = max(probs, key=probs.get)
        detail = label
        if label == MISMATCH:
            batch = encode_batch(self.tokenizer, [source], device=self.device)
            with torch.no_grad():
                out = self.model.generate(**batch, max_new_tokens=24, num_beams=1)
            detail = self.tokenizer.decode(out[0], skip_special_tokens=True).strip() or label
        return Verdict(label, probs[label] / total, detail, "model")

    def verify_batch_labels(self, sources: list[str]) -> list[str]:
        """Greedy-decoded verify outputs (used for evaluation)."""
        import torch

        batch = encode_batch(self.tokenizer, sources, device=self.device)
        with torch.no_grad():
            out = self.model.generate(**batch, max_new_tokens=24, num_beams=1)
        return [t.strip() for t in self.tokenizer.batch_decode(out, skip_special_tokens=True)]

    # -- the full loop -------------------------------------------------------------
    def solve(self, prompt: str, text: str, expected: str | None = None, n: int = 4) -> SolveResult:
        candidates = []
        for pipeline in self.generate(prompt, n):
            (run,) = run_pipeline_text(pipeline, [text])
            cand = Candidate(pipeline, run.ok, run.output,
                             run.error.to_dict() if run.error else None)
            if run.ok:
                if expected is not None:
                    hit = run.output == expected
                    cand.verdict = Verdict(MATCH if hit else MISMATCH, 1.0,
                                           "output equals expected" if hit
                                           else "output differs from expected", "expected-output")
                else:
                    cand.verdict = self.verify(prompt, pipeline, text, run.output)
                refs = [expected] if expected is not None else []
                cand.suspicious = bool(guards.leaked_literals(pipeline, [text, *refs], prompt))
            candidates.append(cand)

        def rank(c: Candidate) -> float:
            if not c.ok or c.verdict is None or c.suspicious:
                return -1.0
            p = c.verdict.confidence
            return p if c.verdict.label == MATCH else 1.0 - p

        best = max(candidates, key=rank, default=None)
        if best is not None and rank(best) < 0:
            best = None
        return SolveResult(prompt, best, candidates)
