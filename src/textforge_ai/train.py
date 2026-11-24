"""Multitask fine-tuning (generate + verify) with a plain PyTorch loop.

A hand-written loop instead of ``transformers.Trainer`` on purpose: every step of
training is visible here (batch -> loss -> backward -> optimizer step), which is the
point for a first ML project.

Each epoch ends with a real evaluation on the validation split:
* ``gen_probe_accuracy`` - generated pipelines, scored by evaluate.py (the main metric);
* ``verify_accuracy``    - match/mismatch decisions on execution-labelled examples.
The best checkpoint (by gen_probe_accuracy) is kept in ``<out>/best``.

Resuming: every ``save_every`` steps (and on Ctrl+C / ``kill`` / shutdown) the full
training state is written to ``<out>/checkpoint``: weights, optimizer, LR scheduler,
RNG state and position in the epoch. ``--resume`` continues from exactly that step, so
an interruption costs at most ``save_every`` steps.
"""

from __future__ import annotations

import json
import math
import os
import random
import shutil
import signal
import time
from dataclasses import asdict, dataclass
from pathlib import Path

from . import dataset
from .evaluate import evaluate
from .model import DEFAULT_MODEL, create_tiny, encode_batch, load, pick_device
from .tasks import training_pairs, verify_pairs
from .verify_data import MATCH, MISMATCH


@dataclass
class TrainConfig:
    data: str = "data"
    out: str = "runs/codet5-small"
    model: str = DEFAULT_MODEL
    tiny: bool = False  # offline smoke-test model, see model.create_tiny
    epochs: int = 3
    batch_size: int = 16
    lr: float = 3e-4
    warmup: float = 0.05
    max_train: int | None = None  # use only the first N training examples
    eval_examples: int = 300
    log_every: int = 50
    threads: int = 0  # 0 = all CPU cores
    device: str = "auto"
    seed: int = 0
    no_verify: bool = False  # train the generate task only
    save_every: int = 200  # steps between resumable checkpoints
    resume: bool = False  # continue from <out>/checkpoint
    max_steps: int | None = None  # stop (with a checkpoint) after this many total steps


CHECKPOINT = "checkpoint"
#: settings that define the run; on --resume they are taken from the checkpoint
_RUN_KEYS = ("data", "model", "tiny", "epochs", "batch_size", "lr", "warmup", "max_train",
             "eval_examples", "seed", "no_verify")


def _verify_label(text: str) -> str:
    return MISMATCH if text.startswith(MISMATCH) else MATCH if text.startswith(MATCH) else "?"


def evaluate_model(forge, val_examples: list[dict], batch_size: int = 32) -> dict:
    predictions = {}
    for i in range(0, len(val_examples), batch_size):
        chunk = val_examples[i : i + batch_size]
        for example, pipeline in zip(chunk, forge.generate_batch([e["prompt"] for e in chunk])):
            predictions[example["id"]] = pipeline
    summary, _ = evaluate(val_examples, predictions)

    pairs = verify_pairs(val_examples, seed=1)
    hits = 0
    for i in range(0, len(pairs), batch_size):
        chunk = pairs[i : i + batch_size]
        outputs = forge.verify_batch_labels([s for s, _ in chunk])
        hits += sum(_verify_label(o) == _verify_label(t) for o, (_, t) in zip(outputs, chunk))
    return {
        "gen_probe_accuracy": summary["probe_accuracy"],
        "gen_exact_match": summary["exact_match"],
        "gen_valid_rate": summary["valid_rate"],
        "verify_accuracy": hits / max(len(pairs), 1),
    }


def _save(model, tokenizer, path: Path, info: dict) -> None:
    path.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(path)
    tokenizer.save_pretrained(path)
    (path / "train_info.json").write_text(json.dumps(info, indent=2), encoding="utf-8")


def _save_checkpoint(out: Path, model, tokenizer, optimizer, scheduler, state: dict) -> None:
    """Write atomically: a crash while saving never destroys the previous checkpoint."""
    import torch

    final, tmp, old = out / CHECKPOINT, out / f"{CHECKPOINT}.tmp", out / f"{CHECKPOINT}.old"
    if tmp.exists():
        shutil.rmtree(tmp)
    _save(model, tokenizer, tmp, {"config": state["config"], "history": state["history"]})
    torch.save({**state, "optimizer": optimizer.state_dict(), "scheduler": scheduler.state_dict(),
                "torch_rng": torch.get_rng_state()}, tmp / "trainer_state.pt")
    if final.exists():
        if old.exists():
            shutil.rmtree(old)
        final.rename(old)
    tmp.rename(final)
    if old.exists():
        shutil.rmtree(old)


class _StopFlag:
    """Turn Ctrl+C / SIGTERM into 'finish this step, checkpoint, exit' (2nd Ctrl+C forces)."""

    def __init__(self):
        self.requested = False
        self._previous = {}

    def _handler(self, signum, frame):
        if self.requested:
            raise KeyboardInterrupt
        self.requested = True
        print(f"\nsignal {signum}: finishing current step and saving checkpoint "
              "(press Ctrl+C again to force)", flush=True)

    def __enter__(self):
        for sig in (signal.SIGINT, signal.SIGTERM):
            try:
                self._previous[sig] = signal.signal(sig, self._handler)
            except ValueError:  # not in the main thread (e.g. some test runners)
                pass
        return self

    def __exit__(self, *exc):
        for sig, handler in self._previous.items():
            signal.signal(sig, handler)


def train(cfg: TrainConfig) -> dict:
    import torch
    from transformers import get_linear_schedule_with_warmup

    from .infer import Forge

    out = Path(cfg.out)
    state = None
    if cfg.resume:
        path = out / CHECKPOINT / "trainer_state.pt"
        if not path.exists():
            raise FileNotFoundError(f"no checkpoint to resume at {path}")
        state = torch.load(path, weights_only=False)
        saved = state["config"]
        cfg = TrainConfig(**{**asdict(cfg), **{k: saved[k] for k in _RUN_KEYS}, "resume": True})

    random.seed(cfg.seed)
    torch.manual_seed(cfg.seed)
    torch.set_num_threads(cfg.threads or os.cpu_count() or 1)
    device = pick_device(cfg.device)

    train_examples = list(dataset.read_jsonl(Path(cfg.data) / "train.jsonl"))[: cfg.max_train]
    val_examples = list(dataset.read_jsonl(Path(cfg.data) / "val.jsonl"))[: cfg.eval_examples]
    pairs = training_pairs(train_examples, seed=cfg.seed, verify=not cfg.no_verify)

    if state is not None:
        tokenizer, model = load(str(out / CHECKPOINT))
    elif cfg.tiny:
        tokenizer, model = create_tiny([s for s, _ in pairs] + [t for _, t in pairs])
    else:
        tokenizer, model = load(cfg.model)
    model.to(device)

    steps_per_epoch = math.ceil(len(pairs) / cfg.batch_size)
    total_steps = steps_per_epoch * cfg.epochs
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.lr)
    scheduler = get_linear_schedule_with_warmup(
        optimizer, int(total_steps * cfg.warmup), total_steps
    )

    epoch0, skip, step, history, best, elapsed0 = 1, 0, 0, [], -1.0, 0.0
    if state is not None:
        optimizer.load_state_dict(state["optimizer"])
        scheduler.load_state_dict(state["scheduler"])
        torch.set_rng_state(state["torch_rng"])
        epoch0, skip, step = state["epoch"], state["batch_in_epoch"], state["step"]
        history, best, elapsed0 = state["history"], state["best"], state["elapsed"]
        print(f"resuming from step {step}/{total_steps} (epoch {epoch0}, batch {skip})",
              flush=True)
    print(f"device={device} threads={torch.get_num_threads()} pairs={len(pairs)} "
          f"steps/epoch={steps_per_epoch} total_steps={total_steps}", flush=True)

    session_start, session_step0 = time.time(), step

    def elapsed() -> float:
        return elapsed0 + time.time() - session_start

    def checkpoint(epoch: int, batch_in_epoch: int) -> None:
        _save_checkpoint(out, model, tokenizer, optimizer, scheduler, {
            "config": asdict(cfg), "epoch": epoch, "batch_in_epoch": batch_in_epoch,
            "step": step, "history": history, "best": best, "elapsed": elapsed(),
        })

    with _StopFlag() as stop:
        for epoch in range(epoch0, cfg.epochs + 1):
            model.train()
            order = list(pairs)  # fresh shuffle per epoch: reproducible after a resume
            random.Random(cfg.seed + epoch).shuffle(order)
            running, counted = 0.0, 0
            first = skip if epoch == epoch0 else 0
            for b in range(first, steps_per_epoch):
                chunk = order[b * cfg.batch_size : (b + 1) * cfg.batch_size]
                batch = encode_batch(tokenizer, [s for s, _ in chunk], [t for _, t in chunk],
                                     device=device)
                loss = model(**batch).loss
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()
                step += 1
                running += loss.item()
                counted += 1
                if step % cfg.log_every == 0 or step == total_steps:
                    rate = (time.time() - session_start) / max(step - session_step0, 1)
                    print(f"epoch {epoch} step {step}/{total_steps} "
                          f"loss {running / max(counted, 1):.4f} "
                          f"elapsed {elapsed() / 60:.1f}min "
                          f"eta {rate * (total_steps - step) / 60:.1f}min", flush=True)
                    running, counted = 0.0, 0
                limit = cfg.max_steps is not None and step >= cfg.max_steps
                if stop.requested or limit or step % cfg.save_every == 0:
                    checkpoint(epoch, b + 1)
                if stop.requested or limit:
                    print(f"stopped at step {step}; resume with --resume --out {out}",
                          flush=True)
                    return {"best_gen_probe_accuracy": best, "history": history,
                            "stopped": True, "step": step}

            model.eval()
            metrics = evaluate_model(Forge(tokenizer, model, device), val_examples)
            metrics |= {"epoch": epoch, "step": step, "minutes": elapsed() / 60}
            history.append(metrics)
            print("eval", json.dumps(metrics), flush=True)
            info = {"config": asdict(cfg), "history": history}
            _save(model, tokenizer, out / "last", info)
            if metrics["gen_probe_accuracy"] > best:
                best = metrics["gen_probe_accuracy"]
                _save(model, tokenizer, out / "best", info)
                print(f"new best gen_probe_accuracy={best:.3f} -> {out / 'best'}", flush=True)
            checkpoint(epoch + 1, 0)  # resuming after this point skips the finished eval
    return {"best_gen_probe_accuracy": best, "history": history, "stopped": False, "step": step}
