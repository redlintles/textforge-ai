"""Multitask fine-tuning (generate + verify) with a plain PyTorch loop.

A hand-written loop instead of ``transformers.Trainer`` on purpose: every step of
training is visible here (batch -> loss -> backward -> optimizer step), which is the
point for a first ML project.

Each epoch ends with a real evaluation on the validation split:
* ``gen_probe_accuracy`` - generated pipelines, scored by evaluate.py (the main metric);
* ``verify_accuracy``    - match/mismatch decisions on execution-labelled examples.
The best checkpoint (by gen_probe_accuracy) is kept in ``<out>/best``.
"""

from __future__ import annotations

import json
import math
import os
import random
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


def train(cfg: TrainConfig) -> dict:
    import torch
    from transformers import get_linear_schedule_with_warmup

    from .infer import Forge

    random.seed(cfg.seed)
    torch.manual_seed(cfg.seed)
    torch.set_num_threads(cfg.threads or os.cpu_count() or 1)
    device = pick_device(cfg.device)
    out = Path(cfg.out)

    train_examples = list(dataset.read_jsonl(Path(cfg.data) / "train.jsonl"))[: cfg.max_train]
    val_examples = list(dataset.read_jsonl(Path(cfg.data) / "val.jsonl"))[: cfg.eval_examples]
    pairs = training_pairs(train_examples, seed=cfg.seed, verify=not cfg.no_verify)

    if cfg.tiny:
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
    print(f"device={device} threads={torch.get_num_threads()} pairs={len(pairs)} "
          f"steps/epoch={steps_per_epoch} total_steps={total_steps}", flush=True)

    history, best, step = [], -1.0, 0
    started = time.time()
    try:
        for epoch in range(1, cfg.epochs + 1):
            model.train()
            random.Random(cfg.seed + epoch).shuffle(pairs)
            running, counted = 0.0, 0
            for i in range(0, len(pairs), cfg.batch_size):
                chunk = pairs[i : i + cfg.batch_size]
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
                    elapsed = time.time() - started
                    eta = elapsed / step * (total_steps - step)
                    print(f"epoch {epoch} step {step}/{total_steps} "
                          f"loss {running / max(counted, 1):.4f} "
                          f"elapsed {elapsed / 60:.1f}min eta {eta / 60:.1f}min", flush=True)
                    running, counted = 0.0, 0

            model.eval()
            metrics = evaluate_model(Forge(tokenizer, model, device), val_examples)
            metrics |= {"epoch": epoch, "step": step, "minutes": (time.time() - started) / 60}
            history.append(metrics)
            print("eval", json.dumps(metrics), flush=True)
            info = {"config": asdict(cfg), "history": history}
            _save(model, tokenizer, out / "last", info)
            if metrics["gen_probe_accuracy"] > best:
                best = metrics["gen_probe_accuracy"]
                _save(model, tokenizer, out / "best", info)
                print(f"new best gen_probe_accuracy={best:.3f} -> {out / 'best'}", flush=True)
    except KeyboardInterrupt:
        print("interrupted: saving to", out / "interrupted", flush=True)
        _save(model, tokenizer, out / "interrupted", {"config": asdict(cfg), "history": history})
    return {"best_gen_probe_accuracy": best, "history": history}
