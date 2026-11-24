"""Command line: ``python -m textforge_ai <command>``.

Commands
--------
gen-data   generate a synthetic dataset (train/val/test JSONL)
baseline   score a reference predictor (oracle / identity / cheater)
eval       score a predictions JSONL file ({"id": ..., "pipeline": ...} per line)
run        apply a .tft pipeline to a text file (or stdin)
train      fine-tune the model (generate + verify)            [needs .[train]]
model-eval score a trained model on a split                   [needs .[train]]
solve      prompt + text -> best pipeline, output and verdict  [needs .[train]]
verify     does a given .tft pipeline do what the prompt asks? [needs .[train]]
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from . import dataset
from .baselines import BASELINES
from .engine import run_pipeline_text
from .evaluate import evaluate, evaluate_predictor


def _gen_data(args: argparse.Namespace) -> int:
    examples = dataset.generate(args.n, seed=args.seed, n_probes=args.probes)
    splits = dataset.split_by_pipeline(examples, seed=args.seed)
    out = Path(args.out)
    for name, rows in splits.items():
        count = dataset.write_jsonl(out / f"{name}.jsonl", rows)
        print(f"{name}: {count} examples -> {out / f'{name}.jsonl'}")
    return 0


def _print_report(summary: dict, records: list[dict], details: str | None) -> None:
    print(json.dumps(summary, indent=2))
    if details:
        dataset.write_jsonl(details, records)
        print(f"per-example details -> {details}")


def _baseline(args: argparse.Namespace) -> int:
    examples = list(dataset.read_jsonl(args.data))[: args.limit]
    summary, records = evaluate_predictor(examples, BASELINES[args.kind])
    _print_report(summary, records, args.details)
    return 0


def _eval(args: argparse.Namespace) -> int:
    examples = list(dataset.read_jsonl(args.data))
    predictions = {row["id"]: row["pipeline"] for row in dataset.read_jsonl(args.pred)}
    summary, records = evaluate(examples, predictions)
    _print_report(summary, records, args.details)
    return 0


def _run(args: argparse.Namespace) -> int:
    pipeline = Path(args.pipeline).read_text(encoding="utf-8")
    text = Path(args.input).read_text(encoding="utf-8") if args.input else sys.stdin.read()
    (result,) = run_pipeline_text(pipeline, [text])
    if not result.ok:
        print(json.dumps(result.error.to_dict(), indent=2), file=sys.stderr)
        return 1
    sys.stdout.write(result.output)
    return 0


def _read_prompt(args: argparse.Namespace) -> str:
    if getattr(args, "prompt_file", None):
        return Path(args.prompt_file).read_text(encoding="utf-8").strip()
    if args.prompt:
        return args.prompt
    return input("prompt> ").strip()


def _train(args: argparse.Namespace) -> int:
    from .train import TrainConfig, train

    fields = TrainConfig.__dataclass_fields__
    cfg = TrainConfig(**{k: v for k, v in vars(args).items() if k in fields and v is not None})
    result = train(cfg)
    print(json.dumps({"best_gen_probe_accuracy": result["best_gen_probe_accuracy"]}))
    return 0


def _model_eval(args: argparse.Namespace) -> int:
    from .infer import Forge
    from .train import evaluate_model

    forge = Forge.from_dir(args.model, args.device)
    examples = list(dataset.read_jsonl(args.data))[: args.limit]
    print(json.dumps(evaluate_model(forge, examples), indent=2))
    return 0


def _verdict_dict(verdict) -> dict | None:
    return None if verdict is None else {
        "label": verdict.label, "confidence": round(verdict.confidence, 3),
        "detail": verdict.detail, "method": verdict.method,
    }


def _solve(args: argparse.Namespace) -> int:
    from .infer import Forge

    forge = Forge.from_dir(args.model, args.device)
    prompt = _read_prompt(args)
    text = Path(args.input).read_text(encoding="utf-8")
    expected = Path(args.expected).read_text(encoding="utf-8") if args.expected else None
    result = forge.solve(prompt, text, expected, n=args.candidates)
    for c in result.candidates:
        print(json.dumps({"pipeline": c.pipeline, "ok": c.ok, "error": c.error,
                          "verdict": _verdict_dict(c.verdict), "suspicious": c.suspicious}),
              file=sys.stderr)
    if result.best is None:
        print("no candidate passed validation", file=sys.stderr)
        return 2
    if args.save_pipeline:
        Path(args.save_pipeline).write_text(result.best.pipeline + "\n", encoding="utf-8")
    if args.save_output:
        Path(args.save_output).write_text(result.best.output, encoding="utf-8")
    print("## pipeline\n" + result.best.pipeline)
    print("## verdict\n" + json.dumps(_verdict_dict(result.best.verdict)))
    print("## output\n" + result.best.output)
    return 0


def _verify(args: argparse.Namespace) -> int:
    from .infer import Forge

    prompt = _read_prompt(args)
    pipeline = Path(args.pipeline).read_text(encoding="utf-8").strip()
    text = Path(args.input).read_text(encoding="utf-8")
    (run,) = run_pipeline_text(pipeline, [text])
    if not run.ok:
        print(json.dumps({"label": "error", "error": run.error.to_dict()}, indent=2))
        return 1
    if args.expected:
        hit = run.output == Path(args.expected).read_text(encoding="utf-8")
        print(json.dumps({"label": "match" if hit else "mismatch", "confidence": 1.0,
                          "method": "expected-output"}, indent=2))
        return 0 if hit else 3
    verdict = Forge.from_dir(args.model, args.device).verify(prompt, pipeline, text, run.output)
    print(json.dumps(_verdict_dict(verdict), indent=2))
    return 0 if verdict.label == "match" else 3


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="textforge_ai", description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="command", required=True)

    p = sub.add_parser("gen-data", help="generate a synthetic dataset")
    p.add_argument("--n", type=int, default=20000)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--probes", type=int, default=6)
    p.add_argument("--out", default="data")
    p.set_defaults(func=_gen_data)

    p = sub.add_parser("baseline", help="score a reference predictor")
    p.add_argument("kind", choices=sorted(BASELINES))
    p.add_argument("--data", required=True)
    p.add_argument("--limit", type=int, default=None)
    p.add_argument("--details", default=None)
    p.set_defaults(func=_baseline)

    p = sub.add_parser("eval", help="score a predictions file")
    p.add_argument("--data", required=True)
    p.add_argument("--pred", required=True)
    p.add_argument("--details", default=None)
    p.set_defaults(func=_eval)

    p = sub.add_parser("run", help="apply a .tft pipeline to a text")
    p.add_argument("pipeline")
    p.add_argument("input", nargs="?")
    p.set_defaults(func=_run)

    p = sub.add_parser("train", help="fine-tune the model")
    p.add_argument("--data", default="data")
    p.add_argument("--out", default=None)
    p.add_argument("--model", default=None, help="HF name or checkpoint dir to start from")
    p.add_argument("--tiny", action="store_true", help="offline smoke test with a tiny model")
    p.add_argument("--epochs", type=int)
    p.add_argument("--batch-size", dest="batch_size", type=int)
    p.add_argument("--lr", type=float)
    p.add_argument("--max-train", dest="max_train", type=int)
    p.add_argument("--eval-examples", dest="eval_examples", type=int)
    p.add_argument("--log-every", dest="log_every", type=int)
    p.add_argument("--threads", type=int)
    p.add_argument("--device", default=None)
    p.add_argument("--no-verify", dest="no_verify", action="store_true", default=None)
    p.set_defaults(func=_train)

    p = sub.add_parser("model-eval", help="score a trained model")
    p.add_argument("--model", required=True)
    p.add_argument("--data", required=True)
    p.add_argument("--limit", type=int, default=None)
    p.add_argument("--device", default="auto")
    p.set_defaults(func=_model_eval)

    for name, func in (("solve", _solve), ("verify", _verify)):
        p = sub.add_parser(name)
        p.add_argument("--model", required=True)
        p.add_argument("--input", required=True, help="text file to transform")
        p.add_argument("--prompt", help="instruction (asked interactively if omitted)")
        p.add_argument("--prompt-file", dest="prompt_file")
        p.add_argument("--expected", help="file with the expected output (exact validation)")
        p.add_argument("--device", default="auto")
        if name == "solve":
            p.add_argument("--candidates", type=int, default=4)
            p.add_argument("--save-pipeline", dest="save_pipeline")
            p.add_argument("--save-output", dest="save_output")
        else:
            p.add_argument("--pipeline", required=True, help=".tft file to verify")
        p.set_defaults(func=func)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    return args.func(args)
