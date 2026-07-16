"""Experiment A: which image processing method yields the best OCR accuracy?

Trains one CRNN (same architecture/seed/hyperparameters) per method in the
registry -- train *and* test images both go through that method's
processing, and every method shares the exact same train/val/test split
(``dataset.build_split``). Writes one summary/history/predictions folder per
method plus a single ranked ``comparison.csv``.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from pathlib import Path

import pandas as pd
import torch

from image_processing_study import ocr_train
from image_processing_study.methods import build_registry

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT_DIR = REPO_ROOT / "outputs" / "image_processing_study" / "experiment_a"
DEFAULT_RL_CHECKPOINT = REPO_ROOT / "outputs" / "rl_deblur" / "checkpoints" / "best_deblur_agent.pt"


def run_sweep(args: argparse.Namespace) -> pd.DataFrame:
    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    methods = build_registry(include_rl=args.include_rl, rl_checkpoint=args.rl_checkpoint, device=str(device))
    if args.methods:
        wanted = set(args.methods)
        methods = [m for m in methods if m.name in wanted]

    cfg = ocr_train.OCRTrainConfig(
        output_dir=str(output_dir),
        epochs=args.epochs,
        patience=args.patience,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        lr=args.lr,
        seed=args.seed,
        split_seed=args.split_seed,
        limit_train=args.limit_train,
        limit_val=args.limit_val,
        limit_test=args.limit_test,
    )

    rows = []
    for method in methods:
        print(f"=== Training method: {method.name} ({method.chapter}) ===")
        summary = ocr_train.fit(method, cfg, device=device)
        rows.append(
            {
                "method": summary["method"],
                "chapter": summary["chapter"],
                "description": summary["description"],
                "num_params": summary["num_params"],
                "best_epoch": summary["best_epoch"],
                "epochs_run": summary["epochs_run"],
                "early_stopped": summary["early_stopped"],
                "best_val_exact": summary["best_val_exact"],
                "test_exact_acc": summary["test_metrics"]["exact_acc"],
                "test_cer": summary["test_metrics"]["cer"],
                "test_char_acc": summary["test_metrics"]["char_acc"],
                "test_samples": summary["test_metrics"]["samples"],
            }
        )

    results = pd.DataFrame(rows).sort_values(["test_exact_acc", "test_char_acc"], ascending=False).reset_index(drop=True)
    results.to_csv(output_dir / "comparison.csv", index=False)
    (output_dir / "run_config.json").write_text(json.dumps(asdict(cfg), ensure_ascii=False, indent=2), encoding="utf-8")
    return results


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR))
    parser.add_argument("--methods", nargs="*", default=None, help="Subset of method names to run (default: all).")
    parser.add_argument("--include-rl", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--rl-checkpoint", default=str(DEFAULT_RL_CHECKPOINT))
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--patience", type=int, default=10, help="Stop early after this many epochs with no val_exact_acc improvement.")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--split-seed", type=int, default=ocr_train.SPLIT_SEED)
    parser.add_argument("--limit-train", type=int, default=None)
    parser.add_argument("--limit-val", type=int, default=None)
    parser.add_argument("--limit-test", type=int, default=None)
    parser.add_argument("--device", default="")
    return parser.parse_args()


if __name__ == "__main__":
    results_df = run_sweep(parse_args())
    print(results_df[["method", "chapter", "test_exact_acc", "test_cer", "best_val_exact"]])
