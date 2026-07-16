"""Experiment B: classical restoration quality on synthetically degraded plates.

Complements Experiment A (which measures whether a processing method helps a
*freshly trained* model) with the textbook Ch.3/Ch.8 restoration story: given
a known degradation (blur kernel or noise level, see ``degrade.py``), which
method best recovers the original pixels (PSNR/SSIM) and best rescues OCR
accuracy of the frozen ``raw``-trained model from Experiment A?

Lower/upper bounds included for sanity: ``degraded`` (no restoration) should
score worst, ``clean`` (ground truth passed through untouched) should score
best on every metric -- if it doesn't, something in the harness is broken.
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from skimage.metrics import structural_similarity as ssim_fn
from skimage.metrics import peak_signal_noise_ratio as psnr_fn

from image_processing_study import ocr_train
from image_processing_study.common import edit_distance, normalize_plate_text
from image_processing_study.degrade import build_degraded_records
from image_processing_study.methods import gaussian_psf, get_core_method, try_build_rl_deblur_method, wiener_deconvolve
from image_processing_study.model import ctc_greedy_decode

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT_DIR = REPO_ROOT / "outputs" / "image_processing_study" / "experiment_b"
DEFAULT_RAW_CHECKPOINT = REPO_ROOT / "outputs" / "image_processing_study" / "experiment_a" / "raw" / "best_model.pt"
DEFAULT_RL_CHECKPOINT = REPO_ROOT / "outputs" / "rl_deblur" / "checkpoints" / "best_deblur_agent.pt"

RESTORATION_METHOD_NAMES = ["median_denoise", "bilateral_denoise", "wavelet_denoise", "homomorphic", "wiener_restore"]


def arrays_to_tensor_batch(arrays: list[np.ndarray], device: torch.device) -> torch.Tensor:
    stacked = np.stack(arrays).astype(np.float32) / 255.0
    tensor = torch.from_numpy(stacked)
    tensor = (tensor - 0.5) / 0.5
    return tensor.unsqueeze(1).to(device)  # (B, 1, H, W)


@torch.no_grad()
def ocr_predict(model, arrays: list[np.ndarray], device: torch.device) -> list[str]:
    if model is None:
        return [""] * len(arrays)
    images = arrays_to_tensor_batch(arrays, device)
    log_probs = model(images)
    preds, _confs = ctc_greedy_decode(log_probs)
    return [normalize_plate_text(p) for p in preds]


def wiener_restore_true_psf(degraded: np.ndarray, psf: np.ndarray | None) -> np.ndarray:
    """Textbook Wiener/MMSE restoration (Ch.3.1/3.6): deconvolve with the
    *actual* degradation kernel used to synthesize ``degraded``. Falls back
    to the same assumed-PSF operator as Experiment A when the degradation
    had no blur kernel (``gaussian_noise``), since there is nothing to
    deconvolve in that case.
    """
    if psf is None:
        psf = gaussian_psf(size=5, sigma=1.0)
    return wiener_deconvolve(degraded, psf, balance=0.1)


def run_evaluation(args: argparse.Namespace) -> pd.DataFrame:
    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    records = build_degraded_records(seed=args.degrade_seed, split_seed=args.split_seed, limit=args.limit)
    logger.info("Built %d degraded test records.", len(records))

    ocr_model = None
    raw_ckpt = Path(args.raw_checkpoint)
    if raw_ckpt.exists():
        ocr_model, _ckpt = ocr_train.load_checkpoint(raw_ckpt, device=device)
        ocr_model.eval()
    else:
        logger.warning("raw-trained checkpoint not found at %s -- OCR columns will be NaN.", raw_ckpt)

    restoration_methods = [get_core_method(name) for name in RESTORATION_METHOD_NAMES]
    rl_method = try_build_rl_deblur_method(args.rl_checkpoint, device=str(device)) if args.include_rl else None

    clean_arrays = [r["clean"] for r in records]
    degraded_arrays = [r["degraded"] for r in records]
    labels = [normalize_plate_text(r["label"]) for r in records]

    rows = []

    def score(name: str, chapter: str, restored_arrays: list[np.ndarray]) -> None:
        psnrs = [psnr_fn(c.astype(np.float64), r.astype(np.float64), data_range=255) for c, r in zip(clean_arrays, restored_arrays)]
        ssims = [ssim_fn(c, r, data_range=255) for c, r in zip(clean_arrays, restored_arrays)]
        preds = ocr_predict(ocr_model, restored_arrays, device)
        exact = [p == t for p, t in zip(preds, labels)]
        cers = [edit_distance(p, t) / max(len(t), 1) for p, t in zip(preds, labels)]
        rows.append(
            {
                "method": name,
                "chapter": chapter,
                "psnr_mean": float(np.mean(psnrs)),
                "ssim_mean": float(np.mean(ssims)),
                "exact_acc": float(np.mean(exact)) if ocr_model is not None else float("nan"),
                "cer": float(np.mean(cers)) if ocr_model is not None else float("nan"),
                "samples": len(records),
            }
        )

    score("degraded", "lower bound (no restoration)", degraded_arrays)
    score("clean", "upper bound (ground truth)", clean_arrays)

    for method in restoration_methods:
        if method.name == "wiener_restore":
            restored = [wiener_restore_true_psf(r["degraded"], r["psf"]) for r in records]
            score("wiener_restore_true_psf", method.chapter, restored)
        else:
            restored = [method.process(arr) for arr in degraded_arrays]
            score(method.name, method.chapter, restored)

    if rl_method is not None:
        restored = [rl_method.process(arr) for arr in degraded_arrays]
        score(rl_method.name, rl_method.chapter, restored)

    results = pd.DataFrame(rows).sort_values(["psnr_mean"], ascending=False).reset_index(drop=True)
    results.to_csv(output_dir / "comparison.csv", index=False)

    by_kind = (
        pd.DataFrame({"kind": [r["kind"] for r in records]})
        .value_counts()
        .rename("count")
        .reset_index()
    )
    by_kind.to_csv(output_dir / "degradation_kind_counts.csv", index=False)

    summary = {
        "num_records": len(records),
        "ocr_model_used": str(raw_ckpt) if ocr_model is not None else None,
        "restoration_methods": RESTORATION_METHOD_NAMES,
        "include_rl": rl_method is not None,
    }
    (output_dir / "run_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    return results


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR))
    parser.add_argument("--raw-checkpoint", default=str(DEFAULT_RAW_CHECKPOINT))
    parser.add_argument("--include-rl", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--rl-checkpoint", default=str(DEFAULT_RL_CHECKPOINT))
    parser.add_argument("--degrade-seed", type=int, default=123)
    parser.add_argument("--split-seed", type=int, default=ocr_train.SPLIT_SEED)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--device", default="")
    return parser.parse_args()


if __name__ == "__main__":
    results_df = run_evaluation(parse_args())
    print(results_df)
