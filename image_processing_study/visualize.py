"""Report figures: per-method before/after grids and an accuracy bar chart."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
from PIL import Image, ImageDraw

from image_processing_study.dataset import build_split
from image_processing_study.degrade import build_degraded_records
from image_processing_study.methods import CORE_METHODS, Method, get_core_method, to_canvas_gray

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SAMPLE_DIR = REPO_ROOT / "outputs" / "image_processing_study" / "samples"
CELL_W, CELL_H, PAD, LABEL_H = 128, 32, 8, 14


def _grid(cells: list[tuple[str, "np.ndarray"]], cols: int) -> Image.Image:
    import numpy as np  # local import: only needed for typing/usage here

    rows = (len(cells) + cols - 1) // cols
    grid_w = cols * (CELL_W + PAD) + PAD
    grid_h = rows * (CELL_H + LABEL_H + PAD) + PAD
    canvas = Image.new("L", (grid_w, grid_h), 255)
    draw = ImageDraw.Draw(canvas)
    for i, (label, arr) in enumerate(cells):
        r, c = divmod(i, cols)
        x = PAD + c * (CELL_W + PAD)
        y = PAD + r * (CELL_H + LABEL_H + PAD)
        canvas.paste(Image.fromarray(np.asarray(arr, dtype="uint8")), (x, y))
        draw.text((x, y + CELL_H), label, fill=0)
    return canvas


def make_method_grid(
    image_path: str | Path,
    output_path: str | Path,
    cols: int = 3,
    extra_methods: list[Method] | None = None,
) -> Path:
    """All CORE_METHODS (+ optional ``extra_methods``, e.g. ``rl_deblur_restore``)
    applied to a single plate crop, side by side.
    """
    image = Image.open(image_path).convert("RGB")
    cells = []
    for method in list(CORE_METHODS) + list(extra_methods or []):
        canvas = to_canvas_gray(image, method.resample)
        cells.append((method.name, method.process(canvas)))
    grid = _grid(cells, cols)
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    grid.save(output_path)
    return output_path


def make_restoration_grid(
    output_path: str | Path,
    num_samples: int = 5,
    seed: int = 123,
    extra_methods: list[Method] | None = None,
) -> Path:
    """Rows = sample plates, columns = clean | degraded | restored(per method).

    ``extra_methods`` -- e.g. ``[try_build_rl_deblur_method(...)]`` -- is
    appended after the classical restoration methods so ``rl_deblur_restore``
    shows up here too when a checkpoint is available (it is not part of
    ``CORE_METHODS``, so it is never included unless passed explicitly).
    """
    records = build_degraded_records(seed=seed, limit=num_samples)
    method_names = ["median_denoise", "bilateral_denoise", "wavelet_denoise", "homomorphic", "wiener_restore"]
    methods_to_plot = [get_core_method(name) for name in method_names] + list(extra_methods or [])
    cols = 2 + len(methods_to_plot)
    cells = []
    for record in records:
        cells.append(("clean", record["clean"]))
        cells.append((f"degraded\n({record['kind']})", record["degraded"]))
        for method in methods_to_plot:
            cells.append((method.name, method.process(record["degraded"])))
    grid = _grid(cells, cols)
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    grid.save(output_path)
    return output_path


def make_method_preview_grid(
    output_path: str | Path,
    num_samples: int = 5,
    methods: list[str] | None = None,
    extra_methods: list[Method] | None = None,
    split: str = "train",
    seed: int = 42,
) -> Path:
    """1 row per method, ``num_samples`` different plate crops per row.

    Lets you eyeball exactly what pixels each method feeds the model with
    *before* spending time training 13 CRNNs on it -- e.g. catch a method
    that degenerates to near-blank/near-black images on this dataset.

    ``extra_methods`` -- e.g. ``[try_build_rl_deblur_method(...)]`` -- is
    appended after the ``CORE_METHODS`` rows so ``rl_deblur_restore`` shows
    up here too when a checkpoint is available.
    """
    chosen = [get_core_method(name) for name in methods] if methods else list(CORE_METHODS)
    chosen = chosen + list(extra_methods or [])
    samples = build_split(seed=seed)[split][:num_samples]

    fig, axes = plt.subplots(
        len(chosen),
        len(samples),
        figsize=(max(6.0, 1.6 * len(samples)), 1.15 * len(chosen)),
        squeeze=False,
    )
    for row, method in enumerate(chosen):
        for col, (path, label) in enumerate(samples):
            image = Image.open(path).convert("RGB")
            canvas = to_canvas_gray(image, method.resample)
            processed = method.process(canvas)
            ax = axes[row][col]
            ax.imshow(processed, cmap="gray", vmin=0, vmax=255)
            ax.set_xticks([])
            ax.set_yticks([])
            if col == 0:
                ax.set_ylabel(method.name, fontsize=8, rotation=0, ha="right", va="center")
            if row == 0:
                ax.set_title(label, fontsize=7)
    fig.suptitle(f"Ảnh sau xử lý cho từng phương pháp ({num_samples} mẫu từ split '{split}')")
    fig.tight_layout()
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
    return output_path


def plot_experiment_a_bar_chart(comparison_csv: str | Path, output_path: str | Path) -> Path:
    df = pd.read_csv(comparison_csv).sort_values("test_exact_acc", ascending=False)
    fig, ax = plt.subplots(figsize=(max(6, 0.6 * len(df)), 4.5))
    ax.bar(df["method"], df["test_exact_acc"], color="#4C72B0")
    ax.set_ylabel("Test exact-match accuracy")
    ax.set_title("Experiment A: OCR accuracy by image processing method")
    ax.set_ylim(0, max(1.0, float(df["test_exact_acc"].max()) * 1.15 + 1e-6))
    ax.tick_params(axis="x", rotation=60)
    for label in ax.get_xticklabels():
        label.set_ha("right")
    fig.tight_layout()
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
    return output_path


def plot_experiment_b_bar_chart(comparison_csv: str | Path, output_path: str | Path) -> Path:
    df = pd.read_csv(comparison_csv).sort_values("psnr_mean", ascending=False)
    finite = df[df["psnr_mean"].apply(lambda v: v == v and v not in (float("inf"), float("-inf")))]
    fig, ax = plt.subplots(figsize=(max(6, 0.6 * len(finite)), 4.5))
    ax.bar(finite["method"], finite["psnr_mean"], color="#55A868")
    ax.set_ylabel("PSNR vs clean ground truth (dB)")
    ax.set_title("Experiment B: restoration quality on synthetic degradation")
    ax.tick_params(axis="x", rotation=60)
    for label in ax.get_xticklabels():
        label.set_ha("right")
    fig.tight_layout()
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
    return output_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sample-image", default="")
    parser.add_argument("--experiment-a-csv", default=str(REPO_ROOT / "outputs" / "image_processing_study" / "experiment_a" / "comparison.csv"))
    parser.add_argument("--experiment-b-csv", default=str(REPO_ROOT / "outputs" / "image_processing_study" / "experiment_b" / "comparison.csv"))
    parser.add_argument("--output-dir", default=str(DEFAULT_SAMPLE_DIR))
    parser.add_argument("--include-rl", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--rl-checkpoint", default=str(REPO_ROOT / "outputs" / "rl_deblur" / "checkpoints" / "best_deblur_agent.pt"))
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    output_dir = Path(args.output_dir)

    rl_method = None
    if args.include_rl:
        from image_processing_study.methods import try_build_rl_deblur_method

        rl_method = try_build_rl_deblur_method(args.rl_checkpoint)
    extra = [rl_method] if rl_method is not None else None

    path = make_method_preview_grid(output_dir / "preview_processed_samples.png", extra_methods=extra)
    print(f"Saved {path}")
    if args.sample_image:
        path = make_method_grid(args.sample_image, output_dir / "experiment_a_methods_grid.png", extra_methods=extra)
        print(f"Saved {path}")
    path = make_restoration_grid(output_dir / "experiment_b_restoration_grid.png", extra_methods=extra)
    print(f"Saved {path}")
    if Path(args.experiment_a_csv).exists():
        path = plot_experiment_a_bar_chart(args.experiment_a_csv, output_dir / "experiment_a_accuracy_bar.png")
        print(f"Saved {path}")
    if Path(args.experiment_b_csv).exists():
        path = plot_experiment_b_bar_chart(args.experiment_b_csv, output_dir / "experiment_b_psnr_bar.png")
        print(f"Saved {path}")
