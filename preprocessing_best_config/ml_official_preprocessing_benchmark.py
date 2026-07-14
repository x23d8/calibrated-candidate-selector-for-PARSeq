"""Benchmark author-released ML enhancement weights before fine-tuned PARSeq.

Model selection is validation-only.  The locked test split is evaluated for the
two classical anchors and the validation-selected ML finalists.
"""

from __future__ import annotations

import argparse
import gc
import hashlib
import importlib.util
import json
import runpy
import sys
import time
import types
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms as T
from tqdm.auto import tqdm

PIPELINE_DIR = Path(__file__).resolve().parents[1]
OFFICIAL_ROOT = PIPELINE_DIR / ".cache" / "official_ml"
sys.path.insert(0, str(PIPELINE_DIR / "train_no_refinement"))
sys.path.insert(0, str(PIPELINE_DIR / "parseq"))

try:
    from .find_best_preprocessing_config import (
        ManifestPlateDataset,
        build_transform,
        json_safe,
        load_notebook_checkpoint,
        metrics_from_predictions,
        paired_deltas,
    )
    from .preprocessing import iter_named_configs
except ImportError:
    from find_best_preprocessing_config import (  # type: ignore
        ManifestPlateDataset,
        build_transform,
        json_safe,
        load_notebook_checkpoint,
        metrics_from_predictions,
        paired_deltas,
    )
    from preprocessing import iter_named_configs  # type: ignore

from parseq_official_anpr_pipeline import edit_distance, greedy_decode, normalize_plate_text  # noqa: E402


@dataclass(frozen=True)
class OfficialModelSpec:
    name: str
    description: str
    repository: str
    repository_commit: str
    weight_url: str
    weight_path: Path
    sha256: str


MODEL_SPECS = {
    "realesrgan_x2plus": OfficialModelSpec(
        name="realesrgan_x2plus",
        description="Real-ESRGAN x2 blind super-resolution (author release).",
        repository="https://github.com/xinntao/Real-ESRGAN",
        repository_commit="a4abfb2979a7bbff3f69f58f58ae324608821e27",
        weight_url="https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.1/RealESRGAN_x2plus.pth",
        weight_path=OFFICIAL_ROOT / "weights" / "Real-ESRGAN" / "RealESRGAN_x2plus.pth",
        sha256="49FAFD45F8FD7AA8D31AB2A22D14D91B536C34494A5CFE31EB5D89C2FA266ABB",
    ),
    "restormer_motion_deblur": OfficialModelSpec(
        name="restormer_motion_deblur",
        description="Restormer single-image motion deblurring (author release).",
        repository="https://github.com/swz30/Restormer",
        repository_commit="68dc6ac472db26f16361150cb7a96a1bc87da93f",
        weight_url="https://drive.google.com/drive/folders/1czMyfRTQDX3j3ErByYeZ1PM4GVLbJeGK",
        weight_path=OFFICIAL_ROOT / "weights" / "Restormer" / "motion_deblurring.pth",
        sha256="194E38FB5B607C9DC5A5B3E08E65B2E79EE2BF0EF5048E0612F6B2FF2F79DA31",
    ),
    "zero_dce": OfficialModelSpec(
        name="zero_dce",
        description="Zero-DCE zero-reference low-light enhancement (weight committed by authors).",
        repository="https://github.com/Li-Chongyi/Zero-DCE",
        repository_commit="e0f4adc54d0f23348c4a9b84acc08fe8778d5bfd",
        weight_url="https://github.com/Li-Chongyi/Zero-DCE/blob/master/Zero-DCE_code/snapshots/Epoch99.pth",
        weight_path=OFFICIAL_ROOT / "Zero-DCE" / "Zero-DCE_code" / "snapshots" / "Epoch99.pth",
        sha256="A4395ACB874F320375D9704997CEF874EAAAAA26A1777CEB29A92B70F74C3612",
    ),
}

# Base variants enhance a standardized 32x128 input. Native variants enhance
# each original crop and resize only the result passed to PARSeq.
ML_VARIANTS = {
    "realesrgan_x2plus": ("realesrgan_x2plus", "resized"),
    "realesrgan_x2plus_native": ("realesrgan_x2plus", "native"),
    "restormer_motion_deblur": ("restormer_motion_deblur", "resized"),
    "restormer_motion_deblur_native": ("restormer_motion_deblur", "native"),
    "zero_dce": ("zero_dce", "resized"),
    "zero_dce_native": ("zero_dce", "native"),
}

ANCHORS = ("train_baseline", "clahe_clip1_tile4", "raw_rgb")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest().upper()


def verify_official_assets() -> list[dict]:
    records = []
    for spec in MODEL_SPECS.values():
        if not spec.weight_path.exists():
            raise FileNotFoundError(f"Missing official weight: {spec.weight_path}")
        actual = sha256_file(spec.weight_path)
        if actual != spec.sha256:
            raise RuntimeError(f"SHA-256 mismatch for {spec.name}: expected {spec.sha256}, got {actual}")
        records.append(
            {
                "name": spec.name,
                "description": spec.description,
                "repository": spec.repository,
                "repository_commit": spec.repository_commit,
                "weight_url": spec.weight_url,
                "weight_path": str(spec.weight_path.resolve()),
                "size_bytes": spec.weight_path.stat().st_size,
                "sha256": actual,
                "verified": True,
            }
        )
    return records


class ResizedManifestDataset(Dataset):
    """Resize first so every learned enhancer receives the same 32x128 RGB tensor."""

    def __init__(self, manifest: Path, image_size: tuple[int, int]):
        frame = pd.read_csv(manifest)
        missing = {"image_path", "target"} - set(frame.columns)
        if missing:
            raise ValueError(f"{manifest} is missing columns: {sorted(missing)}")
        self.frame = frame.reset_index(drop=True)
        self.frame["target"] = self.frame["target"].map(normalize_plate_text)
        self.transform = T.Compose(
            [T.Resize(image_size, interpolation=T.InterpolationMode.BICUBIC), T.ToTensor()]
        )

    def __len__(self):
        return len(self.frame)

    def __getitem__(self, index):
        row = self.frame.iloc[index]
        image = self.transform(Image.open(row["image_path"]).convert("RGB"))
        metadata = {
            key: row.get(key, "")
            for key in ("split", "source_name", "plate_type", "label_status", "review_status")
        }
        return image, row["target"], str(row["image_path"]), metadata


class NativeManifestDataset(ResizedManifestDataset):
    """Keep the original crop resolution; evaluated with batch size one."""

    def __init__(self, manifest: Path, image_size: tuple[int, int]):
        super().__init__(manifest, image_size)
        self.transform = T.ToTensor()


def collate_batch(batch):
    images, labels, paths, metadata = zip(*batch)
    return torch.stack(list(images)), list(labels), list(paths), list(metadata)


def _load_realesrgan(device: torch.device):
    # basicsr 1.4.2 imports a torchvision module removed in torchvision 0.24;
    # this compatibility alias contains the same public function it expects.
    if "torchvision.transforms.functional_tensor" not in sys.modules:
        from torchvision.transforms.functional import rgb_to_grayscale

        shim = types.ModuleType("torchvision.transforms.functional_tensor")
        shim.rgb_to_grayscale = rgb_to_grayscale
        sys.modules["torchvision.transforms.functional_tensor"] = shim
    from basicsr.archs.rrdbnet_arch import RRDBNet

    model = RRDBNet(
        num_in_ch=3, num_out_ch=3, num_feat=64, num_block=23, num_grow_ch=32, scale=2
    )
    payload = torch.load(MODEL_SPECS["realesrgan_x2plus"].weight_path, map_location="cpu", weights_only=False)
    model.load_state_dict(payload["params_ema"], strict=True)
    return model.eval().to(device)


def _load_restormer(device: torch.device):
    architecture = runpy.run_path(
        str(OFFICIAL_ROOT / "Restormer" / "basicsr" / "models" / "archs" / "restormer_arch.py")
    )
    model = architecture["Restormer"](
        inp_channels=3,
        out_channels=3,
        dim=48,
        num_blocks=[4, 6, 6, 8],
        num_refinement_blocks=4,
        heads=[1, 2, 4, 8],
        ffn_expansion_factor=2.66,
        bias=False,
        LayerNorm_type="WithBias",
        dual_pixel_task=False,
    )
    payload = torch.load(MODEL_SPECS["restormer_motion_deblur"].weight_path, map_location="cpu", weights_only=False)
    model.load_state_dict(payload["params"], strict=True)
    return model.eval().to(device)


def _load_zero_dce(device: torch.device):
    model_file = OFFICIAL_ROOT / "Zero-DCE" / "Zero-DCE_code" / "model.py"
    module_spec = importlib.util.spec_from_file_location("official_zero_dce_model", model_file)
    if module_spec is None or module_spec.loader is None:
        raise ImportError(f"Cannot import {model_file}")
    module = importlib.util.module_from_spec(module_spec)
    module_spec.loader.exec_module(module)
    model = module.enhance_net_nopool()
    payload = torch.load(MODEL_SPECS["zero_dce"].weight_path, map_location="cpu", weights_only=False)
    model.load_state_dict(payload, strict=True)
    return model.eval().to(device)


def load_enhancer(name: str, device: torch.device):
    loaders = {
        "realesrgan_x2plus": _load_realesrgan,
        "restormer_motion_deblur": _load_restormer,
        "zero_dce": _load_zero_dce,
    }
    return loaders[name](device)


def enhance(name: str, model, images: torch.Tensor) -> torch.Tensor:
    original_size = images.shape[-2:]
    divisor = 8 if name == "restormer_motion_deblur" else 2 if name == "realesrgan_x2plus" else 1
    if divisor > 1:
        pad_height = (divisor - original_size[0] % divisor) % divisor
        pad_width = (divisor - original_size[1] % divisor) % divisor
        if pad_height or pad_width:
            images = F.pad(images, (0, pad_width, 0, pad_height), mode="reflect")
    if name == "zero_dce":
        _first, result, _curves = model(images)
    else:
        result = model(images)
    if name == "restormer_motion_deblur":
        result = result[..., : original_size[0], : original_size[1]]
    elif name == "realesrgan_x2plus":
        result = result[..., : original_size[0] * 2, : original_size[1] * 2]
    return result.clamp_(0, 1)


@torch.inference_mode()
def evaluate_ml(
    model, enhancer, variant_name, model_name, input_mode, loader, device, split, max_length, image_size
):
    rows = []
    start = time.perf_counter()
    for images, labels, paths, metadata in tqdm(loader, desc=f"{split}: {variant_name}", leave=False):
        images = images.to(device, non_blocking=True)
        restored = enhance(model_name, enhancer, images)
        if tuple(restored.shape[-2:]) != tuple(image_size):
            restored = F.interpolate(
                restored, size=image_size, mode="bicubic", align_corners=False, antialias=True
            ).clamp_(0, 1)
        parseq_input = restored.sub(0.5).div(0.5)
        preds, confidences = greedy_decode(model, parseq_input, max_length=max_length)
        for path, target, pred, confidence, meta in zip(
            paths, labels, preds, confidences.cpu().tolist(), metadata
        ):
            rows.append(
                {
                    "config": variant_name,
                    "image_path": path,
                    "target": target,
                    "prediction": pred,
                    "exact": pred == target,
                    "edit_distance": edit_distance(pred, target),
                    "target_length": max(len(target), 1),
                    "confidence": confidence,
                    **meta,
                }
            )
    elapsed = time.perf_counter() - start
    predictions = pd.DataFrame(rows)
    metrics = metrics_from_predictions(predictions)
    metrics.update(
        config=variant_name,
        category="official_ml",
        description=f"{MODEL_SPECS[model_name].description} Input mode: {input_mode}.",
        official_model=model_name,
        input_mode=input_mode,
        split=split,
        seconds=elapsed,
        images_per_second=len(predictions) / max(elapsed, 1e-9),
    )
    return metrics, predictions


@torch.inference_mode()
def evaluate_anchor(model, manifest, cfg, device, split, model_cfg, args):
    dataset = ManifestPlateDataset(manifest, build_transform(model_cfg.img_size, cfg))
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        collate_fn=collate_batch,
        pin_memory=device.type == "cuda",
    )
    rows = []
    start = time.perf_counter()
    for images, labels, paths, metadata in tqdm(loader, desc=f"{split}: {cfg.name}", leave=False):
        preds, confidences = greedy_decode(
            model, images.to(device, non_blocking=True), max_length=model_cfg.max_label_length
        )
        for path, target, pred, confidence, meta in zip(
            paths, labels, preds, confidences.cpu().tolist(), metadata
        ):
            rows.append(
                {
                    "config": cfg.name,
                    "image_path": path,
                    "target": target,
                    "prediction": pred,
                    "exact": pred == target,
                    "edit_distance": edit_distance(pred, target),
                    "target_length": max(len(target), 1),
                    "confidence": confidence,
                    **meta,
                }
            )
    elapsed = time.perf_counter() - start
    predictions = pd.DataFrame(rows)
    metrics = metrics_from_predictions(predictions)
    metrics.update(
        config=cfg.name,
        category="classical_anchor",
        description=cfg.description,
        split=split,
        seconds=elapsed,
        images_per_second=len(predictions) / max(elapsed, 1e-9),
    )
    return metrics, predictions


def enrich_results(rows, predictions, baseline_name, args, seed_offset=0):
    baseline = predictions[baseline_name]
    enriched = []
    for index, row in enumerate(rows):
        deltas = paired_deltas(
            predictions[row["config"]], baseline, args.bootstrap_samples, args.seed + seed_offset + index
        )
        enriched.append({**row, **deltas})
    return pd.DataFrame(enriched).sort_values(
        ["exact_acc", "char_acc", "images_per_second"], ascending=[False, False, False]
    ).reset_index(drop=True)


def release_model(model):
    del model
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return None


def run_benchmark(args):
    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    run_dir = Path(args.run_dir).resolve() if args.run_dir else max(
        [
            path
            for path in (PIPELINE_DIR / "outputs").glob("refinement_finetune*")
            if (path / "best_official_parseq_anpr.pt").exists()
            and (path / "eval_val_predictions_best_refine.csv").exists()
        ],
        key=lambda path: path.stat().st_mtime,
    )
    checkpoint = Path(args.checkpoint).resolve() if args.checkpoint else run_dir / "best_official_parseq_anpr.pt"
    val_manifest = Path(args.val_manifest).resolve() if args.val_manifest else run_dir / "eval_val_predictions_best_refine.csv"
    test_manifest = Path(args.test_manifest).resolve() if args.test_manifest else run_dir / "eval_test_predictions_best_refine.csv"
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    provenance = verify_official_assets()
    (output_dir / "official_model_provenance.json").write_text(
        json.dumps(provenance, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    parseq_model, model_cfg, checkpoint_payload = load_notebook_checkpoint(
        checkpoint, device, args.refine_iters
    )
    all_classical = {cfg.name: cfg for cfg in iter_named_configs(None)}
    missing_anchors = set(ANCHORS) - set(all_classical)
    if missing_anchors:
        raise KeyError(f"Missing classical anchors: {sorted(missing_anchors)}")

    def run_split(split, manifest, method_names):
        rows, predictions = [], {}
        for name in method_names:
            if name in all_classical:
                metrics, frame = evaluate_anchor(
                    parseq_model, manifest, all_classical[name], device, split, model_cfg, args
                )
            else:
                model_name, input_mode = ML_VARIANTS[name]
                dataset_class = NativeManifestDataset if input_mode == "native" else ResizedManifestDataset
                dataset = dataset_class(manifest, tuple(model_cfg.img_size))
                loader = DataLoader(
                    dataset,
                    batch_size=1 if input_mode == "native" else args.ml_batch_size,
                    shuffle=False,
                    num_workers=args.num_workers,
                    collate_fn=collate_batch,
                    pin_memory=device.type == "cuda",
                )
                enhancer = load_enhancer(model_name, device)
                metrics, frame = evaluate_ml(
                    parseq_model,
                    enhancer,
                    name,
                    model_name,
                    input_mode,
                    loader,
                    device,
                    split,
                    model_cfg.max_label_length,
                    tuple(model_cfg.img_size),
                )
                enhancer = release_model(enhancer)
            rows.append(metrics)
            predictions[name] = frame
            frame.to_csv(output_dir / f"predictions_{split}_{name}.csv", index=False)
        return rows, predictions

    ml_names = list(ML_VARIANTS)
    val_rows, val_predictions = run_split("val", val_manifest, [*ANCHORS, *ml_names])
    val_results = enrich_results(val_rows, val_predictions, "train_baseline", args)
    val_results.to_csv(output_dir / "validation_results.csv", index=False)

    ranked_ml = val_results[val_results["category"] == "official_ml"]
    selected_ml = ranked_ml.head(args.top_k_ml)["config"].tolist()
    test_methods = [*ANCHORS, *selected_ml]
    test_rows, test_predictions = run_split("test", test_manifest, test_methods)
    test_results = enrich_results(test_rows, test_predictions, "train_baseline", args, seed_offset=1000)
    test_results.to_csv(output_dir / "test_finalists_results.csv", index=False)

    summary = {
        "protocol": {
            "selection": "Rank ML methods by validation exact_acc, then char_acc; test only top-k ML plus anchors.",
            "ml_input": (
                f"Base variants use RGB bicubic resize to {tuple(model_cfg.img_size)} before enhancement; "
                "native variants enhance the original crop before resize."
            ),
            "parseq_input": f"Enhancer output bicubic resize to {tuple(model_cfg.img_size)} then normalize to [-1, 1]",
            "baseline": "train_baseline",
            "top_k_ml": args.top_k_ml,
        },
        "paths": {
            "checkpoint": str(checkpoint.resolve()),
            "validation_manifest": str(val_manifest.resolve()),
            "test_manifest": str(test_manifest.resolve()),
        },
        "device": str(device),
        "checkpoint_epoch": checkpoint_payload.get("epoch"),
        "selected_ml": selected_ml,
        "validation": val_results.to_dict(orient="records"),
        "test": test_results.to_dict(orient="records"),
        "official_models": provenance,
    }
    (output_dir / "summary.json").write_text(
        json.dumps(json_safe(summary), indent=2, ensure_ascii=False, allow_nan=False), encoding="utf-8"
    )
    return val_results, test_results


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", default="")
    parser.add_argument("--checkpoint", default="")
    parser.add_argument("--val-manifest", default="")
    parser.add_argument("--test-manifest", default="")
    parser.add_argument(
        "--output-dir", default=str(PIPELINE_DIR / "outputs" / "ml_official_preprocessing_benchmark")
    )
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--ml-batch-size", type=int, default=8)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--top-k-ml", type=int, default=2)
    parser.add_argument("--refine-iters", type=int, default=2)
    parser.add_argument("--bootstrap-samples", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="")
    return parser.parse_args()


if __name__ == "__main__":
    validation, test = run_benchmark(parse_args())
    columns = ["config", "exact_acc", "char_acc", "delta_exact", "delta_char_acc"]
    print("\nValidation ranking")
    print(validation[columns].to_string(index=False))
    print("\nLocked test finalists")
    print(test[columns].to_string(index=False))
