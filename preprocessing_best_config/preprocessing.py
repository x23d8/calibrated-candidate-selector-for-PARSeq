"""Deterministic plate-image preprocessing variants for PARSeq ANPR.

The variants map directly to topics in the IMP302m course (gray-level
processing, linear/non-linear filtering, morphology, restoration/wavelets,
and binary processing).  All functions return RGB PIL images because PARSeq
expects three input channels even when the useful signal is grayscale.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Iterable

import numpy as np
from PIL import Image, ImageEnhance, ImageOps


@dataclass(frozen=True)
class PreprocessingConfig:
    name: str
    course_topic: str
    description: str
    grayscale: bool = True
    gray_channel: str = "luma"  # luma, red, green, blue, max, hsv_v, lab_l, best_contrast
    autocontrast: bool = False
    histogram_equalization: bool = False
    percentile_low: float | None = None
    percentile_high: float | None = None
    clahe_clip_limit: float | None = None
    clahe_tile_size: int = 8
    gamma: float = 1.0
    illumination: str = "none"  # none, retinex, multiscale_retinex, homomorphic, local_norm
    illumination_sigma: float = 15.0
    denoise: str = "none"  # none, gaussian, median, bilateral, wavelet_haar, nlm, wiener
    gaussian_sigma: float = 0.8
    median_ksize: int = 3
    bilateral_d: int = 5
    bilateral_sigma_color: float = 50.0
    bilateral_sigma_space: float = 50.0
    nlm_h: float = 3.0
    wiener_ksize: int = 3
    sharpen_alpha: float = 0.0
    sharpen_sigma: float = 1.0
    sharpen_method: str = "unsharp"  # unsharp, laplacian, dog
    sharpen_sigma_large: float = 1.6
    morphology: str = "none"  # none, close, blackhat, gradient
    morphology_ksize: int = 3
    morphology_kernel_width: int | None = None
    morphology_kernel_height: int | None = None
    morphology_strength: float = 0.6
    threshold: str = "none"  # none, otsu, adaptive
    adaptive_block_size: int = 25
    adaptive_c: int = 7
    resize_interpolation: str = "bicubic"  # applied by the benchmark transform
    resize_mode: str = "stretch"  # stretch or letterbox

    def to_dict(self) -> dict:
        return asdict(self)


RAW_CONFIG = PreprocessingConfig(
    name="raw_rgb",
    course_topic="Reference",
    description="Original RGB crop; resize and normalize only.",
    grayscale=False,
)

# This exactly reproduces preprocess_plate_image() used to train the supplied
# refinement checkpoint: grayscale -> CLAHE -> bilateral -> unsharp (1.5/-0.5).
DEFAULT_CONFIG = PreprocessingConfig(
    name="train_baseline",
    course_topic="2.1-2.2 Linear and nonlinear enhancement",
    description="Training-time baseline: gray + CLAHE + bilateral + mild unsharp mask.",
    clahe_clip_limit=2.0,
    denoise="bilateral",
    sharpen_alpha=0.5,
)


SWEEP_CONFIGS = [
    DEFAULT_CONFIG,
    RAW_CONFIG,
    PreprocessingConfig(
        name="grayscale",
        course_topic="1.1 Basic gray-level processing",
        description="Grayscale only.",
    ),
    PreprocessingConfig(
        name="autocontrast",
        course_topic="1.1 Basic gray-level processing",
        description="Global percentile-free contrast stretching.",
        autocontrast=True,
    ),
    PreprocessingConfig(
        name="hist_equalization",
        course_topic="1.1 Basic gray-level processing",
        description="Global histogram equalization.",
        histogram_equalization=True,
    ),
    PreprocessingConfig(
        name="clahe_gray",
        course_topic="2 Image enhancement",
        description="Local contrast enhancement on grayscale luminance.",
        clahe_clip_limit=2.0,
    ),
    PreprocessingConfig(
        name="clahe_lab",
        course_topic="3.10 Multichannel image recovery",
        description="Color-preserving CLAHE on the LAB luminance channel.",
        grayscale=False,
        clahe_clip_limit=2.0,
    ),
    PreprocessingConfig(
        name="clahe_gaussian",
        course_topic="2.1 Linear filtering",
        description="CLAHE followed by Gaussian noise suppression.",
        clahe_clip_limit=2.0,
        denoise="gaussian",
    ),
    PreprocessingConfig(
        name="clahe_median",
        course_topic="2.2 Nonlinear filtering",
        description="CLAHE followed by median filtering for impulse noise.",
        clahe_clip_limit=2.0,
        denoise="median",
    ),
    PreprocessingConfig(
        name="clahe_bilateral",
        course_topic="2.2 Nonlinear filtering",
        description="Training baseline without sharpening (edge-preserving denoise ablation).",
        clahe_clip_limit=2.0,
        denoise="bilateral",
    ),
    PreprocessingConfig(
        name="clahe_unsharp",
        course_topic="2.1 Spatial enhancement",
        description="Training baseline without bilateral filtering (sharpening ablation).",
        clahe_clip_limit=2.0,
        sharpen_alpha=0.5,
    ),
    PreprocessingConfig(
        name="baseline_strong_unsharp",
        course_topic="2.1 Spatial enhancement",
        description="Training baseline with stronger high-frequency emphasis.",
        clahe_clip_limit=2.0,
        denoise="bilateral",
        sharpen_alpha=1.0,
    ),
    PreprocessingConfig(
        name="baseline_morph_close",
        course_topic="2.3 Morphological filtering",
        description="Training baseline plus a small closing operation to reconnect strokes.",
        clahe_clip_limit=2.0,
        denoise="bilateral",
        sharpen_alpha=0.5,
        morphology="close",
    ),
    PreprocessingConfig(
        name="baseline_blackhat",
        course_topic="2.3 Morphological filtering",
        description="Training baseline plus black-hat dark-stroke enhancement.",
        clahe_clip_limit=2.0,
        denoise="bilateral",
        sharpen_alpha=0.5,
        morphology="blackhat",
    ),
    PreprocessingConfig(
        name="baseline_gamma_0_8",
        course_topic="1.1 Gray-level transforms",
        description="Gamma brightening before the training-time pipeline.",
        gamma=0.8,
        clahe_clip_limit=2.0,
        denoise="bilateral",
        sharpen_alpha=0.5,
    ),
    PreprocessingConfig(
        name="baseline_gamma_1_2",
        course_topic="1.1 Gray-level transforms",
        description="Gamma darkening before the training-time pipeline.",
        gamma=1.2,
        clahe_clip_limit=2.0,
        denoise="bilateral",
        sharpen_alpha=0.5,
    ),
    PreprocessingConfig(
        name="clahe_wavelet_haar",
        course_topic="3.5 Wavelet denoising",
        description="CLAHE plus single-level Haar soft-threshold denoising.",
        clahe_clip_limit=2.0,
        denoise="wavelet_haar",
        sharpen_alpha=0.5,
    ),
    PreprocessingConfig(
        name="otsu_threshold",
        course_topic="1.2 Basic binary processing",
        description="CLAHE followed by global Otsu binarization.",
        clahe_clip_limit=2.0,
        threshold="otsu",
    ),
    PreprocessingConfig(
        name="adaptive_threshold",
        course_topic="1.2 Basic binary processing",
        description="CLAHE followed by local adaptive binarization.",
        clahe_clip_limit=2.0,
        threshold="adaptive",
    ),
    PreprocessingConfig(
        name="baseline_resize_bilinear",
        course_topic="7.1 Image sampling and interpolation",
        description="Training-time enhancement with bilinear model-input resizing.",
        clahe_clip_limit=2.0,
        denoise="bilateral",
        sharpen_alpha=0.5,
        resize_interpolation="bilinear",
    ),
    PreprocessingConfig(
        name="baseline_resize_lanczos",
        course_topic="7.1 Image sampling and interpolation",
        description="Training-time enhancement with Lanczos model-input resizing.",
        clahe_clip_limit=2.0,
        denoise="bilateral",
        sharpen_alpha=0.5,
        resize_interpolation="lanczos",
    ),
    # Extended classical image-processing sweep.
    PreprocessingConfig(
        name="percentile_stretch_1_99",
        course_topic="1.1 Gray-level transforms",
        description="Robust 1st-99th percentile contrast stretching.",
        percentile_low=1.0,
        percentile_high=99.0,
    ),
    PreprocessingConfig(
        name="percentile_stretch_2_98",
        course_topic="1.1 Gray-level transforms",
        description="Robust 2nd-98th percentile contrast stretching.",
        percentile_low=2.0,
        percentile_high=98.0,
    ),
    PreprocessingConfig(
        name="gamma_0_9",
        course_topic="1.1 Gray-level transforms",
        description="Mild gamma brightening without local enhancement.",
        gamma=0.9,
    ),
    PreprocessingConfig(
        name="gamma_1_1",
        course_topic="1.1 Gray-level transforms",
        description="Mild gamma darkening without local enhancement.",
        gamma=1.1,
    ),
    PreprocessingConfig(
        name="retinex_single",
        course_topic="3 Image restoration",
        description="Single-scale Retinex illumination normalization.",
        illumination="retinex",
        illumination_sigma=15.0,
    ),
    PreprocessingConfig(
        name="retinex_multiscale",
        course_topic="3 Image restoration",
        description="Multi-scale Retinex illumination normalization.",
        illumination="multiscale_retinex",
    ),
    PreprocessingConfig(
        name="homomorphic_filter",
        course_topic="2.5 Frequency-domain filtering",
        description="Homomorphic high-pass illumination correction in the log-frequency domain.",
        illumination="homomorphic",
    ),
    PreprocessingConfig(
        name="local_contrast_norm",
        course_topic="2.4 Spatial filtering",
        description="Local mean and variance normalization.",
        illumination="local_norm",
    ),
    PreprocessingConfig(
        name="nlm_denoise",
        course_topic="3.3 Noise reduction",
        description="Mild non-local means denoising.",
        denoise="nlm",
        nlm_h=3.0,
    ),
    PreprocessingConfig(
        name="wiener_3x3",
        course_topic="3.6 Minimum mean square error filtering",
        description="Adaptive local Wiener denoising with a 3x3 window.",
        denoise="wiener",
    ),
    PreprocessingConfig(
        name="unsharp_mild",
        course_topic="2.1 Spatial enhancement",
        description="Mild unsharp masking on grayscale.",
        sharpen_alpha=0.25,
    ),
    PreprocessingConfig(
        name="laplacian_mild",
        course_topic="2.4 Spatial filtering",
        description="Mild Laplacian high-frequency sharpening.",
        sharpen_alpha=0.20,
        sharpen_method="laplacian",
    ),
    PreprocessingConfig(
        name="dog_sharpen",
        course_topic="2.4 Spatial filtering",
        description="Difference-of-Gaussians band-pass sharpening.",
        sharpen_alpha=0.35,
        sharpen_method="dog",
        sharpen_sigma=0.6,
        sharpen_sigma_large=1.4,
    ),
    PreprocessingConfig(
        name="channel_red",
        course_topic="3.10 Multichannel image recovery",
        description="Use the red channel as a replicated grayscale input.",
        gray_channel="red",
    ),
    PreprocessingConfig(
        name="channel_green",
        course_topic="3.10 Multichannel image recovery",
        description="Use the green channel as a replicated grayscale input.",
        gray_channel="green",
    ),
    PreprocessingConfig(
        name="channel_blue",
        course_topic="3.10 Multichannel image recovery",
        description="Use the blue channel; potentially useful for yellow plates.",
        gray_channel="blue",
    ),
    PreprocessingConfig(
        name="channel_max_rgb",
        course_topic="3.10 Multichannel image recovery",
        description="Use the maximum RGB channel per pixel.",
        gray_channel="max",
    ),
    PreprocessingConfig(
        name="channel_hsv_value",
        course_topic="3.10 Multichannel image recovery",
        description="Use the HSV value channel.",
        gray_channel="hsv_v",
    ),
    PreprocessingConfig(
        name="channel_lab_l",
        course_topic="3.10 Multichannel image recovery",
        description="Use perceptual LAB luminance.",
        gray_channel="lab_l",
    ),
    PreprocessingConfig(
        name="channel_best_contrast",
        course_topic="3.10 Multichannel image recovery",
        description="Select the RGB/luma channel with the largest robust intensity range per image.",
        gray_channel="best_contrast",
    ),
    PreprocessingConfig(
        name="morph_close_horizontal",
        course_topic="2.3 Morphological filtering",
        description="Light horizontal 3x1 closing to reconnect broken horizontal strokes.",
        morphology="close",
        morphology_kernel_width=3,
        morphology_kernel_height=1,
    ),
    PreprocessingConfig(
        name="morph_close_vertical",
        course_topic="2.3 Morphological filtering",
        description="Light vertical 1x3 closing to reconnect broken vertical strokes.",
        morphology="close",
        morphology_kernel_width=1,
        morphology_kernel_height=3,
    ),
    PreprocessingConfig(
        name="morph_gradient_mild",
        course_topic="2.3 Morphological filtering",
        description="Blend a mild morphological edge gradient into grayscale.",
        morphology="gradient",
        morphology_strength=0.20,
    ),
    PreprocessingConfig(
        name="clahe_clip1_tile4",
        course_topic="2 Image enhancement",
        description="Gentler CLAHE with smaller 4x4 tiles.",
        clahe_clip_limit=1.0,
        clahe_tile_size=4,
    ),
    PreprocessingConfig(
        name="clahe_clip1_tile4_unsharp",
        course_topic="2 Image enhancement",
        description="Gentle 4x4 CLAHE plus mild unsharp masking.",
        clahe_clip_limit=1.0,
        clahe_tile_size=4,
        sharpen_alpha=0.25,
    ),
    PreprocessingConfig(
        name="grayscale_letterbox",
        course_topic="7.1 Image sampling and interpolation",
        description="Preserve crop aspect ratio and pad before PARSeq normalization.",
        resize_mode="letterbox",
    ),
]

# Backwards-compatible aliases used by older commands in this repository.
_ALIASES = {
    "raw": "raw_rgb",
    "clahe_sharpen": "train_baseline",
    "clahe_1_5": "clahe_gray",
    "clahe_2_sharp_1_2": "clahe_unsharp",
    "clahe_3_sharp_1_5": "baseline_strong_unsharp",
    "adaptive_thresh": "adaptive_threshold",
}


def get_preprocessing_config(name: str) -> PreprocessingConfig:
    name = _ALIASES.get(name, name)
    for cfg in SWEEP_CONFIGS:
        if cfg.name == name:
            return cfg
    raise KeyError(f"Unknown preprocessing config: {name}")


def list_preprocessing_configs() -> list[str]:
    return [cfg.name for cfg in SWEEP_CONFIGS]


def _odd_at_least(value: int, minimum: int = 3) -> int:
    value = max(int(value), minimum)
    return value if value % 2 else value + 1


def _haar_soft_threshold(channel: np.ndarray) -> np.ndarray:
    """Dependency-free, one-level orthonormal Haar wavelet denoising."""
    src = channel.astype(np.float32)
    orig_h, orig_w = src.shape
    if orig_h % 2 or orig_w % 2:
        src = np.pad(src, ((0, orig_h % 2), (0, orig_w % 2)), mode="reflect")
    a, b = src[0::2, 0::2], src[0::2, 1::2]
    c, d = src[1::2, 0::2], src[1::2, 1::2]
    ll = (a + b + c + d) / 2.0
    lh = (a - b + c - d) / 2.0
    hl = (a + b - c - d) / 2.0
    hh = (a - b - c + d) / 2.0
    sigma = float(np.median(np.abs(hh)) / 0.6745) if hh.size else 0.0
    threshold = sigma * np.sqrt(2.0 * np.log(max(src.size, 2)))

    def shrink(detail: np.ndarray) -> np.ndarray:
        return np.sign(detail) * np.maximum(np.abs(detail) - threshold, 0.0)

    lh, hl, hh = shrink(lh), shrink(hl), shrink(hh)
    out = np.empty_like(src)
    out[0::2, 0::2] = (ll + lh + hl + hh) / 2.0
    out[0::2, 1::2] = (ll - lh + hl - hh) / 2.0
    out[1::2, 0::2] = (ll + lh - hl - hh) / 2.0
    out[1::2, 1::2] = (ll - lh - hl + hh) / 2.0
    return np.clip(out[:orig_h, :orig_w], 0, 255).astype(np.uint8)


def _robust_rescale(channel: np.ndarray, low_percentile: float = 1.0, high_percentile: float = 99.0) -> np.ndarray:
    src = channel.astype(np.float32)
    low, high = np.percentile(src, [float(low_percentile), float(high_percentile)])
    if high <= low + 1e-6:
        return np.clip(src, 0, 255).astype(np.uint8)
    return np.clip((src - low) * (255.0 / (high - low)), 0, 255).astype(np.uint8)


def _select_gray_channel(arr: np.ndarray, mode: str) -> np.ndarray:
    import cv2

    channels = {
        "red": arr[:, :, 0],
        "green": arr[:, :, 1],
        "blue": arr[:, :, 2],
        "max": arr.max(axis=2),
        "luma": cv2.cvtColor(arr, cv2.COLOR_RGB2GRAY),
        "hsv_v": cv2.cvtColor(arr, cv2.COLOR_RGB2HSV)[:, :, 2],
        "lab_l": cv2.cvtColor(arr, cv2.COLOR_RGB2LAB)[:, :, 0],
    }
    if mode == "best_contrast":
        candidates = [channels[key] for key in ("luma", "red", "green", "blue")]
        scores = [float(np.percentile(channel, 95) - np.percentile(channel, 5)) for channel in candidates]
        return candidates[int(np.argmax(scores))]
    if mode not in channels:
        raise ValueError(f"Unsupported grayscale channel: {mode}")
    return channels[mode]


def _retinex(channel: np.ndarray, sigmas: tuple[float, ...]) -> np.ndarray:
    import cv2

    src = channel.astype(np.float32) + 1.0
    response = np.zeros_like(src)
    for sigma in sigmas:
        illumination = cv2.GaussianBlur(src, (0, 0), float(sigma))
        response += np.log(src) - np.log(np.maximum(illumination, 1e-6))
    return _robust_rescale(response / len(sigmas), 1.0, 99.0)


def _homomorphic(channel: np.ndarray) -> np.ndarray:
    import cv2

    src = np.log1p(channel.astype(np.float32))
    rows, cols = src.shape
    yy, xx = np.ogrid[:rows, :cols]
    distance2 = (yy - rows / 2.0) ** 2 + (xx - cols / 2.0) ** 2
    cutoff = max(min(rows, cols) / 4.0, 2.0)
    low_gain, high_gain = 0.7, 1.4
    transfer = (high_gain - low_gain) * (1.0 - np.exp(-distance2 / (cutoff * cutoff))) + low_gain
    spectrum = np.fft.fftshift(np.fft.fft2(src))
    restored = np.real(np.fft.ifft2(np.fft.ifftshift(spectrum * transfer)))
    return _robust_rescale(np.expm1(restored), 1.0, 99.0)


def _local_contrast_normalize(channel: np.ndarray, sigma: float = 7.0) -> np.ndarray:
    import cv2

    src = channel.astype(np.float32)
    mean = cv2.GaussianBlur(src, (0, 0), sigma)
    mean_square = cv2.GaussianBlur(src * src, (0, 0), sigma)
    std = np.sqrt(np.maximum(mean_square - mean * mean, 0.0))
    normalized = 127.5 + 40.0 * (src - mean) / np.maximum(std, 10.0)
    return np.clip(normalized, 0, 255).astype(np.uint8)


def _adaptive_wiener(channel: np.ndarray, window: int) -> np.ndarray:
    import cv2

    src = channel.astype(np.float32)
    size = (int(window), int(window))
    mean = cv2.boxFilter(src, cv2.CV_32F, size, normalize=True, borderType=cv2.BORDER_REFLECT)
    mean_square = cv2.boxFilter(src * src, cv2.CV_32F, size, normalize=True, borderType=cv2.BORDER_REFLECT)
    variance = np.maximum(mean_square - mean * mean, 0.0)
    noise_variance = float(np.mean(variance))
    gain = np.maximum(variance - noise_variance, 0.0) / np.maximum(variance, 1e-6)
    return np.clip(mean + gain * (src - mean), 0, 255).astype(np.uint8)


def _opencv_preprocess(image: Image.Image, cfg: PreprocessingConfig) -> Image.Image:
    import cv2

    arr = np.asarray(image.convert("RGB"))
    work = _select_gray_channel(arr, cfg.gray_channel) if cfg.grayscale else arr.copy()

    if cfg.gamma != 1.0:
        lut = np.clip((np.arange(256, dtype=np.float32) / 255.0) ** float(cfg.gamma) * 255.0, 0, 255)
        work = cv2.LUT(work, lut.astype(np.uint8))

    if cfg.percentile_low is not None and cfg.percentile_high is not None:
        if work.ndim == 2:
            work = _robust_rescale(work, cfg.percentile_low, cfg.percentile_high)
        else:
            work = np.stack(
                [_robust_rescale(work[:, :, idx], cfg.percentile_low, cfg.percentile_high) for idx in range(3)],
                axis=-1,
            )

    if cfg.illumination != "none":
        if work.ndim != 2:
            raise ValueError("Illumination normalization currently requires grayscale input")
        if cfg.illumination == "retinex":
            work = _retinex(work, (float(cfg.illumination_sigma),))
        elif cfg.illumination == "multiscale_retinex":
            work = _retinex(work, (5.0, 15.0, 40.0))
        elif cfg.illumination == "homomorphic":
            work = _homomorphic(work)
        elif cfg.illumination == "local_norm":
            work = _local_contrast_normalize(work)
        else:
            raise ValueError(f"Unsupported illumination method: {cfg.illumination}")

    if cfg.autocontrast:
        work = np.asarray(ImageOps.autocontrast(Image.fromarray(work)))
    if cfg.histogram_equalization:
        if work.ndim == 2:
            work = cv2.equalizeHist(work)
        else:
            lab = cv2.cvtColor(work, cv2.COLOR_RGB2LAB)
            lab[:, :, 0] = cv2.equalizeHist(lab[:, :, 0])
            work = cv2.cvtColor(lab, cv2.COLOR_LAB2RGB)
    if cfg.clahe_clip_limit is not None:
        clahe = cv2.createCLAHE(
            clipLimit=float(cfg.clahe_clip_limit),
            tileGridSize=(int(cfg.clahe_tile_size), int(cfg.clahe_tile_size)),
        )
        if work.ndim == 2:
            work = clahe.apply(work)
        else:
            lab = cv2.cvtColor(work, cv2.COLOR_RGB2LAB)
            lab[:, :, 0] = clahe.apply(lab[:, :, 0])
            work = cv2.cvtColor(lab, cv2.COLOR_LAB2RGB)

    if cfg.denoise == "gaussian":
        work = cv2.GaussianBlur(work, (0, 0), float(cfg.gaussian_sigma))
    elif cfg.denoise == "median":
        work = cv2.medianBlur(work, _odd_at_least(cfg.median_ksize))
    elif cfg.denoise == "bilateral":
        work = cv2.bilateralFilter(
            work,
            int(cfg.bilateral_d),
            float(cfg.bilateral_sigma_color),
            float(cfg.bilateral_sigma_space),
        )
    elif cfg.denoise == "wavelet_haar":
        if work.ndim == 2:
            work = _haar_soft_threshold(work)
        else:
            work = np.stack([_haar_soft_threshold(work[:, :, idx]) for idx in range(3)], axis=-1)
    elif cfg.denoise == "nlm":
        if work.ndim == 2:
            work = cv2.fastNlMeansDenoising(work, None, float(cfg.nlm_h), 7, 21)
        else:
            work = cv2.fastNlMeansDenoisingColored(work, None, float(cfg.nlm_h), float(cfg.nlm_h), 7, 21)
    elif cfg.denoise == "wiener":
        window = _odd_at_least(cfg.wiener_ksize)
        if work.ndim == 2:
            work = _adaptive_wiener(work, window)
        else:
            work = np.stack(
                [_adaptive_wiener(work[:, :, idx], window) for idx in range(3)], axis=-1
            )
    elif cfg.denoise != "none":
        raise ValueError(f"Unsupported denoise method: {cfg.denoise}")

    if cfg.sharpen_alpha > 0:
        alpha = float(cfg.sharpen_alpha)
        if cfg.sharpen_method == "unsharp":
            blur = cv2.GaussianBlur(work, (0, 0), float(cfg.sharpen_sigma))
            work = cv2.addWeighted(work, 1.0 + alpha, blur, -alpha, 0)
        elif cfg.sharpen_method == "laplacian":
            laplacian = cv2.Laplacian(work, cv2.CV_32F, ksize=3)
            work = np.clip(work.astype(np.float32) - alpha * laplacian, 0, 255).astype(np.uint8)
        elif cfg.sharpen_method == "dog":
            small = cv2.GaussianBlur(work, (0, 0), float(cfg.sharpen_sigma))
            large = cv2.GaussianBlur(work, (0, 0), float(cfg.sharpen_sigma_large))
            detail = small.astype(np.float32) - large.astype(np.float32)
            work = np.clip(work.astype(np.float32) + alpha * detail, 0, 255).astype(np.uint8)
        else:
            raise ValueError(f"Unsupported sharpen method: {cfg.sharpen_method}")

    if cfg.morphology != "none":
        kernel_width = cfg.morphology_kernel_width or _odd_at_least(cfg.morphology_ksize)
        kernel_height = cfg.morphology_kernel_height or _odd_at_least(cfg.morphology_ksize)
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (int(kernel_width), int(kernel_height)))
        if cfg.morphology == "close":
            work = cv2.morphologyEx(work, cv2.MORPH_CLOSE, kernel)
        elif cfg.morphology == "blackhat":
            blackhat = cv2.morphologyEx(work, cv2.MORPH_BLACKHAT, kernel)
            work = cv2.addWeighted(work, 1.0, blackhat, -float(cfg.morphology_strength), 0)
        elif cfg.morphology == "gradient":
            gradient = cv2.morphologyEx(work, cv2.MORPH_GRADIENT, kernel)
            work = cv2.addWeighted(work, 1.0, gradient, float(cfg.morphology_strength), 0)
        else:
            raise ValueError(f"Unsupported morphology method: {cfg.morphology}")

    if cfg.threshold != "none":
        gray = work if work.ndim == 2 else cv2.cvtColor(work, cv2.COLOR_RGB2GRAY)
        if cfg.threshold == "otsu":
            _level, work = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        elif cfg.threshold == "adaptive":
            work = cv2.adaptiveThreshold(
                gray,
                255,
                cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                cv2.THRESH_BINARY,
                _odd_at_least(cfg.adaptive_block_size),
                int(cfg.adaptive_c),
            )
        else:
            raise ValueError(f"Unsupported threshold method: {cfg.threshold}")

    if work.ndim == 2:
        work = np.repeat(work[:, :, None], 3, axis=2)
    return Image.fromarray(work.astype(np.uint8), mode="RGB")


def preprocess_plate_image(image: Image.Image, cfg: PreprocessingConfig | str | None = None) -> Image.Image:
    cfg = DEFAULT_CONFIG if cfg is None else get_preprocessing_config(cfg) if isinstance(cfg, str) else cfg
    if cfg.name == RAW_CONFIG.name:
        return image.convert("RGB")
    try:
        return _opencv_preprocess(image, cfg)
    except ImportError:
        gray = ImageOps.grayscale(image)
        if cfg.autocontrast or cfg.histogram_equalization or cfg.clahe_clip_limit is not None:
            gray = ImageOps.autocontrast(gray)
        if cfg.sharpen_alpha > 0:
            gray = ImageEnhance.Sharpness(gray).enhance(1.0 + cfg.sharpen_alpha)
        return Image.merge("RGB", (gray, gray, gray))


def iter_named_configs(names: Iterable[str] | None = None) -> list[PreprocessingConfig]:
    if names is None:
        return list(SWEEP_CONFIGS)
    return [get_preprocessing_config(name) for name in names]
