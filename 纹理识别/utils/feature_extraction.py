from __future__ import annotations

import math
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
from skimage.feature import local_binary_pattern

from utils.image_utils import (
    build_valid_mask,
    erode_mask,
    get_masked_values,
    iter_image_paths,
    load_rgb_image,
    masked_max,
    masked_mean,
    masked_percentile,
    masked_ratio,
    masked_std,
    safe_ratio,
)

CLASS_TO_LABEL = {"glint": 0, "fire": 1}
METADATA_COLUMNS = ["image_path", "file_name", "class_name", "label"]


def _feature_name(prefix: str, name: str) -> str:
    return f"{prefix}{name}" if prefix else name


def _mask_bbox(mask: np.ndarray) -> tuple[int, int, int, int]:
    if not mask.any():
        height, width = mask.shape
        return 0, height, 0, width

    ys, xs = np.where(mask)
    return int(ys.min()), int(ys.max()) + 1, int(xs.min()), int(xs.max()) + 1


def _resize_gray_and_mask(gray: np.ndarray, mask: np.ndarray, size: tuple[int, int]) -> tuple[np.ndarray, np.ndarray]:
    resized_gray = cv2.resize(gray, size, interpolation=cv2.INTER_AREA)
    resized_mask = cv2.resize(mask.astype(np.uint8), size, interpolation=cv2.INTER_NEAREST) > 0
    return resized_gray.astype(np.uint8), resized_mask


def _center_crop_region(array: np.ndarray, mask: np.ndarray, crop_size: int = 256) -> tuple[np.ndarray, np.ndarray]:
    height, width = mask.shape
    crop_size = min(crop_size, height, width)
    y0 = max(0, (height - crop_size) // 2)
    x0 = max(0, (width - crop_size) // 2)
    y1 = y0 + crop_size
    x1 = x0 + crop_size
    return array[y0:y1, x0:x1], mask[y0:y1, x0:x1]


def _extract_rgb_channel_stats(image_rgb: np.ndarray, mask: np.ndarray, prefix: str = "") -> dict[str, float]:
    r = image_rgb[:, :, 0].astype(np.float32)
    g = image_rgb[:, :, 1].astype(np.float32)
    b = image_rgb[:, :, 2].astype(np.float32)
    return {
        _feature_name(prefix, "R_mean"): masked_mean(r, mask),
        _feature_name(prefix, "G_mean"): masked_mean(g, mask),
        _feature_name(prefix, "B_mean"): masked_mean(b, mask),
        _feature_name(prefix, "R_std"): masked_std(r, mask),
        _feature_name(prefix, "G_std"): masked_std(g, mask),
        _feature_name(prefix, "B_std"): masked_std(b, mask),
    }


def _extract_hsv_channel_stats(image_rgb: np.ndarray, mask: np.ndarray, prefix: str = "") -> dict[str, float]:
    hsv = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2HSV)
    h = hsv[:, :, 0].astype(np.float32)
    s = hsv[:, :, 1].astype(np.float32)
    v = hsv[:, :, 2].astype(np.float32)
    return {
        _feature_name(prefix, "H_mean"): masked_mean(h, mask),
        _feature_name(prefix, "S_mean"): masked_mean(s, mask),
        _feature_name(prefix, "V_mean"): masked_mean(v, mask),
        _feature_name(prefix, "H_std"): masked_std(h, mask),
        _feature_name(prefix, "S_std"): masked_std(s, mask),
        _feature_name(prefix, "V_std"): masked_std(v, mask),
    }


def build_highlight_mask(gray: np.ndarray, mask: np.ndarray, percentile: float = 90.0) -> np.ndarray:
    """Extract the high-brightness core where the class-specific structure is strongest.

    The brightest pixels usually correspond to the most discriminative regions:
    flame cores, reflective streaks, and compact specular highlights.
    """

    values = get_masked_values(gray.astype(np.float32), mask)
    if values.size == 0:
        return np.zeros_like(mask, dtype=bool)

    threshold = float(np.percentile(values, percentile))
    highlight_mask = np.logical_and(gray >= threshold, mask)

    if highlight_mask.any():
        kernel = np.ones((3, 3), dtype=np.uint8)
        refined = cv2.morphologyEx(highlight_mask.astype(np.uint8), cv2.MORPH_OPEN, kernel)
        refined = cv2.morphologyEx(refined, cv2.MORPH_CLOSE, kernel)
        if refined.any():
            highlight_mask = refined.astype(bool)

    return highlight_mask


def extract_color_features(image_rgb: np.ndarray, mask: np.ndarray) -> dict[str, float]:
    """Color statistics capture warm-vs-reflective bias and per-pixel color balance.

    Fire often shows stronger red dominance, while glint tends to be closer to
    white light or reflected environment colors.
    """

    r = image_rgb[:, :, 0].astype(np.float32)
    g = image_rgb[:, :, 1].astype(np.float32)
    b = image_rgb[:, :, 2].astype(np.float32)

    r_mean = masked_mean(r, mask)
    g_mean = masked_mean(g, mask)
    b_mean = masked_mean(b, mask)
    rgb_sum = r + g + b + 1e-6

    features = {
        "R_mean": r_mean,
        "G_mean": g_mean,
        "B_mean": b_mean,
        "R_std": masked_std(r, mask),
        "G_std": masked_std(g, mask),
        "B_std": masked_std(b, mask),
        "R_over_G": safe_ratio(r_mean, g_mean),
        "R_over_B": safe_ratio(r_mean, b_mean),
        "G_over_B": safe_ratio(g_mean, b_mean),
        "R_minus_G": float(r_mean - g_mean),
        "R_minus_B": float(r_mean - b_mean),
        "RGB_max_minus_min": float(max(r_mean, g_mean, b_mean) - min(r_mean, g_mean, b_mean)),
        "pixel_R_over_G_mean": masked_mean(r / (g + 1e-6), mask),
        "pixel_R_over_B_mean": masked_mean(r / (b + 1e-6), mask),
        "pixel_G_over_B_mean": masked_mean(g / (b + 1e-6), mask),
        "pixel_RG_normalized_diff_mean": masked_mean((r - g) / rgb_sum, mask),
        "pixel_RB_normalized_diff_mean": masked_mean((r - b) / rgb_sum, mask),
    }
    return features


def extract_hsv_features(image_rgb: np.ndarray, mask: np.ndarray) -> dict[str, float]:
    """HSV summarizes hue, saturation, and brightness more directly than RGB.

    This helps separate the saturated warm flame distribution from reflective
    highlights that may be bright but less color-stable.
    """

    hsv_stats = _extract_hsv_channel_stats(image_rgb, mask)
    return {
        "H_mean": hsv_stats["H_mean"],
        "S_mean": hsv_stats["S_mean"],
        "V_mean": hsv_stats["V_mean"],
        "S_std": hsv_stats["S_std"],
        "V_std": hsv_stats["V_std"],
    }


def extract_gray_features(gray: np.ndarray, mask: np.ndarray, prefix: str = "") -> dict[str, float]:
    """Gray-level brightness features describe highlight intensity and dynamic range.

    Flame and water glint can both be bright, but their percentile profile and
    bright-pixel occupancy often differ.
    """

    return {
        _feature_name(prefix, "gray_mean"): masked_mean(gray, mask),
        _feature_name(prefix, "gray_std"): masked_std(gray, mask),
        _feature_name(prefix, "gray_max"): masked_max(gray, mask),
        _feature_name(prefix, "gray_p95"): masked_percentile(gray, mask, 95),
        _feature_name(prefix, "gray_p99"): masked_percentile(gray, mask, 99),
        _feature_name(prefix, "bright_pixel_ratio"): masked_ratio(gray >= 220, mask),
    }


def extract_lbp_features(gray: np.ndarray, mask: np.ndarray, n_bins: int = 16, prefix: str = "") -> dict[str, float]:
    """LBP captures local micro-texture such as flame flicker or ripple-like glare.

    We erode the mask before reading LBP values so each center pixel keeps a
    valid neighborhood and does not mix in black padding.
    """

    radius = 1
    n_points = 8 * radius
    lbp = local_binary_pattern(gray, n_points, radius, method="default")
    valid_mask = erode_mask(mask, kernel_size=3)
    values = get_masked_values(lbp, valid_mask if valid_mask.any() else mask)

    if values.size == 0:
        hist = np.zeros(n_bins, dtype=np.float32)
    else:
        hist, _ = np.histogram(values, bins=n_bins, range=(0, 2**n_points))
        hist = hist.astype(np.float32)
        hist /= hist.sum() if hist.sum() > 0 else 1.0

    return {_feature_name(prefix, f"LBP_hist_{idx:02d}"): float(value) for idx, value in enumerate(hist)}


def extract_glcm_features(gray: np.ndarray, mask: np.ndarray, levels: int = 16, prefix: str = "") -> dict[str, float]:
    """GLCM measures second-order texture structure like smoothness and repetition.

    The custom implementation counts only pixel pairs whose two endpoints are
    both valid, which prevents padding from polluting co-occurrence statistics.
    """

    gray_quant = np.clip((gray.astype(np.float32) / 256.0 * levels).astype(np.int32), 0, levels - 1)
    glcm = np.zeros((levels, levels), dtype=np.float64)
    offsets = [(0, 1), (1, 0), (1, 1), (-1, 1)]
    height, width = gray_quant.shape

    for dy, dx in offsets:
        y_src_start = max(0, -dy)
        y_src_end = min(height, height - dy)
        x_src_start = max(0, -dx)
        x_src_end = min(width, width - dx)

        y_dst_start = y_src_start + dy
        y_dst_end = y_src_end + dy
        x_dst_start = x_src_start + dx
        x_dst_end = x_src_end + dx

        src_mask = mask[y_src_start:y_src_end, x_src_start:x_src_end]
        dst_mask = mask[y_dst_start:y_dst_end, x_dst_start:x_dst_end]
        valid_pairs = np.logical_and(src_mask, dst_mask)
        if not valid_pairs.any():
            continue

        src_vals = gray_quant[y_src_start:y_src_end, x_src_start:x_src_end][valid_pairs]
        dst_vals = gray_quant[y_dst_start:y_dst_end, x_dst_start:x_dst_end][valid_pairs]
        np.add.at(glcm, (src_vals, dst_vals), 1)
        np.add.at(glcm, (dst_vals, src_vals), 1)

    if glcm.sum() == 0:
        return {
            _feature_name(prefix, "GLCM_contrast"): 0.0,
            _feature_name(prefix, "GLCM_homogeneity"): 0.0,
            _feature_name(prefix, "GLCM_energy"): 0.0,
            _feature_name(prefix, "GLCM_correlation"): 0.0,
        }

    p = glcm / glcm.sum()
    indices = np.arange(levels, dtype=np.float64)
    i_grid, j_grid = np.meshgrid(indices, indices, indexing="ij")
    diff_sq = (i_grid - j_grid) ** 2

    contrast = float(np.sum(p * diff_sq))
    homogeneity = float(np.sum(p / (1.0 + diff_sq)))
    asm = float(np.sum(p**2))
    energy = float(np.sqrt(asm))

    p_i = p.sum(axis=1)
    p_j = p.sum(axis=0)
    mu_i = float(np.sum(indices * p_i))
    mu_j = float(np.sum(indices * p_j))
    sigma_i = math.sqrt(float(np.sum(((indices - mu_i) ** 2) * p_i)))
    sigma_j = math.sqrt(float(np.sum(((indices - mu_j) ** 2) * p_j)))
    if sigma_i == 0.0 or sigma_j == 0.0:
        correlation = 0.0
    else:
        correlation = float(np.sum(((i_grid - mu_i) * (j_grid - mu_j) * p) / (sigma_i * sigma_j)))

    return {
        _feature_name(prefix, "GLCM_contrast"): contrast,
        _feature_name(prefix, "GLCM_homogeneity"): homogeneity,
        _feature_name(prefix, "GLCM_energy"): energy,
        _feature_name(prefix, "GLCM_correlation"): correlation,
    }


def extract_entropy_feature(gray: np.ndarray, mask: np.ndarray, prefix: str = "") -> dict[str, float]:
    """Entropy measures texture randomness, which is often higher in flickering fire."""

    values = get_masked_values(gray, mask)
    if values.size == 0:
        return {_feature_name(prefix, "entropy"): 0.0}

    hist, _ = np.histogram(values, bins=256, range=(0, 256))
    probs = hist.astype(np.float64)
    probs /= probs.sum() if probs.sum() > 0 else 1.0
    probs = probs[probs > 0]
    entropy = float(-np.sum(probs * np.log2(probs)))
    return {_feature_name(prefix, "entropy"): entropy}


def extract_edge_features(gray: np.ndarray, mask: np.ndarray, prefix: str = "") -> dict[str, float]:
    """Gradient features capture contour roughness and fine edge density.

    Fire tends to have unstable, irregular boundaries, while glint often forms
    sharper but more mirror-like highlight structures.
    """

    gray_float = gray.astype(np.float32)
    valid_mask = erode_mask(mask, kernel_size=3)
    analysis_mask = valid_mask if valid_mask.any() else mask

    sobel_x = cv2.Sobel(gray_float, cv2.CV_32F, 1, 0, ksize=3)
    sobel_y = cv2.Sobel(gray_float, cv2.CV_32F, 0, 1, ksize=3)
    sobel_mag = cv2.magnitude(sobel_x, sobel_y)
    sobel_vals = get_masked_values(sobel_mag, analysis_mask)

    laplacian = cv2.Laplacian(gray_float, cv2.CV_32F, ksize=3)
    lap_vals = get_masked_values(laplacian, analysis_mask)

    edges = cv2.Canny(gray.astype(np.uint8), threshold1=100, threshold2=200) > 0

    return {
        _feature_name(prefix, "Sobel_mean"): float(sobel_vals.mean()) if sobel_vals.size else 0.0,
        _feature_name(prefix, "Sobel_std"): float(sobel_vals.std()) if sobel_vals.size else 0.0,
        _feature_name(prefix, "Laplacian_variance"): float(lap_vals.var()) if lap_vals.size else 0.0,
        _feature_name(prefix, "Canny_edge_ratio"): masked_ratio(edges, analysis_mask),
    }


def extract_highlight_features(
    image_rgb: np.ndarray,
    gray: np.ndarray,
    mask: np.ndarray,
    highlight_mask: np.ndarray | None = None,
) -> dict[str, float]:
    """Focus on the bright core where flame and glint differ most in structure.

    This module isolates the top-brightness region and describes its color,
    texture, and edge pattern separately from the full target area.
    """

    if highlight_mask is None:
        highlight_mask = build_highlight_mask(gray, mask)

    features: dict[str, float] = {
        "highlight_pixel_ratio": safe_ratio(float(highlight_mask.sum()), float(mask.sum())),
    }

    if not highlight_mask.any():
        zero_features = {}
        zero_features.update(_extract_rgb_channel_stats(image_rgb, highlight_mask, prefix="highlight_"))
        zero_features.update(_extract_hsv_channel_stats(image_rgb, highlight_mask, prefix="highlight_"))
        zero_features.update(extract_lbp_features(gray, highlight_mask, n_bins=8, prefix="highlight_"))
        zero_features.update(extract_glcm_features(gray, highlight_mask, prefix="highlight_"))
        zero_features.update(extract_entropy_feature(gray, highlight_mask, prefix="highlight_"))
        zero_features.update(extract_edge_features(gray, highlight_mask, prefix="highlight_"))
        zero_features["highlight_gray_over_global_gray"] = 0.0
        zero_features["highlight_saturation_over_global_saturation"] = 0.0
        zero_features["highlight_entropy_minus_global_entropy"] = 0.0
        features.update(zero_features)
        return features

    full_hsv = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2HSV)
    full_saturation = full_hsv[:, :, 1].astype(np.float32)
    highlight_entropy = extract_entropy_feature(gray, highlight_mask, prefix="highlight_")["highlight_entropy"]
    global_entropy = extract_entropy_feature(gray, mask)["entropy"]

    features.update(_extract_rgb_channel_stats(image_rgb, highlight_mask, prefix="highlight_"))
    features.update(_extract_hsv_channel_stats(image_rgb, highlight_mask, prefix="highlight_"))
    features.update(extract_lbp_features(gray, highlight_mask, n_bins=8, prefix="highlight_"))
    features.update(extract_glcm_features(gray, highlight_mask, prefix="highlight_"))
    features["highlight_entropy"] = highlight_entropy
    features.update(extract_edge_features(gray, highlight_mask, prefix="highlight_"))
    features["highlight_gray_over_global_gray"] = safe_ratio(masked_mean(gray, highlight_mask), masked_mean(gray, mask))
    features["highlight_saturation_over_global_saturation"] = safe_ratio(
        masked_mean(full_saturation, highlight_mask),
        masked_mean(full_saturation, mask),
    )
    features["highlight_entropy_minus_global_entropy"] = float(highlight_entropy - global_entropy)
    return features


def extract_block_features(image_rgb: np.ndarray, gray: np.ndarray, mask: np.ndarray, grid_size: int = 4) -> dict[str, float]:
    """Measure spatial unevenness so we can separate turbulent flames from smoother glints.

    Block-wise variability captures whether brightness and color ratios stay
    coherent across the target or fluctuate strongly from patch to patch.
    """

    hsv = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2HSV)
    value_channel = hsv[:, :, 2].astype(np.float32)
    height, width = gray.shape

    gray_block_values: list[float] = []
    rg_block_values: list[float] = []
    value_block_values: list[float] = []
    block_coverages: list[float] = []

    for row_idx in range(grid_size):
        y0 = row_idx * height // grid_size
        y1 = (row_idx + 1) * height // grid_size
        for col_idx in range(grid_size):
            x0 = col_idx * width // grid_size
            x1 = (col_idx + 1) * width // grid_size

            block_mask = mask[y0:y1, x0:x1]
            valid_fraction = float(block_mask.mean())
            if valid_fraction < 0.05:
                continue

            block_gray = gray[y0:y1, x0:x1]
            block_r = image_rgb[y0:y1, x0:x1, 0].astype(np.float32)
            block_g = image_rgb[y0:y1, x0:x1, 1].astype(np.float32)
            block_v = value_channel[y0:y1, x0:x1]

            gray_block_values.append(masked_mean(block_gray, block_mask))
            rg_block_values.append(safe_ratio(masked_mean(block_r, block_mask), masked_mean(block_g, block_mask)))
            value_block_values.append(masked_mean(block_v, block_mask))
            block_coverages.append(valid_fraction)

    def summarize(values: list[float], base_name: str) -> dict[str, float]:
        if not values:
            return {
                f"{base_name}_var": 0.0,
                f"{base_name}_range": 0.0,
            }
        arr = np.asarray(values, dtype=np.float32)
        return {
            f"{base_name}_var": float(arr.var()),
            f"{base_name}_range": float(arr.max() - arr.min()),
        }

    features = {
        "block_nonempty_ratio": safe_ratio(float(len(gray_block_values)), float(grid_size * grid_size)),
        "block_coverage_std": float(np.std(block_coverages)) if block_coverages else 0.0,
    }
    features.update(summarize(gray_block_values, "block_gray"))
    features.update(summarize(rg_block_values, "block_rg_ratio"))
    features.update(summarize(value_block_values, "block_value"))
    return features


def extract_hog_features(gray: np.ndarray, mask: np.ndarray, n_bins: int = 9) -> dict[str, float]:
    """Describe gradient direction preference to separate line-like glints from flames.

    Specular reflections often show stronger dominant orientations, while fire
    tends to distribute gradients more diffusely.
    """

    gray_float = gray.astype(np.float32)
    valid_mask = erode_mask(mask, kernel_size=3)
    analysis_mask = valid_mask if valid_mask.any() else mask

    sobel_x = cv2.Sobel(gray_float, cv2.CV_32F, 1, 0, ksize=3)
    sobel_y = cv2.Sobel(gray_float, cv2.CV_32F, 0, 1, ksize=3)
    magnitude = cv2.magnitude(sobel_x, sobel_y)
    angle = (np.degrees(np.arctan2(sobel_y, sobel_x)) + 180.0) % 180.0

    mag_values = get_masked_values(magnitude, analysis_mask)
    angle_values = get_masked_values(angle, analysis_mask)

    if mag_values.size == 0:
        hist = np.zeros(n_bins, dtype=np.float32)
        dominant_ratio = 0.0
        direction_entropy = 0.0
        grad_ratio = 0.0
    else:
        hist, _ = np.histogram(angle_values, bins=n_bins, range=(0, 180), weights=mag_values)
        hist = hist.astype(np.float32)
        hist /= hist.sum() if hist.sum() > 0 else 1.0
        dominant_ratio = float(hist.max())
        nonzero = hist[hist > 0]
        direction_entropy = float(-np.sum(nonzero * np.log2(nonzero)))
        grad_ratio = safe_ratio(np.abs(get_masked_values(sobel_x, analysis_mask)).sum(), np.abs(get_masked_values(sobel_y, analysis_mask)).sum())

    features = {_feature_name("", f"HOG_hist_{idx:02d}"): float(value) for idx, value in enumerate(hist)}
    features["Sobel_x_over_y"] = float(grad_ratio)
    features["dominant_direction_ratio"] = dominant_ratio
    features["gradient_direction_entropy"] = direction_entropy
    return features


def extract_shape_features(
    gray: np.ndarray,
    mask: np.ndarray,
    highlight_mask: np.ndarray | None = None,
) -> dict[str, float]:
    """Measure geometry of bright cores because flame highlights are often fragmented.

    Glint regions are more likely to form compact or strip-like structures,
    while fire frequently produces irregular, multi-part bright blobs.
    """

    if highlight_mask is None:
        highlight_mask = build_highlight_mask(gray, mask)

    features = {
        "highlight_component_count": 0.0,
        "highlight_total_area_ratio": 0.0,
        "highlight_largest_area_ratio": 0.0,
        "highlight_bbox_aspect_ratio": 0.0,
        "highlight_circularity": 0.0,
        "highlight_boundary_complexity": 0.0,
    }
    if not highlight_mask.any():
        return features

    components_mask = highlight_mask.astype(np.uint8)
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(components_mask, connectivity=8)
    component_count = max(0, num_labels - 1)
    if component_count == 0:
        return features

    component_areas = stats[1:, cv2.CC_STAT_AREA].astype(np.float32)
    total_valid_area = float(mask.sum())
    largest_index = 1 + int(np.argmax(component_areas))
    largest_area = float(stats[largest_index, cv2.CC_STAT_AREA])
    x = int(stats[largest_index, cv2.CC_STAT_LEFT])
    y = int(stats[largest_index, cv2.CC_STAT_TOP])
    w = int(stats[largest_index, cv2.CC_STAT_WIDTH])
    h = int(stats[largest_index, cv2.CC_STAT_HEIGHT])

    largest_component = (labels == largest_index).astype(np.uint8)
    contours, _ = cv2.findContours(largest_component, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if contours:
        contour = max(contours, key=cv2.contourArea)
        perimeter = float(cv2.arcLength(contour, True))
        circularity = float(4.0 * math.pi * largest_area / ((perimeter**2) + 1e-6))
        complexity = float((perimeter**2) / (largest_area + 1e-6))
    else:
        circularity = 0.0
        complexity = 0.0

    features.update(
        {
            "highlight_component_count": float(component_count),
            "highlight_total_area_ratio": safe_ratio(float(component_areas.sum()), total_valid_area),
            "highlight_largest_area_ratio": safe_ratio(largest_area, total_valid_area),
            "highlight_bbox_aspect_ratio": safe_ratio(float(max(w, h)), float(min(w, h))),
            "highlight_circularity": circularity,
            "highlight_boundary_complexity": complexity,
        }
    )
    return features


def apply_clahe(gray: np.ndarray, mask: np.ndarray, clip_limit: float = 2.0, tile_grid_size: tuple[int, int] = (8, 8)) -> np.ndarray:
    """Boost local contrast inside the valid region so weak textures become measurable.

    CLAHE often reveals subtle ripple or flame-edge details that are muted in
    the original grayscale image.
    """

    if not mask.any():
        return np.zeros_like(gray, dtype=np.uint8)

    y0, y1, x0, x1 = _mask_bbox(mask)
    roi = gray[y0:y1, x0:x1].astype(np.uint8)
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_grid_size)
    enhanced_roi = clahe.apply(roi)

    enhanced = np.zeros_like(gray, dtype=np.uint8)
    enhanced[y0:y1, x0:x1] = enhanced_roi
    enhanced[~mask] = 0
    return enhanced


def extract_multiscale_features(gray: np.ndarray, mask: np.ndarray) -> dict[str, float]:
    """Aggregate texture cues at multiple scales so the model sees coarse and fine structure.

    Downsampled views emphasize global organization, while the center crop keeps
    local detail around the target core.
    """

    features: dict[str, float] = {}

    gray_256, mask_256 = _resize_gray_and_mask(gray, mask, (256, 256))
    center_gray, center_mask = _center_crop_region(gray, mask, crop_size=256)

    for prefix, region_gray, region_mask in [
        ("ms256_", gray_256, mask_256),
        ("center_", center_gray, center_mask),
    ]:
        features.update(extract_gray_features(region_gray, region_mask, prefix=prefix))
        features.update(extract_lbp_features(region_gray, region_mask, n_bins=8, prefix=prefix))
        features.update(extract_glcm_features(region_gray, region_mask, prefix=prefix))

    return features


def extract_feature_vector(image_rgb: np.ndarray, mask: np.ndarray) -> dict[str, float]:
    """Combine all handcrafted features into one model-ready vector."""

    gray = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2GRAY)
    highlight_mask = build_highlight_mask(gray, mask)
    clahe_gray = apply_clahe(gray, mask)

    features: dict[str, float] = {}
    features.update(extract_color_features(image_rgb, mask))
    features.update(extract_hsv_features(image_rgb, mask))
    features.update(extract_gray_features(gray, mask))
    features.update(extract_lbp_features(gray, mask))
    features.update(extract_glcm_features(gray, mask))
    features.update(extract_entropy_feature(gray, mask))
    features.update(extract_edge_features(gray, mask))

    features.update(extract_highlight_features(image_rgb, gray, mask, highlight_mask=highlight_mask))
    features.update(extract_block_features(image_rgb, gray, mask))
    features.update(extract_hog_features(gray, mask))
    features.update(extract_shape_features(gray, mask, highlight_mask=highlight_mask))

    features.update(extract_lbp_features(clahe_gray, mask, n_bins=8, prefix="clahe_"))
    features.update(extract_glcm_features(clahe_gray, mask, prefix="clahe_"))
    features.update(extract_edge_features(clahe_gray, mask, prefix="clahe_"))

    features.update(extract_multiscale_features(gray, mask))
    return features


def scan_dataset(data_root: Path) -> list[dict[str, object]]:
    """Scan class folders and attach labels from folder names automatically."""

    samples: list[dict[str, object]] = []
    for class_name, label in CLASS_TO_LABEL.items():
        class_dir = data_root / class_name
        if not class_dir.exists():
            raise FileNotFoundError(f"Missing class folder: {class_dir}")

        for image_path in iter_image_paths(class_dir):
            samples.append(
                {
                    "image_path": image_path,
                    "file_name": image_path.name,
                    "class_name": class_name,
                    "label": label,
                }
            )

    if not samples:
        raise ValueError(f"No images found under {data_root}")
    return samples


def build_feature_dataframe(data_root: Path) -> pd.DataFrame:
    """Extract features for all images and return a pandas table."""

    records: list[dict[str, object]] = []
    for sample in scan_dataset(data_root):
        image = load_rgb_image(Path(sample["image_path"]))
        mask = build_valid_mask(image)
        feature_record = extract_feature_vector(image, mask)
        feature_record.update(
            {
                "image_path": str(sample["image_path"]),
                "file_name": sample["file_name"],
                "class_name": sample["class_name"],
                "label": sample["label"],
            }
        )
        records.append(feature_record)

    df = pd.DataFrame(records)
    ordered_columns = METADATA_COLUMNS + [col for col in df.columns if col not in METADATA_COLUMNS]
    return df[ordered_columns]


def get_feature_columns(df: pd.DataFrame) -> list[str]:
    return [column for column in df.columns if column not in METADATA_COLUMNS]
