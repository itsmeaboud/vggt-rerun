
from typing import Literal

import matplotlib.cm as cm
import numpy as np
from jaxtyping import Float

from scripts.inference import VGGTOutput

ColorMode = Literal['rgb', 'confidence']


def normalize(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype = np.float32)
    if values.size == 0:
        return values
    
    low = np.nanmin(values)
    high = np.nanmax(values)
    if not np.isfinite(low) or not np.isfinite(high) or high <= low:
        return np.zeros_like(values, dtype = np.float32)
    return (values - low) / (high - low)

def confidence_colors(confidence: Float[np.ndarray, "N"]):

    normalized = normalize(confidence)
    rgba = cm.viridis(normalized)
    return (rgba[:, :3] * 255).astype(np.uint8)


def confidence_image(confidence: Float[np.ndarray, "H W"]):
    normalized = normalize(confidence)
    rgba = cm.viridis(normalized)
    return (rgba[..., :3] * 255).astype(np.uint8)


def rgb_to_uint8(rgb: np.ndarray) -> np.ndarray:
    if rgb.dtype == np.uint8:
        return rgb
    return np.clip(rgb * 255.0, 0, 255).astype(np.uint8)

def point_cloud_to_frame(
        data: VGGTOutput,
        frame_idx: int,
        *,
        percentile: float | None = None,
        color_mode: ColorMode = 'rgb'
    ) -> tuple[np.ndarray, np.ndarray]:
    
    points = data.world_points[frame_idx].reshape(-1, 3)
    confidence = data.depth_conf[frame_idx].reshape(-1)

    mask = np.ones(confidence.shape, dtype = bool)
    if percentile is not None:
        threshold = np.nanpercentile(confidence, percentile)
        mask = confidence >= threshold
    
    points = points[mask]

    if color_mode == "confidence":
        colors = confidence_colors(confidence[mask])
    else:
        colors = rgb_to_uint8(data.images[frame_idx].reshape(-1, 3)[mask])

    return points, colors
