from dataclasses import dataclass
from typing import Optional
import numpy as np


@dataclass
class NDVISample:
    date: str
    ndvi_mean: float
    ndvi_std: float
    valid_pixel_ratio: float
    cloud_masked: bool


def calculate_ndvi(
    b08: np.ndarray,
    b04: np.ndarray,
    scl: Optional[np.ndarray] = None
) -> tuple[float, float, float]:
    """
    Calculate NDVI from Sentinel-2 B08 (NIR) and B04 (Red) bands with optional cloud masking.
    
    Args:
        b08: Near-infrared band (B08) array
        b04: Red band (B04) array
        scl: Scene Classification Layer (SCL) array for cloud masking
    
    Returns:
        Tuple of (mean_ndvi, std_ndvi, valid_pixel_ratio)
    """
    if scl is not None:
        cloud_classes = {3, 8, 9, 10}
        cloud_mask = np.isin(scl, list(cloud_classes))
        valid_mask = ~cloud_mask
    else:
        valid_mask = np.ones_like(b08, dtype=bool)
    
    valid_mask = valid_mask & (b08 > 0) & (b04 > 0)
    
    with np.errstate(divide='ignore', invalid='ignore'):
        ndvi = np.where(
            (b08 + b04) != 0,
            (b08 - b04) / (b08 + b04),
            np.nan
        )
    
    valid_ndvi = ndvi[valid_mask]
    
    if valid_ndvi.size == 0:
        return 0.0, 0.0, 0.0
    
    ndvi_mean = float(np.nanmean(valid_ndvi))
    ndvi_std = float(np.nanstd(valid_ndvi))
    valid_ratio = float(valid_ndvi.size) / float(np.prod(b08.shape))
    
    return ndvi_mean, ndvi_std, valid_ratio


def compute_anomaly(
    observed_ndvi: float,
    baseline_ndvi: float
) -> tuple[float, str]:
    """
    Compute NDVI anomaly and classify severity.
    
    Args:
        observed_ndvi: Current observed NDVI value
        baseline_ndvi: Baseline/reference NDVI value
    
    Returns:
        Tuple of (delta, severity_classification)
    """
    delta = observed_ndvi - baseline_ndvi
    
    if baseline_ndvi <= 0:
        return delta, "normal"
    
    drop_ratio = -delta / baseline_ndvi
    
    if drop_ratio < 0.15:
        severity = "normal"
    elif drop_ratio <= 0.30:
        severity = "moderate"
    else:
        severity = "severe"
    
    return delta, severity