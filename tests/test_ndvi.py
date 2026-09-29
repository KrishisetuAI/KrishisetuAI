"""Tests for NDVI evidence engine (Prompt 8)."""

import numpy as np
import pytest

from krishisethu.evidence.ndvi import (
    NDVISample,
    calculate_ndvi,
    compute_anomaly,
)


def test_ndvi_bounds():
    """NDVI values must stay within [-1.0, 1.0]."""
    b08 = np.array([[1000, 2000], [3000, 4000]], dtype=np.float32)
    b04 = np.array([[500, 1000], [1500, 2000]], dtype=np.float32)
    
    mean, std, ratio = calculate_ndvi(b08, b04)
    
    assert -1.0 <= mean <= 1.0
    assert std >= 0.0
    assert 0.0 <= ratio <= 1.0


def test_ndvi_division_by_zero():
    """Division by zero (B08 + B04 == 0) must be handled safely."""
    b08 = np.array([[0, 100], [0, 200]], dtype=np.float32)
    b04 = np.array([[0, 50], [0, 100]], dtype=np.float32)
    
    mean, std, ratio = calculate_ndvi(b08, b04)
    
    # Should not raise; NaN values are masked out
    assert np.isfinite(mean) or mean == 0.0
    assert np.isfinite(std) or std == 0.0
    assert 0.0 <= ratio <= 1.0


def test_scl_cloud_masking():
    """SCL cloud masking must exclude classes 3, 8, 9, 10."""
    b08 = np.array([[1000, 2000, 3000], [4000, 5000, 6000]], dtype=np.float32)
    b04 = np.array([[500, 1000, 1500], [2000, 2500, 3000]], dtype=np.float32)
    scl = np.array([
        [3, 4, 5],   # 3 = cloud shadow (should be masked)
        [8, 9, 10],  # 8, 9, 10 = cloud (should be masked)
    ], dtype=np.uint8)
    
    mean_masked, std_masked, ratio_masked = calculate_ndvi(b08, b04, scl)
    mean_unmasked, std_unmasked, ratio_unmasked = calculate_ndvi(b08, b04, None)
    
    # With cloud masking, fewer valid pixels
    assert ratio_masked < ratio_unmasked
    # Mean should still be valid
    assert -1.0 <= mean_masked <= 1.0
    assert np.isfinite(mean_masked)


def test_sohna_flood_anomaly():
    """Sohna flood scenario: baseline 0.79, observed 0.44 -> delta ~ -0.35, severe."""
    baseline = 0.79
    observed = 0.44
    
    delta, severity = compute_anomaly(observed, baseline)
    
    assert abs(delta - (-0.35)) < 0.01
    assert severity == "severe"


def test_anomaly_classifications():
    """Test all anomaly severity classifications."""
    # Normal: drop < 15%
    delta, severity = compute_anomaly(0.70, 0.80)  # -0.10, drop = 12.5%
    assert severity == "normal"
    
    # Moderate: 15% <= drop <= 30%
    delta, severity = compute_anomaly(0.60, 0.80)  # -0.20, drop = 25%
    assert severity == "moderate"
    
    # Severe: drop > 30%
    delta, severity = compute_anomaly(0.50, 0.80)  # -0.30, drop = 37.5%
    assert severity == "severe"
    
    # Edge cases
    delta, severity = compute_anomaly(0.68, 0.80)  # -0.12, drop = 15%
    assert severity == "moderate"
    
    delta, severity = compute_anomaly(0.56, 0.80)  # -0.24, drop = 30%
    assert severity == "moderate"
    
    delta, severity = compute_anomaly(0.55, 0.80)  # -0.25, drop = 31.25%
    assert severity == "severe"


def test_anomaly_baseline_zero():
    """Anomaly with zero baseline should return normal."""
    delta, severity = compute_anomaly(0.5, 0.0)
    assert delta == 0.5
    assert severity == "normal"
    
    delta, severity = compute_anomaly(0.5, -0.1)
    assert delta == 0.6
    assert severity == "normal"


def test_ndvi_sample_dataclass():
    """NDVISample dataclass can be instantiated."""
    sample = NDVISample(
        date="2024-07-15",
        ndvi_mean=0.65,
        ndvi_std=0.12,
        valid_pixel_ratio=0.88,
        cloud_masked=True,
    )
    assert sample.date == "2024-07-15"
    assert sample.ndvi_mean == 0.65
    assert sample.ndvi_std == 0.12
    assert sample.valid_pixel_ratio == 0.88
    assert sample.cloud_masked is True