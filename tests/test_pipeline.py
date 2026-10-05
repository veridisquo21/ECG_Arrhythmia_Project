import numpy as np
import pytest

from src.preprocess import calculate_bpm, normalize_beat, peak_detection_metrics, segment_beats
from train_model import grouped_splits


def test_segment_includes_zero_boundary_and_has_fixed_length():
    beats = segment_beats(np.arange(10), [2, 8], window_size=2)
    assert beats.shape == (2, 4)
    assert np.array_equal(beats[0], [0, 1, 2, 3])


def test_constant_normalization_is_finite_zero():
    result = normalize_beat(np.ones(4))
    assert np.array_equal(result, np.zeros(4))


def test_bpm_requires_two_peaks():
    assert calculate_bpm([10], 100) is None
    assert calculate_bpm([0, 100, 200], 100) == pytest.approx(60.0)


def test_peak_metrics_match_with_tolerance():
    result = peak_detection_metrics([100, 200], [98, 205], 100)
    assert result["sensitivity"] == 1.0
    assert result["ppv"] == 1.0


def test_grouped_splits_have_no_record_overlap():
    y = np.tile(np.arange(4), 6)
    groups = np.repeat([str(i) for i in range(6)], 4)
    splits = grouped_splits(y, groups)
    split_groups = [set(groups[index]) for index in splits]
    assert not any(split_groups[i] & split_groups[j] for i in range(3) for j in range(i))
