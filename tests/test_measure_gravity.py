import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

from measure_gravity import free_fall_samples

from trackmania_rl.hybrid_track_profile import TrackProfileConfig


def test_free_fall_samples_recover_gravity_and_skip_ground():
    dt = 0.1
    t = np.arange(0, 3, dt)
    ground = np.full(15, 10.0)
    flight = 10.0 + 8.0 * t - 0.5 * 24.9 * t**2
    y = np.concatenate([ground, flight])

    samples = free_fall_samples(y, dt)

    assert len(samples) > 20
    assert np.median(samples) == pytest.approx(-24.9, abs=0.01)


def test_profile_uses_the_measured_gravity():
    assert TrackProfileConfig().gravity_mps2 == pytest.approx(24.9)
