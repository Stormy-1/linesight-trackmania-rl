import math

import numpy as np
import pytest

from trackmania_rl.hybrid_block_map import Block, BlockMap
from trackmania_rl.range_sensor import (
    BIN_M,
    BINS,
    FootprintModel,
    RangeSensor,
    canonical_local,
    densify,
    shape_key,
    world_from_canonical,
)

WALL = "StadiumPlatformWallBorder"  # a hazard name
DRIVE_Y = 9.0  # drive level 0: a car at y=9 drives on blocks at level 0, hazards at levels 1-2


@pytest.mark.parametrize("rotation", [0, 1, 2, 3])
def test_canonical_frame_round_trips(rotation):
    x, z = np.array([3.0, 17.5, 30.0]), np.array([1.0, 8.0, 25.0])
    u, v = canonical_local(x, z, rotation)
    back_x, back_z = world_from_canonical(u, v, rotation)
    assert np.allclose(back_x, x) and np.allclose(back_z, z)


def test_wall_edge_maps_to_canonical_v0_for_every_rotation():
    # PlatformWallBorder edges (HANDOVER §62): rot 0 z=0, rot 1 x=32, rot 2 z=32, rot 3 x=0
    edges = {0: (16.0, 0.0), 1: (32.0, 16.0), 2: (16.0, 32.0), 3: (0.0, 16.0)}
    for rotation, (x, z) in edges.items():
        _, v = canonical_local(np.array(x), np.array(z), rotation)
        assert float(v) == pytest.approx(0.0)


def test_densify_fills_gaps_between_ghost_samples():
    path = np.array([[0.0, 0.0, 0.0], [10.0, 0.0, 0.0]])
    dense = densify(path, step_m=1.0)
    assert len(dense) == 10
    assert np.allclose(np.diff(dense[:, 0]), 1.0)


def _lane_model() -> FootprintModel:
    """A shape whose canonical u = 0..8 m is solid (never visited), the rest free."""

    counts = np.ones((BINS, BINS), dtype=np.int64)
    counts[:4, :] = 0
    return FootprintModel({shape_key(WALL, 1): counts})


def test_learned_shape_marks_unvisited_bins_solid():
    model = FootprintModel()
    block_map = BlockMap([Block(WALL, 0, 1, 0, 0)])
    path = np.array([[20.0, DRIVE_Y, 0.1], [20.0, DRIVE_Y, 33.0]])  # drives along u=20 through the cell

    model.add_path(block_map, path)

    mask = model.solid_mask(shape_key(WALL, 1))
    assert not mask[int(20.0 // BIN_M), :].any()  # u=20 m bin column was driven
    assert mask[0, :].all()  # u=0 m never driven -> solid
    assert model.solid_mask("Unknown@1").all()


def test_scan_measures_distance_to_a_learned_wall():
    block_map = BlockMap([Block(WALL, 1, 1, 0, 0)])  # cell x 32..64; solid x 32..40
    sensor = RangeSensor(block_map, _lane_model())
    car = np.array([20.0, DRIVE_Y, 16.0])

    ranges = sensor.scan(car, heading_rad=0.0, angles_rad=(0.0, math.pi / 2), max_range_m=40.0)

    assert ranges[0] == pytest.approx(12.0, abs=0.6)  # straight ahead (+x): wall face at x=32
    assert ranges[1] == pytest.approx(40.0)  # to the right (+z): nothing


def test_obstacles_only_count_in_the_cars_height_band():
    block_map = BlockMap([Block(WALL, 1, 5, 0, 0)])  # far above the car
    sensor = RangeSensor(block_map, _lane_model())
    ranges = sensor.scan(np.array([20.0, DRIVE_Y, 16.0]), 0.0, angles_rad=(0.0,), max_range_m=40.0)
    assert ranges[0] == pytest.approx(40.0)


def test_footprints_save_load_and_leave_one_out(tmp_path):
    a = FootprintModel({"x@1": np.full((BINS, BINS), 2, dtype=np.int64)})
    b = FootprintModel({"x@1": np.full((BINS, BINS), 2, dtype=np.int64)})
    both = a.merged(b)
    assert not both.merged(b, sign=-1).solid_mask("x@1").any()
    both.save(tmp_path / "f.npz")
    assert np.array_equal(FootprintModel.load(tmp_path / "f.npz").visits["x@1"], both.visits["x@1"])


def test_prebuild_and_ray_names():
    from trackmania_rl.range_sensor import ray_field_names

    sensor = RangeSensor(BlockMap([Block(WALL, 1, 3, 0, 0)]), _lane_model())
    sensor.prebuild()
    assert set(sensor._rasters) == {2, 1}  # hazard at level 3 matters to cars at levels 2 and 1
    names = ray_field_names()
    assert names[0] == "range_-90" and names[6] == "range_+0" and names[-1] == "range_+90"
