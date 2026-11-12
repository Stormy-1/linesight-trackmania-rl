import numpy as np
import pytest

from trackmania_rl.hybrid_block_map import (
    Block,
    BlockMap,
    block_kind,
    find_campaign_challenge,
    load_block_map,
)


@pytest.mark.parametrize(
    ("name", "kind"),
    [
        ("StadiumRoadDirtFenceStraight", "hazard"),
        ("StadiumPlatformWallBorder", "hazard"),
        ("StadiumCircuitBorderStraight", "hazard"),
        ("StadiumTubePillar", "hazard"),
        ("StadiumInflatableCactus", "hazard"),
        ("StadiumRoadMain", "drivable"),
        ("StadiumCircuitBase", "drivable"),
        ("StadiumDirtHill", "drivable"),
        ("StadiumPlatformRoad", "drivable"),
        ("StadiumTubeRoad", "drivable"),
        ("StadiumWater", "other"),
    ],
)
def test_block_kind(name, kind):
    assert block_kind(name) == kind


def test_world_to_cell_uses_32m_columns_and_drive_level():
    # The car drives on top of the block one level below floor(y / 8).
    assert BlockMap.cell_of(np.array([466.0, 40.6, 314.8])) == (14, 4, 9)
    assert BlockMap.cell_of(np.array([-1.0, 24.0, 31.9])) == (-1, 2, 0)


def test_hazards_at_car_height_only():
    fence = Block("StadiumRoadDirtFenceStraight", 14, 5, 9, 1)
    pillar_below = Block("StadiumTubePillar", 14, 2, 9, 0)
    road = Block("StadiumRoadMain", 14, 4, 9, 0)
    block_map = BlockMap([fence, pillar_below, road])
    hits = block_map.hazards_near(np.array([466.0, 40.6, 314.8]))
    assert hits == [fence]


def test_hazards_near_reaches_neighbour_columns():
    wall = Block("StadiumPlatformWallBorder", 15, 5, 9, 3)
    block_map = BlockMap([wall])
    position = np.array([466.0, 40.6, 314.8])
    assert block_map.hazards_near(position, radius_cells=0) == []
    assert block_map.hazards_near(position, radius_cells=1) == [wall]


def test_real_a05_fence_sits_where_the_car_crashes():
    path = find_campaign_challenge("A05-Race")
    if path is None:
        pytest.skip("TrackMania campaign files not installed")
    hits = load_block_map(path).hazards_near(np.array([464.0, 40.6, 315.0]), radius_cells=0)
    assert any("Fence" in block.name for block in hits)
