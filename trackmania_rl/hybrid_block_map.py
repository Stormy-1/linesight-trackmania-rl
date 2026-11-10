"""Environment map built from a TrackMania map's block layout.

The game gives exact pose, so a robotics stack's localisation problem is
already solved. Its environment map is not: the controller only knows a recorded
centerline. Every ``.Challenge.Gbx`` file lists its blocks: name, grid cell,
rotation. This module turns that list into a queryable map.

Grid convention, calibrated on the A01 centerline (2026-11-10):
* a cell is 32 m x 8 m x 32 m, and world ``x = gx * 32``, ``z = gz * 32`` with no offset;
* a car at height ``y`` drives on top of the block at level ``floor(y / 8) - 1``.
  Walls, fences and borders that a car can hit sit one or two levels above that.

Block footprints inside a cell are not modelled yet. A block's kind comes only
from its name, which is coarse but covers all 242 block types used by the 65
Nations campaign maps.
"""

from __future__ import annotations

import math
import os
from dataclasses import dataclass
from pathlib import Path

import numpy as np

CELL_XZ_M = 32.0
CELL_Y_M = 8.0
DEFAULT_GAME_DIR = Path(r"C:\Program Files (x86)\Steam\steamapps\common\TrackMania Nations Forever")
CAMPAIGN_SUBDIR = Path("GameData") / "Tracks" / "Campaigns"

# A car can hit these. Checked in order, before the drivable keywords, so a
# "RoadDirtFenceStraight" is a hazard and a "TubeRoad" is drivable.
_DRIVABLE_EXCEPTIONS = ("TubeRoad", "PlatformRoad")
_HAZARD_KEYWORDS = (
    "Fence", "Wall", "Border", "Pillar", "Inflatable", "Tube", "Sculpt", "Control", "Deadend",
    "Fabric", "Airship", "Support",
)
_DRIVABLE_KEYWORDS = (
    "Road", "Platform", "Circuit", "Dirt", "Hill", "Grass", "Ramp", "Bump", "Loop", "Hole", "Stretch",
)
# Hazards that can touch a car driving at level L sit at these level offsets.
HAZARD_LEVEL_OFFSETS = (1, 2)


def block_kind(name: str) -> str:
    """Return ``"hazard"``, ``"drivable"`` or ``"other"`` for a block name."""

    if any(word in name for word in _DRIVABLE_EXCEPTIONS):
        return "drivable"
    if any(word in name for word in _HAZARD_KEYWORDS):
        return "hazard"
    if any(word in name for word in _DRIVABLE_KEYWORDS):
        return "drivable"
    return "other"


@dataclass(frozen=True)
class Block:
    name: str
    gx: int
    gy: int
    gz: int
    rotation: int

    @property
    def kind(self) -> str:
        return block_kind(self.name)


class BlockMap:
    """Blocks indexed by grid column for fast neighbourhood queries."""

    def __init__(self, blocks: list[Block]):
        self.blocks = tuple(blocks)
        columns: dict[tuple[int, int], list[Block]] = {}
        for block in self.blocks:
            columns.setdefault((block.gx, block.gz), []).append(block)
        self._columns = {key: tuple(value) for key, value in columns.items()}

    @staticmethod
    def cell_of(position: np.ndarray) -> tuple[int, int, int]:
        """Grid cell ``(gx, drive_level, gz)`` of a world position."""

        x, y, z = (float(v) for v in position[:3])
        return math.floor(x / CELL_XZ_M), math.floor(y / CELL_Y_M) - 1, math.floor(z / CELL_XZ_M)

    def blocks_in_column(self, gx: int, gz: int) -> tuple[Block, ...]:
        return self._columns.get((gx, gz), ())

    def hazards_near(self, position: np.ndarray, radius_cells: int = 1) -> list[Block]:
        """Hazard blocks at car height within ``radius_cells`` columns of ``position``."""

        gx, level, gz = self.cell_of(position)
        levels = {level + offset for offset in HAZARD_LEVEL_OFFSETS}
        hits = []
        for dx in range(-radius_cells, radius_cells + 1):
            for dz in range(-radius_cells, radius_cells + 1):
                hits.extend(
                    block
                    for block in self.blocks_in_column(gx + dx, gz + dz)
                    if block.kind == "hazard" and block.gy in levels
                )
        return hits


def find_campaign_challenge(map_stem: str, game_dir: Path | None = None) -> Path | None:
    """Locate ``<map_stem>.Challenge.Gbx`` in the installed campaigns, or ``None``."""

    root = Path(game_dir or os.environ.get("TMNF_GAME_DIR", DEFAULT_GAME_DIR)) / CAMPAIGN_SUBDIR
    if not root.is_dir():
        return None
    matches = sorted(root.rglob(f"{map_stem}.Challenge.Gbx"))
    return matches[0] if matches else None


def load_block_map(challenge_path: Path) -> BlockMap:
    """Parse a ``.Challenge.Gbx`` file into a :class:`BlockMap`."""

    from pygbx import Gbx, GbxType  # optional dependency, only needed here

    challenge = Gbx(str(challenge_path)).get_class_by_id(GbxType.CHALLENGE)
    if challenge is None:
        raise ValueError(f"{challenge_path} holds no challenge (map) data")
    return BlockMap(
        [
            Block(b.name, int(b.position.x), int(b.position.y), int(b.position.z), int(b.rotation))
            for b in challenge.blocks
        ]
    )
