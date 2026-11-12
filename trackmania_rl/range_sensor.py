"""Virtual range sensor (a simulated 2D lidar) over the map's block layout.

A robot senses obstacles with a lidar. We have no sensor, but we do have the map:
every block, its 32 m grid cell and its rotation (``hybrid_block_map``). What we
don't have is each obstacle block's *shape* inside its cell. A ``PlatformWallBorder``
is a ~3 m wall along one edge, and a start gate is an arch you drive through.

**Shapes are learned from data.** Pool the official replay ghosts of the training
maps. For every hazard block type, rotate each instance into one canonical frame
and mark every 2 m bin the car's centre ever passed through at the block's height.
Bins no pro ever passed through are solid. Because they're measured from car
centres, the solid region already includes the car's half-width: it's the
**configuration-space obstacle**, so a range of ~0 m means contact. Learning is per
block *type*, never per map, so the shapes transfer to unseen maps.

**Sensing:** for a car at drive level ``L`` (``BlockMap.cell_of``), obstacles are
hazard blocks at levels ``L + 1`` and ``L + 2`` (the car's own height slab and tall
walls). Rays are marched over a 1 m raster of those shapes.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from trackmania_rl.hybrid_block_map import CELL_XZ_M, HAZARD_LEVEL_OFFSETS, BlockMap

BIN_M = 1.0
BINS = int(CELL_XZ_M / BIN_M)  # 16 x 16 bins per cell
RASTER_M = 1.0
MIN_VISITS_FOR_SHAPE = 1  # a type no ghost ever entered stays fully solid
DEFAULT_RAY_ANGLES_RAD = tuple(np.deg2rad(np.arange(-90, 91, 15)))  # 13 rays, left to right
DEFAULT_MAX_RANGE_M = 60.0
RAY_STEP_M = 0.5
DENSIFY_STEP_M = 1.0


def canonical_local(local_x: np.ndarray, local_z: np.ndarray, rotation: int) -> tuple[np.ndarray, np.ndarray]:
    """Rotate in-cell coordinates (0..32 m) into the block type's canonical frame.

    Calibrated on ``PlatformWallBorder`` (HANDOVER §62): its wall lies on edge z=0 / x=32 /
    z=32 / x=0 for rotation 0 / 1 / 2 / 3, so all four map to canonical v=0.
    """

    size = CELL_XZ_M
    rotation %= 4
    if rotation == 0:
        return local_x, local_z
    if rotation == 1:
        return local_z, size - local_x
    if rotation == 2:
        return size - local_x, size - local_z
    return size - local_z, local_x


def world_from_canonical(u: np.ndarray, v: np.ndarray, rotation: int) -> tuple[np.ndarray, np.ndarray]:
    """Inverse of :func:`canonical_local`."""

    size = CELL_XZ_M
    rotation %= 4
    if rotation == 0:
        return u, v
    if rotation == 1:
        return size - v, u
    if rotation == 2:
        return size - u, size - v
    return v, size - u


def _obstacle_blocks(block_map: BlockMap, gx: int, gz: int, level: int):
    """(block, height offset) for hazard blocks in the car's height band."""

    return [
        (b, b.gy - level)
        for b in block_map.blocks_in_column(gx, gz)
        if b.kind == "hazard" and b.gy - level in HAZARD_LEVEL_OFFSETS
    ]


def shape_key(name: str, offset: int) -> str:
    """Shapes differ by height: a wall border one slab up is often driven under or past."""

    return f"{name}@{offset}"


def densify(positions: np.ndarray, step_m: float = DENSIFY_STEP_M) -> np.ndarray:
    """Linearly resample a ghost path (100 ms samples, ~6 m apart at speed) to ``step_m``."""

    positions = np.asarray(positions, dtype=float)
    if len(positions) < 2:
        return positions
    arc = np.concatenate(([0.0], np.cumsum(np.linalg.norm(np.diff(positions, axis=0), axis=1))))
    samples = np.arange(0.0, arc[-1], step_m)
    return np.stack([np.interp(samples, arc, positions[:, k]) for k in range(3)], axis=1)


@dataclass
class FootprintModel:
    """Per hazard type: how often a car centre visited each canonical 2 m bin."""

    visits: dict[str, np.ndarray] = field(default_factory=dict)

    def add_path(self, block_map: BlockMap, positions: np.ndarray) -> None:
        for position in densify(positions):
            gx, level, gz = block_map.cell_of(position)
            blocks = _obstacle_blocks(block_map, gx, gz, level)
            if not blocks:
                continue
            local_x = float(position[0]) - gx * CELL_XZ_M
            local_z = float(position[2]) - gz * CELL_XZ_M
            for block, offset in blocks:
                u, v = canonical_local(np.array(local_x), np.array(local_z), block.rotation)
                i = min(int(u // BIN_M), BINS - 1)
                j = min(int(v // BIN_M), BINS - 1)
                key = shape_key(block.name, offset)
                self.visits.setdefault(key, np.zeros((BINS, BINS), dtype=np.int64))[i, j] += 1

    def merged(self, other: FootprintModel, sign: int = 1) -> FootprintModel:
        """A new model with ``other``'s counts added (sign=1) or removed (sign=-1)."""

        names = set(self.visits) | set(other.visits)
        zero = np.zeros((BINS, BINS), dtype=np.int64)
        return FootprintModel(
            {n: np.maximum(self.visits.get(n, zero) + sign * other.visits.get(n, zero), 0) for n in names}
        )

    def solid_mask(self, key: str) -> np.ndarray:
        """(u, v) bins that are solid in the canonical frame. Unknown shapes are fully solid."""

        counts = self.visits.get(key)
        if counts is None or counts.sum() < MIN_VISITS_FOR_SHAPE:
            return np.ones((BINS, BINS), dtype=bool)
        return counts == 0

    def save(self, path) -> None:
        np.savez_compressed(path, **{name: counts for name, counts in self.visits.items()})

    @classmethod
    def load(cls, path) -> FootprintModel:
        with np.load(path) as data:
            return cls({name: data[name] for name in data.files})


class RangeSensor:
    """Ray casting against the learned obstacle shapes of one map."""

    def __init__(self, block_map: BlockMap, footprints: FootprintModel):
        self.block_map = block_map
        self.footprints = footprints
        gxs = [b.gx for b in block_map.blocks] or [0]
        gzs = [b.gz for b in block_map.blocks] or [0]
        self.origin_gx, self.origin_gz = min(gxs) - 1, min(gzs) - 1
        self.cells_x = max(gxs) - self.origin_gx + 2
        self.cells_z = max(gzs) - self.origin_gz + 2
        self._rasters: dict[int, np.ndarray] = {}
        cell_px = int(CELL_XZ_M / RASTER_M)
        centres = (np.arange(cell_px) + 0.5) * RASTER_M
        self._cell_x, self._cell_z = np.meshgrid(centres, centres, indexing="ij")

    def raster(self, level: int) -> np.ndarray:
        """1 m solid/free raster for a car driving at ``level``."""

        if level in self._rasters:
            return self._rasters[level]
        px = int(CELL_XZ_M / RASTER_M)
        grid = np.zeros((self.cells_x * px, self.cells_z * px), dtype=bool)
        for block in self.block_map.blocks:
            if block.kind != "hazard" or block.gy - level not in HAZARD_LEVEL_OFFSETS:
                continue
            mask = self.footprints.solid_mask(shape_key(block.name, block.gy - level))
            u, v = canonical_local(self._cell_x, self._cell_z, block.rotation)
            cell = mask[np.minimum((u // BIN_M).astype(int), BINS - 1), np.minimum((v // BIN_M).astype(int), BINS - 1)]
            ox = (block.gx - self.origin_gx) * px
            oz = (block.gz - self.origin_gz) * px
            grid[ox : ox + px, oz : oz + px] |= cell
        self._rasters[level] = grid
        return grid

    def prebuild(self) -> None:
        """Build every height level's raster up front, so a live step never stalls on one."""

        for block in self.block_map.blocks:
            if block.kind == "hazard":
                for offset in HAZARD_LEVEL_OFFSETS:
                    self.raster(block.gy - offset)

    def is_solid(self, x: np.ndarray, z: np.ndarray, level: int) -> np.ndarray:
        grid = self.raster(level)
        i = np.floor(np.asarray(x) / RASTER_M).astype(int) - self.origin_gx * int(CELL_XZ_M / RASTER_M)
        j = np.floor(np.asarray(z) / RASTER_M).astype(int) - self.origin_gz * int(CELL_XZ_M / RASTER_M)
        inside = (i >= 0) & (j >= 0) & (i < grid.shape[0]) & (j < grid.shape[1])
        out = np.zeros(np.shape(i), dtype=bool)
        out[inside] = grid[i[inside], j[inside]]
        return out

    def scan(
        self,
        position: np.ndarray,
        heading_rad: float,
        angles_rad: tuple[float, ...] = DEFAULT_RAY_ANGLES_RAD,
        max_range_m: float = DEFAULT_MAX_RANGE_M,
    ) -> np.ndarray:
        """Range to the first solid point along each ray (``max_range_m`` if none).

        ``heading_rad`` is the world yaw ``atan2(dz, dx)``. A positive ray angle points
        to the car's right, matching positive steering (it increases world yaw).
        """

        _, level, _ = self.block_map.cell_of(position)
        steps = np.arange(RAY_STEP_M, max_range_m + 1e-9, RAY_STEP_M)
        yaw = heading_rad + np.asarray(angles_rad)[:, None]
        xs = float(position[0]) + np.cos(yaw) * steps
        zs = float(position[2]) + np.sin(yaw) * steps
        hit = self.is_solid(xs, zs, level)
        first = np.where(hit.any(axis=1), hit.argmax(axis=1), -1)
        return np.where(first >= 0, steps[np.maximum(first, 0)] - RAY_STEP_M, max_range_m)


def ray_field_names(angles_rad: tuple[float, ...] = DEFAULT_RAY_ANGLES_RAD) -> tuple[str, ...]:
    """CSV column per ray, e.g. ``range_-90`` (left) ... ``range_+90`` (right)."""

    return tuple(f"range_{round(float(np.rad2deg(a))):+d}" for a in angles_rad)
