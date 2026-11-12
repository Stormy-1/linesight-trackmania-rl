"""Build 0.5 m centerlines for every Nations campaign map from the game's official replays.

For each ``<stem>.Replay.gbx`` shipped with the game this writes
``maps/<stem>_official_0.5m_cl.npy`` and records it in ``maps/official_manifest.json``:
group, author time, respawns, the largest gap between ghost samples, and the line length.
Gaps and respawns flag lines that teleport and need a look before they're driven.

With ``--install`` it also copies each ``.Challenge.Gbx`` into the game's
``Tracks/Challenges/Official Maps`` folder (only missing files; never overwrites), so
``run_hybrid_controller_live.py --map <stem>`` can load it.

Usage: python scripts/build_official_centerlines.py [--install] [--force]
"""

from __future__ import annotations

import argparse
import contextlib
import io
import json
import shutil
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from config_files import config_copy
from trackmania_rl.geometry import extract_cp_distance_interval
from trackmania_rl.hybrid_block_map import CAMPAIGN_SUBDIR, DEFAULT_GAME_DIR
from trackmania_rl.map_loader import gbx_to_raw_pos_list

MAPS_DIR = ROOT / "maps"
MANIFEST = MAPS_DIR / "official_manifest.json"
SIDE_EFFECT_FILE = MAPS_DIR / "map.npy"  # extract_cp_distance_interval always writes here
OFFICIAL_MAPS_SUBDIR = Path("Tracks") / "Challenges" / "Official Maps"
TELEPORT_GAP_M = 25.0  # > 250 m/s between 100 ms samples is a teleport, not driving


def official_centerline_name(stem: str) -> str:
    return f"{stem}_official_0.5m_cl.npy"


def build_one(replay: Path) -> tuple[np.ndarray, dict]:
    from pygbx import Gbx, GbxType

    ghosts = Gbx(str(replay)).get_classes_by_ids([GbxType.CTN_GHOST])
    if not ghosts:
        raise ValueError("no ghost in replay")
    ghost = min(ghosts, key=lambda g: g.cp_times[-1])
    with contextlib.redirect_stdout(io.StringIO()):
        raw = np.array(gbx_to_raw_pos_list(replay))
        centerline = np.asarray(extract_cp_distance_interval(list(raw), 0.5, ROOT))
    SIDE_EFFECT_FILE.unlink()
    gaps = np.linalg.norm(np.diff(raw, axis=0), axis=1)
    info = {
        "author_time_s": ghost.race_time / 1000,
        "respawns": int(ghost.num_respawns),
        "max_sample_gap_m": round(float(gaps.max()), 2),
        "length_m": round(float(np.linalg.norm(np.diff(centerline, axis=0), axis=1).sum()), 1),
        "points": len(centerline),
    }
    info["suspect"] = info["respawns"] > 0 or info["max_sample_gap_m"] > TELEPORT_GAP_M
    return centerline, info


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--game-dir", type=Path, default=DEFAULT_GAME_DIR)
    parser.add_argument("--install", action="store_true", help="copy missing .Challenge.Gbx files into Official Maps")
    parser.add_argument("--force", action="store_true", help="rebuild centerlines that already exist")
    args = parser.parse_args()

    if SIDE_EFFECT_FILE.exists():
        raise SystemExit(f"{SIDE_EFFECT_FILE} exists; move it away first (the builder writes and removes it)")
    campaign = args.game_dir / CAMPAIGN_SUBDIR / "Nations"
    install_dir = config_copy.trackmania_base_path / OFFICIAL_MAPS_SUBDIR
    manifest = json.loads(MANIFEST.read_text()) if MANIFEST.exists() else {}

    for challenge in sorted(campaign.rglob("*.Challenge.Gbx")):
        stem = challenge.name.removesuffix(".Challenge.Gbx")
        entry = manifest.get(stem, {}) | {"group": challenge.parent.name}
        if args.install and not (install_dir / challenge.name).exists():
            shutil.copy2(challenge, install_dir / challenge.name)
        replay = challenge.with_name(f"{stem}.Replay.gbx")
        out = MAPS_DIR / official_centerline_name(stem)
        if not replay.exists():
            entry["error"] = "no official replay"
        elif out.exists() and not args.force and "length_m" in entry:
            pass
        else:
            try:
                centerline, info = build_one(replay)
                np.save(out, centerline.round(4))
                entry.update(info)
                entry.pop("error", None)
            except (ValueError, AssertionError) as error:
                entry["error"] = str(error) or type(error).__name__
        entry["centerline"] = out.name if out.exists() else None
        manifest[stem] = entry
        status = entry.get("error") or f"{entry['length_m']} m, author {entry['author_time_s']} s" + (
            "  SUSPECT" if entry["suspect"] else ""
        )
        print(f"{stem:20s} {status}")

    MANIFEST.write_text(json.dumps(manifest, indent=1, sort_keys=True))
    built = sum(1 for e in manifest.values() if e.get("centerline"))
    suspect = sorted(k for k, e in manifest.items() if e.get("suspect"))
    print(f"\n{built} centerlines, {len(suspect)} suspect: {suspect}")


if __name__ == "__main__":
    main()
