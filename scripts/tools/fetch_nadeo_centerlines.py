"""
Fetch official Nadeo map files and world-class ghost replays from TMNF Exchange,
copy challenge files to TrackMania directory, and generate 0.5m centerline .npy files.
"""

import json
import os
import shutil
import time
import urllib.request
from pathlib import Path

from pygbx import Gbx, GbxType
from trackmania_rl.geometry import extract_cp_distance_interval
from trackmania_rl.map_loader import gbx_to_raw_pos_list

GAME_CAMPAIGN_DIR = Path(
    r"C:\Program Files (x86)\Steam\steamapps\common\TrackMania Nations Forever\GameData\Tracks\Campaigns\Nations\White"
)

PROJECT_DIR = Path(__file__).resolve().parents[2]
MAPS_DIR = PROJECT_DIR / "maps"
REPLAY_CACHE = MAPS_DIR / "replay_cache"

USERPROFILE = Path(os.environ["USERPROFILE"])
TM_CHALLENGES_DIR = USERPROFILE / "OneDrive" / "Documents" / "TrackMania" / "Tracks" / "Challenges"
TM_OFFICIAL_MAPS_DIR = TM_CHALLENGES_DIR / "Official Maps"

DEFAULT_TARGETS = [
    "A01-Race",
    "A03-Race",
    "A05-Race",
    "A07-Race",
    "A11-Race",
]


def process_map(map_name: str):
    print(f"\n==========================================")
    print(f" Processing: {map_name}")
    print(f"==========================================")

    challenge_filename = f"{map_name}.Challenge.Gbx"
    src_challenge = GAME_CAMPAIGN_DIR / challenge_filename

    if not src_challenge.exists():
        print(f"ERROR: {src_challenge} does not exist.")
        return False

    # 1. Copy challenge files into user's TrackMania directories
    TM_CHALLENGES_DIR.mkdir(parents=True, exist_ok=True)
    TM_OFFICIAL_MAPS_DIR.mkdir(parents=True, exist_ok=True)

    dest_official = TM_OFFICIAL_MAPS_DIR / challenge_filename
    dest_root = TM_CHALLENGES_DIR / challenge_filename

    shutil.copy2(src_challenge, dest_official)
    shutil.copy2(src_challenge, dest_root)
    print(f"Copied challenge to:\n  - {dest_official}\n  - {dest_root}")

    # 2. Extract map UID
    gbx = Gbx(str(src_challenge))
    challenge = gbx.get_class_by_id(GbxType.CHALLENGE)
    map_uid = getattr(challenge, "map_uid", None)
    print(f"Map UID: {map_uid}")

    if not map_uid:
        print(f"ERROR: Could not find map_uid for {map_name}")
        return False

    # 3. Query TMNF Exchange for TrackId
    track_url = f"https://tmnf.exchange/api/tracks?uid={map_uid}&fields=TrackId,TrackName,UId"
    req = urllib.request.Request(track_url, headers={"User-Agent": "Linesight-RL/1.0"})

    try:
        with urllib.request.urlopen(req) as resp:
            data = json.loads(resp.read().decode())
            results = data.get("Results", [])
            if not results:
                print(f"ERROR: No results on TMNF Exchange for UID {map_uid}")
                return False
            track_id = results[0]["TrackId"]
            print(f"TMNF Exchange TrackId: {track_id}")
    except Exception as e:
        print(f"ERROR querying track: {e}")
        return False

    # 4. Query leaderboards for best replay
    replay_url = f"https://tmnf.exchange/api/replays?trackId={track_id}&fields=ReplayId,ReplayTime,Score"
    req = urllib.request.Request(replay_url, headers={"User-Agent": "Linesight-RL/1.0"})

    try:
        with urllib.request.urlopen(req) as resp:
            data = json.loads(resp.read().decode())
            replays = data.get("Results", [])
            if not replays:
                print(f"ERROR: No replays found for TrackId {track_id}")
                return False

            # Sort by ReplayTime or Score (lowest first)
            replays_with_time = [r for r in replays if r.get("ReplayTime") or r.get("Score")]
            if replays_with_time:
                best_replay = min(
                    replays_with_time,
                    key=lambda r: r.get("ReplayTime") or r.get("Score") or 999999,
                )
            else:
                best_replay = replays[0]

            replay_id = best_replay["ReplayId"]
            best_time = (best_replay.get("ReplayTime") or best_replay.get("Score", 0)) / 1000.0
            print(f"Selected Replay ID: {replay_id} (Time: {best_time:.3f}s)")
    except Exception as e:
        print(f"ERROR querying replays: {e}")
        return False

    # 5. Download replay
    REPLAY_CACHE.mkdir(parents=True, exist_ok=True)
    replay_path = REPLAY_CACHE / f"{map_name}.Replay.Gbx"

    download_url = f"https://tmnf.exchange/recordgbx/{replay_id}"
    req = urllib.request.Request(download_url, headers={"User-Agent": "Linesight-RL/1.0"})

    try:
        with urllib.request.urlopen(req) as resp:
            replay_path.write_bytes(resp.read())
        print(f"Downloaded replay to {replay_path} ({replay_path.stat().st_size} bytes)")
    except Exception as e:
        print(f"ERROR downloading replay: {e}")
        return False

    # 6. Extract ghost coordinates & generate 0.5m centerline
    try:
        pos_list = gbx_to_raw_pos_list(replay_path)
        cps = extract_cp_distance_interval(pos_list, 0.5, PROJECT_DIR)
        print(f"Generated centerline shape: {cps.shape}")

        dest_npy = MAPS_DIR / f"{map_name}_0.5m_cl2.npy"
        temp_map_npy = MAPS_DIR / "map.npy"
        if temp_map_npy.exists():
            shutil.move(temp_map_npy, dest_npy)
            print(f"SUCCESS: Saved {dest_npy.name} ({dest_npy.stat().st_size} bytes)")
            return True
        else:
            print("ERROR: maps/map.npy was not created.")
            return False
    except Exception as e:
        print(f"ERROR extracting centerline: {e}")
        return False


def main():
    successes = []
    failures = []

    for map_name in DEFAULT_TARGETS:
        ok = process_map(map_name)
        if ok:
            successes.append(map_name)
        else:
            failures.append(map_name)
        time.sleep(1)  # Polite API pacing

    print("\n" + "=" * 50)
    print(" SUMMARY")
    print("=" * 50)
    print(f"Successful ({len(successes)}): {', '.join(successes)}")
    if failures:
        print(f"Failed ({len(failures)}): {', '.join(failures)}")


if __name__ == "__main__":
    main()
