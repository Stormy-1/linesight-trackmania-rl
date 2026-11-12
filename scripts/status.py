import sys
from pathlib import Path
import joblib

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

try:
    from config_files.config import run_name
except Exception:
    run_name = "gen_v1"

base_dir = Path(__file__).resolve().parents[1]
stats_path = base_dir / "save" / run_name / "accumulated_stats.joblib"

if not stats_path.exists():
    fallback_path = base_dir / "save" / "hocko_v1" / "accumulated_stats.joblib"
    if fallback_path.exists():
        stats_path = fallback_path
    else:
        print(f"Stats file not found at: {stats_path}")
        sys.exit(1)

stats = joblib.load(stats_path)
frames = stats.get("cumul_number_frames_played", 0)
alltime_min_ms = stats.get("alltime_min_ms", {})
rolling_mean_ms = stats.get("rolling_mean_ms", {})

print("=" * 65)
print(f" PROJECT LINESIGHT - RUN STATUS: [{run_name}]")
print("=" * 65)
print(f"Total Frames Played:  {frames:,}")
print(f"Total Training Hours: {stats.get('cumul_training_hours', 0.0):.2f}h")
print("-" * 65)
print(f"{'Map':<10} | {'PB (ms)':<10} | {'PB (s)':<10} | {'Rolling Mean (s)':<18}")
print("-" * 65)

all_maps = sorted(set(list(alltime_min_ms.keys()) + list(rolling_mean_ms.keys())))
if not all_maps:
    print("No map runs recorded yet.")
else:
    for m in all_maps:
        pb = alltime_min_ms.get(m, None)
        mean = rolling_mean_ms.get(m, None)
        pb_str = f"{pb/1000.0:.3f}s" if pb else "N/A"
        pb_ms_str = f"{pb}ms" if pb else "N/A"
        mean_str = f"{mean/1000.0:.3f}s" if mean else "N/A"
        print(f"{m:<10} | {pb_ms_str:<10} | {pb_str:<10} | {mean_str:<18}")

print("=" * 65)
