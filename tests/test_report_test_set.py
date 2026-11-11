import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

from report_test_set import aggregate, format_table, run_metrics
from run_test_set import command_for, max_race_s


def summary(finished=True, time=30.0, pct=100.0, caps=()):
    run = {"finished": finished, "termination": "Finished" if finished else "Stuck_Cutoff_2s",
           "race_time_s": time, "completion_pct": pct, "recovery_steps": 4}
    return {"speed_caps": list(caps), "max_speed_override_mps": None, "runs": [run]}


def row(t, speed, y=0.0, impact=False):
    return {"race_time_s": str(t), "y": str(y), "impact_suspect": str(impact), "progress_m": str(t * 50),
            "speed_mps": str(speed), "speed_drop_mps": "6.0" if impact else "0.0", "lateral_error_m": "8.0",
            "heading_error_rad": "0.4", "elevation_error_m": "0.0", "forward_velocity_angle_rad": "0.4",
            "throttle": "1.0", "brake": "0.0", "recovery": "False"}


def test_finished_run_gets_time_ratio_against_replay():
    m = run_metrics("X01", "White", 20.0, summary(time=30.0), [row(0.0, 50.0), row(0.05, 50.0)])
    assert m.finished and m.time_ratio == 1.5
    assert (m.landings, m.bumps, m.walls, m.scrapes) == (0, 0, 0, 0)


def test_failed_run_has_no_time_ratio_and_counts_a_wall():
    rows = [row(0.0, 50.0), row(0.05, 50.0), row(0.10, 40.0, impact=True)]
    m = run_metrics("X02", "Red", 20.0, summary(finished=False, pct=42.0), rows)
    assert m.time_ratio is None and m.walls == 1


def test_speed_capped_runs_are_rejected():
    with pytest.raises(ValueError):
        run_metrics("X03", "White", 20.0, summary(caps=[(0, 10, 40)]), [row(0.0, 50.0)])


def test_aggregate_finish_rate_and_median_ratio():
    a = run_metrics("A", "White", 20.0, summary(time=30.0), [row(0.0, 50.0)])
    b = run_metrics("B", "White", 10.0, summary(time=20.0), [row(0.0, 50.0)])
    c = run_metrics("C", "Red", 10.0, summary(finished=False, pct=50.0), [row(0.0, 50.0)])
    agg = aggregate([a, b, c])
    assert agg["finished"] == 2 and agg["finish_rate"] == round(2 / 3, 3)
    assert agg["median_time_ratio"] == 1.75
    assert agg["mean_completion_pct"] == pytest.approx(83.3)
    assert "Finished 2/3" in format_table([a, b, c])


def test_runner_uses_identical_args_for_every_map():
    cmd = command_for("A09-Race", 25.98, "v4")
    assert "--speed-cap" not in cmd and "--max-speed" not in cmd
    assert cmd[cmd.index("--name") + 1] == "A09-Race_test_v4"
    assert max_race_s(25.98) == 300 and max_race_s(153.26) == 613
