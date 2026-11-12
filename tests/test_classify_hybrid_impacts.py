import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

from classify_hybrid_impacts import classify_impacts, find_scrapes


def row(t, y, impact=False, speed=50.0, fva=0.0, lateral=0.0, heading=0.0):
    return {
        "race_time_s": str(t),
        "y": str(y),
        "impact_suspect": str(impact),
        "progress_m": str(t * 50.0),
        "speed_mps": str(speed),
        "speed_drop_mps": "6.0" if impact else "0.0",
        "lateral_error_m": str(lateral),
        "heading_error_rad": str(heading),
        "elevation_error_m": "0.0",
        "forward_velocity_angle_rad": str(fva),
    }


def test_falling_car_that_stops_falling_is_a_landing():
    # Falls at 10, 11, 12 m/s, then the impact step barely moves vertically.
    ys = [100.0, 99.5, 98.95, 98.35, 98.33, 98.33]
    rows = [row(i * 0.05, y, impact=(i == 4)) for i, y in enumerate(ys)]
    [impact] = classify_impacts(rows)
    assert impact.kind == "landing"


def test_flat_speed_loss_that_deflects_the_car_is_a_wall_hit():
    rows = [row(i * 0.05, 10.0, impact=(i == 4), fva=0.4 if i >= 4 else 0.0) for i in range(6)]
    [impact] = classify_impacts(rows)
    assert impact.kind == "wall"


def test_speed_loss_at_the_road_edge_is_a_wall_hit():
    rows = [row(i * 0.05, 10.0, impact=(i == 4), lateral=8.0) for i in range(6)]
    [impact] = classify_impacts(rows)
    assert impact.kind == "wall"


def test_undeflected_speed_loss_on_the_line_is_a_bump():
    # Hockolicious v4a 3456 m: a road-transition lip, lat +0.86, fva 0, heading unchanged.
    rows = [row(i * 0.05, 10.0, impact=(i == 4), lateral=0.86) for i in range(7)]
    [impact] = classify_impacts(rows)
    assert impact.kind == "bump"


def drive_row(t, speed, lateral, y=56.0, throttle=1.0):
    base = row(t, y, speed=speed, lateral=lateral)
    return {**base, "progress_m": str(t * 40.0), "throttle": str(throttle), "brake": "0.0", "recovery": "False"}


def test_sustained_deceleration_at_the_road_edge_is_a_scrape():
    # B09 v4 347-383 m: full throttle, no slip, flat, 52 -> 27 m/s on a platform edge at lat +5.
    rows = [drive_row(i * 0.05, 52.0 - 1.5 * i, 5.0) for i in range(12)]
    [scrape] = find_scrapes(rows)
    assert scrape.speed_loss_mps > 10.0


def test_deceleration_on_the_line_or_after_a_fall_is_not_a_scrape():
    on_line = [drive_row(i * 0.05, 52.0 - 1.5 * i, 0.5) for i in range(12)]
    # Landing (A03 1327 m): falling at 20 m/s just before the speed loss.
    landing = [drive_row(i * 0.05, 52.0 - 2.5 * min(max(i - 3, 0), 3), 5.0, y=60.0 - min(i, 3)) for i in range(12)]
    assert find_scrapes(on_line) == []
    assert find_scrapes(landing) == []
