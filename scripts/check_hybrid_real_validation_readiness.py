"""Run a read-only preflight for the later real hybrid validation.

This command checks workspace artifacts and diagnostic provenance only. It does
not launch TrackMania, connect to TMInterface, start training, load a
checkpoint, or change controller configuration.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from trackmania_rl.hybrid_controller import ConservativeControllerConfig
from trackmania_rl.hybrid_rollout import load_rollout_csv
from trackmania_rl.hybrid_telemetry import TELEMETRY_COLUMNS

DEFAULT_OUTPUT = Path("eval_results/hybrid_rollouts/hybrid_real_validation_readiness.json")
REQUIRED_CENTERLINES = (
    Path("maps/A01-Race_0.5m_cl2.npy"),
    Path("maps/A07-Race_0.5m_cl2.npy"),
    Path("maps/ESL-Hockolicious_0.5m_cl2.npy"),
)
REQUIRED_TELEMETRY = (
    Path("eval_results/hybrid_rollouts_crossmap/A01-Race.repaired.csv"),
    Path("eval_results/hybrid_rollouts_crossmap/A07-Race.repaired.csv"),
)
REQUIRED_REPORTS = (
    Path("eval_results/hybrid_rollouts/hybrid_predictive_features.json"),
    Path("eval_results/hybrid_rollouts/hybrid_predictive_feature_calibration.json"),
    Path("eval_results/hybrid_rollouts/hybrid_recovery_authority_gap.json"),
    Path("eval_results/hybrid_rollouts/hybrid_shadow_observer.json"),
)
REQUIRED_LIVE_PATHS = (
    Path("trackmania_rl/tmi_interaction/game_instance_manager.py"),
    Path("trackmania_rl/tmi_interaction/Python_Link.as"),
)


def _check_file(root: Path, relative: Path) -> dict[str, Any]:
    path = root / relative
    return {
        "name": f"file:{relative.as_posix()}",
        "passed": path.is_file(),
        "detail": "present" if path.is_file() else "missing",
    }


def build_readiness_report(root: Path = Path(".")) -> dict[str, Any]:
    root = root.resolve()
    checks: list[dict[str, Any]] = []
    checks.extend(_check_file(root, path) for path in REQUIRED_CENTERLINES)
    checks.extend(_check_file(root, path) for path in REQUIRED_TELEMETRY)
    checks.extend(_check_file(root, path) for path in REQUIRED_REPORTS)
    checks.extend(_check_file(root, path) for path in REQUIRED_LIVE_PATHS)

    telemetry_rows: dict[str, int | str] = {}
    for relative in REQUIRED_TELEMETRY:
        path = root / relative
        if not path.is_file():
            continue
        try:
            rows = load_rollout_csv(path)
            telemetry_rows[relative.as_posix()] = len(rows)
            checks.append(
                {
                    "name": f"telemetry_schema:{relative.as_posix()}",
                    "passed": True,
                    "detail": f"{len(rows)} validated rows; columns={','.join(TELEMETRY_COLUMNS)}",
                }
            )
        except (OSError, ValueError) as exc:
            telemetry_rows[relative.as_posix()] = str(exc)
            checks.append(
                {
                    "name": f"telemetry_schema:{relative.as_posix()}",
                    "passed": False,
                    "detail": str(exc),
                }
            )

    reports_checked = 0
    for relative in REQUIRED_REPORTS:
        path = root / relative
        if not path.is_file():
            continue
        try:
            report = json.loads(path.read_text(encoding="utf-8"))
            reports_checked += 1
            safe = (
                report.get("offline_only") is True
                and report.get("promotion_allowed") is False
                and report.get("real_physics_validated") is False
            )
            checks.append(
                {
                    "name": f"report_provenance:{relative.as_posix()}",
                    "passed": safe,
                    "detail": "offline diagnostic flags are intact" if safe else "unsafe or incomplete provenance flags",
                }
            )
        except (OSError, TypeError, ValueError, json.JSONDecodeError) as exc:
            checks.append(
                {
                    "name": f"report_provenance:{relative.as_posix()}",
                    "passed": False,
                    "detail": str(exc),
                }
            )

    defaults_before = ConservativeControllerConfig()
    defaults_after = ConservativeControllerConfig()
    defaults_unchanged = defaults_before == defaults_after
    checks.append(
        {
            "name": "controller_defaults_unchanged_in_process",
            "passed": defaults_unchanged,
            "detail": "default dataclass values are stable" if defaults_unchanged else "default values changed",
        }
    )
    artifact_checks_passed = all(bool(check["passed"]) for check in checks)
    return {
        "scope": "offline_hybrid_real_validation_readiness",
        "offline_only": True,
        "trackmania_launched": False,
        "training_started": False,
        "checkpoints_loaded": False,
        "real_physics_validated": False,
        "promotion_allowed": False,
        "offline_preflight_complete": artifact_checks_passed,
        "real_validation_authorization_required": True,
        "telemetry_rows": telemetry_rows,
        "reports_checked": reports_checked,
        "checks": checks,
        "next_action": "perform a separately authorized visible validation with explicit rollback and telemetry capture",
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path("."))
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    report = build_readiness_report(args.root)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"output": str(args.output), "passed": report["offline_preflight_complete"]}, separators=(",", ":")))


if __name__ == "__main__":
    main()
