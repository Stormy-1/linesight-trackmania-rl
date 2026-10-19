"""Repair an older telemetry CSV using finite-difference position headings."""

from __future__ import annotations

import argparse
from pathlib import Path

from trackmania_rl.hybrid_rollout import derive_headings_from_positions, load_rollout_csv
from trackmania_rl.hybrid_telemetry import HybridTelemetryWriter


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    samples = derive_headings_from_positions(load_rollout_csv(args.input))
    with HybridTelemetryWriter(args.output) as writer:
        for sample in samples:
            writer.write_sample(sample.timestamp_s, sample.position, sample.heading, sample.speed_mps)
    print({"output": str(args.output), "samples": len(samples)})


if __name__ == "__main__":
    main()
