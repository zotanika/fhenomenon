"""Capture and report an OpenVLA training graph without FHE execution."""

import argparse
import json
from pathlib import Path

import torch

from .graph import capture_training_graph
from .source import DEFAULT_SNAPSHOT


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--snapshot", default=str(DEFAULT_SNAPSHOT))
    parser.add_argument(
        "--profile", choices=("micro", "tiny", "full7b_meta"), default="tiny"
    )
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--threads", type=int, default=2)
    parser.add_argument("--max-report-nodes", type=int, default=256)
    parser.add_argument("--output", required=True)
    parser.add_argument(
        "--graph-code",
        help="Optional FX graph source output, containing no tensor values",
    )
    args = parser.parse_args()
    if args.threads < 1:
        parser.error("--threads must be positive")
    torch.set_num_threads(args.threads)
    captured = capture_training_graph(
        args.snapshot,
        args.profile,
        learning_rate=args.learning_rate,
        seed=args.seed,
        max_report_nodes=args.max_report_nodes,
    )
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(captured.metadata, indent=2, allow_nan=False) + "\n")
    if args.graph_code:
        code = Path(args.graph_code)
        code.parent.mkdir(parents=True, exist_ok=True)
        code.write_text(captured.gm.code)
    print(
        json.dumps(
            {
                "profile": args.profile,
                "report": str(output.resolve()),
                "nodes": captured.metadata["node_count"],
                "numerically_validated": captured.validation is not None,
                "validation_passed": None
                if captured.validation is None
                else captured.validation["passed"],
                "encrypted_execution_performed": False,
            }
        )
    )
    return 0 if captured.validation is None or captured.validation["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
