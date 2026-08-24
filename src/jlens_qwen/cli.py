"""Command-line interface for the committed minimal protocol."""

from __future__ import annotations

import argparse
from pathlib import Path

from .experiment import reproduce


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m jlens_qwen",
        description="Reproduce a minimal Jacobian lens on a pinned Qwen model.",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)
    command = subparsers.add_parser("reproduce", help="fit and evaluate the fixed protocol")
    command.add_argument("--config", type=Path, required=True)
    command.add_argument("--output-dir", type=Path, required=True)
    command.add_argument(
        "--device",
        default="auto",
        help="auto, cpu, mps, cuda, or a concrete torch device such as cuda:1",
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    if args.command == "reproduce":
        report = reproduce(args.config, args.output_dir.resolve(), args.device)
        result = report["evaluation"]
        print(
            f"Completed {report['environment']['experiment_id']}: "
            f"answer rank={result['final_answer_rank']}, "
            f"sanity_pass={result['sanity_check_passed']}"
        )
