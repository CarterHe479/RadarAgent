"""
RadarAgent – CLI entry point.

Modes
─────
  Interactive      python main.py --interactive
  Single query     python main.py --query "Describe motion 000021"
  Evaluation       python main.py --evaluate [--split test] [--max-samples N]
                                              [--output path/to/results.json]
  Visualise        python main.py --visualize 000021 [--mode skeleton]
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


# ── helpers ───────────────────────────────────────────────────────────────────

def _load_agent():
    """Lazy import + model loading so --visualize/--help don't require GPU."""
    from agent.agent import RadarAgent
    logger.info("Loading Qwen 3 8B model (this may take a minute) …")
    return RadarAgent()


# ── subcommand handlers ───────────────────────────────────────────────────────

def cmd_interactive(args) -> None:
    agent = _load_agent()
    print("\nRadarAgent ready. Type 'quit' or 'exit' to stop.\n")
    while True:
        try:
            query = input("You: ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break
        if not query:
            continue
        if query.lower() in {"quit", "exit", "q"}:
            break
        answer = agent.run(query)
        print(f"\nAgent: {answer}\n")


def cmd_query(args) -> None:
    agent = _load_agent()
    answer = agent.run(args.query)
    print(answer)


def cmd_evaluate(args) -> None:
    from eval.evaluate import run_evaluation, print_metrics_table

    agent = _load_agent()
    output_path = Path(args.output) if args.output else None
    temperature = getattr(args, "temperature", 0.3)

    result = run_evaluation(
        agent,
        split=args.split,
        max_samples=args.max_samples,
        output_path=output_path,
        temperature=temperature,
    )

    if result:
        print_metrics_table(result["metrics"],
                            title=f"RadarAgent – {args.split} split ({result['n_samples']} samples)")
        if output_path:
            print(f"Full results saved to: {output_path}")


def cmd_visualize(args) -> None:
    from tools.visualization import visualize_motion

    result = visualize_motion(
        motion_id=args.motion_id,
        mode=args.mode,
        num_frames=args.num_frames,
    )
    if "error" in result:
        print(f"Error: {result['error']}", file=sys.stderr)
        sys.exit(1)
    print(f"Saved: {result['image_path']}")


# ── argument parser ───────────────────────────────────────────────────────────

def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="radar_agent",
        description="RadarAgent: LLM agent for radar-based human motion understanding",
    )
    sub = parser.add_subparsers(dest="command")

    # interactive
    sub.add_parser("interactive", aliases=["chat"],
                   help="Start an interactive chat session with the agent")

    # single query
    q_parser = sub.add_parser("query", help="Run a single query and print the answer")
    q_parser.add_argument("query", type=str, help="The question or instruction")

    # evaluate
    ev = sub.add_parser("evaluate", aliases=["eval"],
                        help="Run batch evaluation on a dataset split")
    ev.add_argument("--split", default="test", choices=["train", "val", "test"],
                    help="Dataset split to evaluate on (default: test)")
    ev.add_argument("--max-samples", type=int, default=None, dest="max_samples",
                    help="Limit evaluation to N samples (default: all)")
    ev.add_argument("--output", type=str, default=None,
                    help="Path to save JSON results (default: outputs/results/eval_{split}.json)")
    ev.add_argument("--temperature", type=float, default=0.3,
                    help="Generation temperature during evaluation (default: 0.3)")

    # visualise
    viz = sub.add_parser("visualize", aliases=["viz"],
                         help="Generate a motion visualisation")
    viz.add_argument("motion_id", type=str, help="HumanML3D motion identifier")
    viz.add_argument("--mode", default="point_cloud",
                     choices=["point_cloud", "skeleton", "trajectory"],
                     help="Visualisation type (default: point_cloud)")
    viz.add_argument("--num-frames", type=int, default=6, dest="num_frames",
                     help="Number of frames to show (default: 6)")

    # legacy flat flags for backward compatibility
    parser.add_argument("--interactive", action="store_true",
                        help="[legacy] Start interactive mode")
    parser.add_argument("--query", type=str, default=None,
                        help="[legacy] Single query mode")
    parser.add_argument("--evaluate", action="store_true",
                        help="[legacy] Run evaluation")
    parser.add_argument("--split", default="test",
                        choices=["train", "val", "test"])
    parser.add_argument("--max-samples", type=int, default=None, dest="max_samples")
    parser.add_argument("--output", type=str, default=None)
    parser.add_argument("--visualize", type=str, default=None, metavar="MOTION_ID",
                        help="[legacy] Visualise a motion")
    parser.add_argument("--mode", default="point_cloud",
                        choices=["point_cloud", "skeleton", "trajectory"])
    parser.add_argument("--num-frames", type=int, default=6, dest="num_frames")

    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    # subcommand dispatch
    if args.command in {"interactive", "chat"}:
        cmd_interactive(args)
    elif args.command == "query":
        cmd_query(args)
    elif args.command in {"evaluate", "eval"}:
        cmd_evaluate(args)
    elif args.command in {"visualize", "viz"}:
        cmd_visualize(args)

    # legacy flat-flag dispatch
    elif args.interactive:
        cmd_interactive(args)
    elif args.query:
        cmd_query(args)
    elif args.evaluate:
        cmd_evaluate(args)
    elif args.visualize:
        args.motion_id = args.visualize
        cmd_visualize(args)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
