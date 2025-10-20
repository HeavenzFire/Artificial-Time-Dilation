import argparse
from pathlib import Path

from .framework import AngelusFramework
from .common.types import Intent, SigilSpec
from .common.sigil import generate_sigil_svg


def cmd_guidance(args: argparse.Namespace) -> int:
    framework = AngelusFramework()
    intent = Intent(user_id=args.user_id, text=args.text, tags=args.tags or [])
    g = framework.run(intent)
    print("Summary:", g.summary)
    print("Archetype:", g.archetype)
    print("Confidence:", g.confidence)
    for step in g.steps:
        print(" -", step)
    return 0


def cmd_sigil(args: argparse.Namespace) -> int:
    spec = SigilSpec(seed=args.seed, color=args.color, background=args.background, size=args.size, stroke=args.stroke)
    svg = generate_sigil_svg(spec)
    out_path = Path(args.out)
    out_path.write_text(svg, encoding="utf-8")
    print(f"Sigil saved to {out_path}")
    return 0


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="angelus", description="Spirit Angelus Framework CLI")
    sub = p.add_subparsers(dest="command", required=True)

    g = sub.add_parser("guidance", help="Get guidance for an intent")
    g.add_argument("--text", required=True, help="Intent text")
    g.add_argument("--user-id", default=None)
    g.add_argument("--tags", nargs="*", default=[])
    g.set_defaults(func=cmd_guidance)

    s = sub.add_parser("sigil", help="Generate sigil SVG")
    s.add_argument("--seed", required=True)
    s.add_argument("--out", default="angelus_sigil.svg")
    s.add_argument("--color", default="#4B6FFF")
    s.add_argument("--background", default="#0B1020")
    s.add_argument("--size", type=int, default=256)
    s.add_argument("--stroke", type=int, default=3)
    s.set_defaults(func=cmd_sigil)

    return p


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
