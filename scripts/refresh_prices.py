from __future__ import annotations

import argparse
import sys

from brainstorm_llm import model_prices

DEFAULT_PROVIDERS = ("commandcode", "gemini", "openrouter", "opencode")


def main() -> int:
    parser = argparse.ArgumentParser(description="Refresh BrainStorm model price caches.")
    parser.add_argument(
        "--providers",
        nargs="+",
        default=list(DEFAULT_PROVIDERS),
        help="Providers to refresh (default: all).",
    )
    parser.add_argument("--timeout", type=float, default=120.0)
    args = parser.parse_args()

    failed = 0
    for provider in args.providers:
        try:
            pricing = model_prices(provider, "live", timeout=args.timeout)
        except Exception as exc:  # noqa: BLE001 - CLI summarizes failures
            failed += 1
            print(f"{provider}: FAILED - {exc}", file=sys.stderr)
            continue
        print(f"{provider}: {len(pricing.prices)} models from {pricing.source}")

    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
