#!/usr/bin/env python3
"""Compute the ACTUAL $ cost of LLM-benchmark batch runs from their response archives.

Portkey's provider-batch mode is stateless and does NOT log token cost to its
dashboard, so the Portkey UI under-reports spend for these runs. This script
reconstructs the real cost directly from the per-call `usage` recorded in each
`*__responses.jsonl` archive, priced at the providers' published *batch* rates
(50% off standard). It is the self-serve replacement for the (unavailable)
Portkey invoice line.

Usage:
    python scripts/llm_cost.py                       # all results/forecasts/*__responses.jsonl
    python scripts/llm_cost.py results/forecasts/llm_v8_q0__*sonnet_5*__responses.jsonl
    python scripts/llm_cost.py --sonnet-list <files> # bill Sonnet 5 at list, not intro

Rates are batch (-50%) $/MTok. Sonnet 5 defaults to INTRO pricing ($2/$10 list
-> $1/$5 batch), valid through 2026-08-31; pass --sonnet-list for the $3/$15
list ceiling. Cache read = 0.1x input; cache write = 1.25x input (Anthropic);
OpenAI has no cache-write premium.
"""
import argparse
import glob
import json
import sys
from collections import defaultdict
from pathlib import Path

DEFAULT_GLOB = "results/forecasts/*__responses.jsonl"


def batch_rate_table(sonnet_intro: bool) -> dict:
    """Return {model: {inp,out,cr,cw}} in $/MTok at batch (-50%) rates."""
    s_in, s_out = (1.00, 5.00) if sonnet_intro else (1.50, 7.50)
    return {
        "claude-opus-4-8": dict(inp=2.50, out=12.50, cr=2.50 * 0.10, cw=2.50 * 1.25),
        "claude-sonnet-5": dict(inp=s_in, out=s_out, cr=s_in * 0.10, cw=s_in * 1.25),
        "gpt-5.5":         dict(inp=2.50, out=15.00, cr=0.25, cw=0.0),
    }


def norm_model(m: str) -> str:
    """Strip any '@slug/...' routing prefix and lowercase -> bare model id."""
    if not m:
        return "?"
    return m.split("/")[-1].strip().lower()


def cost_of(usage: dict, rate: dict) -> float:
    if not usage or not rate:
        return 0.0
    return (
        usage.get("input_tokens", 0) * rate["inp"]
        + usage.get("output_tokens", 0) * rate["out"]
        + usage.get("cache_read_input_tokens", 0) * rate["cr"]
        + usage.get("cache_creation_input_tokens", 0) * rate["cw"]
    ) / 1_000_000.0


def add_usage(acc: dict, u: dict) -> None:
    for k in ("input_tokens", "output_tokens",
              "cache_creation_input_tokens", "cache_read_input_tokens"):
        acc[k] += (u.get(k, 0) or 0)


def scan_file(path: Path, rates: dict, unknown: set, near_cap_threshold: int) -> dict:
    """Dedupe by custom_id (keep the completed call), return per-file aggregate.

    Also flags records whose output_tokens are within near_cap_threshold of the
    model's max_tokens cap — a proxy for silent truncation (a response that hit
    the cap parses its emitted horizons with no explicit truncation flag).
    """
    best = {}           # custom_id -> (output_tokens, model, usage)  [successes]
    error_cids = set()  # distinct custom_ids with an error record (deduped)
    dupes = 0
    total_lines = 0
    with path.open("r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            total_lines += 1
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                continue
            cid = rec.get("custom_id") or f"{rec.get('gvkey')}_{rec.get('q0')}"
            usage = rec.get("usage")
            model = norm_model(rec.get("model", ""))
            if rec.get("error") and not usage:
                error_cids.add(cid)
                continue
            if not usage:
                continue
            out = usage.get("output_tokens", 0) or 0
            if cid in best:
                dupes += 1
                if out <= best[cid][0]:
                    continue
            best[cid] = (out, model, usage)

    # An origin that errored on one attempt but later succeeded (gap-fill on a
    # subsequent --resume, appended to the SAME file) is NOT a failure. Count as
    # errored only custom_ids that errored AND have no successful usage record;
    # the rest are transient errors that were later gap-filled.
    succeeded = set(best.keys())
    errored_origins = error_cids - succeeded
    transient_errors = error_cids & succeeded

    agg = {
        "file": path.name,
        "origins": len(best),
        "errors": len(errored_origins),
        "transient_errors": len(transient_errors),
        "dupes": dupes,
        "total_lines": total_lines,
        "by_model": defaultdict(lambda: {"origins": 0, "cost": 0.0,
                                         "tok": defaultdict(int)}),
        "cost": 0.0,
        "near_cap": [],   # (custom_id, model, output_tokens)
    }
    for cid, (out, model, usage) in best.items():
        rate = rates.get(model)
        if rate is None:
            unknown.add(model)
        c = cost_of(usage, rate) if rate else 0.0
        m = agg["by_model"][model]
        m["origins"] += 1
        m["cost"] += c
        add_usage(m["tok"], usage)
        agg["cost"] += c
        if out >= near_cap_threshold:
            agg["near_cap"].append((cid, model, out))
    return agg


def fmt_tok(t: int) -> str:
    return f"{t:,}"


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("paths", nargs="*", help="response .jsonl files or globs "
                    f"(default: {DEFAULT_GLOB})")
    ap.add_argument("--sonnet-list", action="store_true",
                    help="bill Sonnet 5 at list ($1.5/$7.5 batch) instead of intro ($1/$5)")
    ap.add_argument("--max-tokens", type=int, default=64000,
                    help="the run's max_tokens cap (default 64000)")
    ap.add_argument("--near-cap-threshold", type=int, default=None,
                    help="flag records with output_tokens >= this as possible "
                         "truncation (default: 0.94 * --max-tokens, i.e. 60160)")
    args = ap.parse_args()
    near_cap_threshold = (args.near_cap_threshold
                          if args.near_cap_threshold is not None
                          else int(0.94 * args.max_tokens))

    patterns = args.paths or [DEFAULT_GLOB]
    files = []
    for p in patterns:
        matched = glob.glob(p)
        if matched:
            files.extend(matched)
        elif Path(p).exists():
            files.append(p)
    files = sorted(set(files))
    if not files:
        print("No response archives matched:", ", ".join(patterns), file=sys.stderr)
        return 1

    rates = batch_rate_table(sonnet_intro=not args.sonnet_list)
    unknown: set = set()

    print(f"Sonnet 5 priced at {'LIST' if args.sonnet_list else 'INTRO'} batch rate; "
          f"all rates are batch (-50%).\n")
    header = f"{'archive':<62} {'model':<16} {'orig':>5} {'in':>11} {'out':>12} {'cost $':>9}  $/orig"
    print(header)
    print("-" * len(header))

    grand = {"origins": 0, "cost": 0.0, "errors": 0, "transient": 0, "dupes": 0,
             "tok": defaultdict(int)}
    per_model = defaultdict(lambda: {"origins": 0, "cost": 0.0})
    near_cap_all = []

    for f in files:
        agg = scan_file(Path(f), rates, unknown, near_cap_threshold)
        near_cap_all.extend((agg["file"], *t) for t in agg["near_cap"])
        for model, m in sorted(agg["by_model"].items()):
            per_orig = m["cost"] / m["origins"] if m["origins"] else 0.0
            print(f"{agg['file']:<62} {model:<16} {m['origins']:>5} "
                  f"{fmt_tok(m['tok']['input_tokens']):>11} "
                  f"{fmt_tok(m['tok']['output_tokens']):>12} "
                  f"{m['cost']:>9.2f}  {per_orig:.4f}")
            per_model[model]["origins"] += m["origins"]
            per_model[model]["cost"] += m["cost"]
            for k, v in m["tok"].items():
                grand["tok"][k] += v
        grand["origins"] += agg["origins"]
        grand["cost"] += agg["cost"]
        grand["errors"] += agg["errors"]
        grand["transient"] += agg["transient_errors"]
        grand["dupes"] += agg["dupes"]

    print("-" * len(header))
    print("\nPer-model totals:")
    for model, m in sorted(per_model.items()):
        per_orig = m["cost"] / m["origins"] if m["origins"] else 0.0
        print(f"  {model:<16} {m['origins']:>6} origins   ${m['cost']:>9.2f}   "
              f"(${per_orig:.4f}/origin)")

    print(f"\nGRAND TOTAL: {grand['origins']:,} origins   "
          f"input={fmt_tok(grand['tok']['input_tokens'])}  "
          f"output={fmt_tok(grand['tok']['output_tokens'])}  "
          f"cache_read={fmt_tok(grand['tok']['cache_read_input_tokens'])}  "
          f"cache_write={fmt_tok(grand['tok']['cache_creation_input_tokens'])}")
    print(f"             ${grand['cost']:.2f}   "
          f"({grand['errors']} unresolved errored origin(s) @ $0, "
          f"{grand['transient']} transient error(s) later gap-filled, "
          f"{grand['dupes']} duplicate record(s) collapsed)")

    print(f"\nTRUNCATION CHECK (output_tokens >= {near_cap_threshold:,} of "
          f"{args.max_tokens:,} cap):")
    if not near_cap_all:
        print("  OK - no near-cap records; nothing looks truncated.")
    else:
        print(f"  WARNING - {len(near_cap_all)} record(s) near the cap "
              f"(likely silent truncation, verify the raw completion):")
        for fname, cid, model, out in sorted(near_cap_all, key=lambda x: -x[3]):
            print(f"    {cid}  {model:<16} out={out:,}   {fname}")

    if unknown:
        print("\nWARNING: no rate for model(s), counted at $0: "
              + ", ".join(sorted(unknown)), file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
