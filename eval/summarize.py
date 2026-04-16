"""Render a markdown summary table for a results directory.

Usage:
    python3 sglang/eval/summarize.py sglang/eval/results/<timestamp>_<tag>
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

METRICS = [
    ("Output throughput (tok/s)",  "output_throughput",     "%.2f"),
    ("Request throughput (req/s)", "request_throughput",    "%.2f"),
    ("Mean TTFT (ms)",             "mean_ttft_ms",          "%.2f"),
    ("P99 TTFT (ms)",              "p99_ttft_ms",           "%.2f"),
    ("Mean E2E latency (ms)",      "mean_e2e_latency_ms",   "%.2f"),
    ("P99 E2E latency (ms)",       "p99_e2e_latency_ms",    "%.2f"),
    ("Cache hit rate (final)",     "cache_hit_rate_final",  "%.3f"),
    ("Eviction count",             "eviction_count",        "%d"),
]

# Metrics where a larger value is better (for delta coloring/sign convention).
HIGHER_IS_BETTER = {"output_throughput", "request_throughput", "cache_hit_rate_final"}


def _fmt(val, fmt: str) -> str:
    if val is None:
        return "—"
    try:
        if fmt == "%d":
            return fmt % int(val)
        return fmt % float(val)
    except (TypeError, ValueError):
        return str(val)


def _delta(baseline, value, key: str) -> str:
    if baseline in (None, 0) or value is None:
        return "—"
    pct = (value - baseline) / baseline * 100
    sign = "+" if pct >= 0 else ""
    return f"{sign}{pct:.1f}%"


def render_summary(run_dir: Path) -> str:
    config = json.loads((run_dir / "config.json").read_text())
    policies: list[str] = config["policies"]
    results = {p: json.loads((run_dir / f"{p}.json").read_text()) for p in policies}
    baseline = policies[0]

    lines: list[str] = []
    lines.append(f"# Eviction eval: {config['tag']}")
    lines.append("")
    lines.append(f"- Run: `{config['timestamp']}`")
    lines.append(f"- Model: `{config['model']}`")
    lines.append(f"- Cache capacity: {config.get('cache_capacity_tokens', '—')} tokens")
    lines.append(f"- sglang SHA: `{config.get('sglang_git_sha', 'unknown')[:12]}`")
    lines.append(f"- Bench params: `{json.dumps(config['bench_params'])}`")
    lines.append(f"- Baseline (for deltas): **{baseline}**")
    lines.append("")

    header = ["Metric"] + policies + [f"Δ vs {baseline}" for p in policies[1:]]
    lines.append("| " + " | ".join(header) + " |")
    lines.append("|" + "|".join("---" for _ in header) + "|")

    for label, key, fmt in METRICS:
        row = [label]
        values = [results[p].get(key) for p in policies]
        row.extend(_fmt(v, fmt) for v in values)
        for p, v in zip(policies[1:], values[1:]):
            row.append(_delta(values[0], v, key))
        lines.append("| " + " | ".join(row) + " |")

    lines.append("")
    return "\n".join(lines)


def main() -> None:
    if len(sys.argv) != 2:
        print("Usage: summarize.py <run-dir>", file=sys.stderr)
        sys.exit(2)
    run_dir = Path(sys.argv[1])
    out = render_summary(run_dir)
    (run_dir / "summary.md").write_text(out)
    print(out)


if __name__ == "__main__":
    main()
