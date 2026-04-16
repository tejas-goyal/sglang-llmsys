"""
Run sglang bench_serving against a deployed eval server.

Usage:
  python run_bench.py --base-url <MODAL_URL> --policy lru
  python run_bench.py --base-url <MODAL_URL> --policy wlfu

Full workflow:
  1. Edit EVICTION_POLICY in eval_eviction.py -> "lru"
  2. modal deploy eval_eviction.py            (note the printed URL)
  3. python run_bench.py --base-url <URL> --policy lru
  4. Edit EVICTION_POLICY -> "wlfu", redeploy, re-bench
  5. Compare results_lru.json vs results_wlfu.json
"""

import argparse
import subprocess
import sys


def main():
    p = argparse.ArgumentParser(description="Benchmark eviction policy")
    p.add_argument("--base-url", required=True, help="Deployed Modal server URL")
    p.add_argument("--policy", default="lru", help="Label for output file (lru/wlfu)")
    p.add_argument("--model", default="Qwen/Qwen3.5-35B-A3B-FP8")
    p.add_argument("--request-rate", type=float, default=16.0)
    p.add_argument("--num-prompts", type=int, default=256)
    p.add_argument("--gsp-num-groups", type=int, default=8)
    p.add_argument("--gsp-prompts-per-group", type=int, default=32)
    p.add_argument("--gsp-system-prompt-len", type=int, default=2048)
    p.add_argument("--gsp-question-len", type=int, default=128)
    p.add_argument("--gsp-output-len", type=int, default=256)
    args = p.parse_args()

    output_file = f"results_{args.policy}.json"

    bench_cmd = [
        sys.executable, "-m", "sglang.bench_serving",
        "--backend", "sglang",
        "--base-url", args.base_url,
        "--model", args.model,
        "--dataset-name", "generated-shared-prefix",
        "--num-prompts", str(args.num_prompts),
        "--request-rate", str(args.request_rate),
        "--gsp-num-groups", str(args.gsp_num_groups),
        "--gsp-prompts-per-group", str(args.gsp_prompts_per_group),
        "--gsp-system-prompt-len", str(args.gsp_system_prompt_len),
        "--gsp-question-len", str(args.gsp_question_len),
        "--gsp-output-len", str(args.gsp_output_len),
        "--output-file", output_file,
    ]

    print(f"\n{'='*60}")
    print(f"  Policy:       {args.policy}")
    print(f"  Server:       {args.base_url}")
    print(f"  Request rate: {args.request_rate} req/s")
    print(f"  Num prompts:  {args.num_prompts}")
    print(f"  GSP groups:   {args.gsp_num_groups} x {args.gsp_prompts_per_group}")
    print(f"  Sys prompt:   {args.gsp_system_prompt_len} tokens")
    print(f"{'='*60}\n")

    print(f"Running: {' '.join(bench_cmd)}\n")
    subprocess.run(bench_cmd, check=True)
    print(f"\nResults saved to {output_file}")


if __name__ == "__main__":
    main()
