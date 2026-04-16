"""
Eval: compare LRU vs WLFU KV cache eviction strategies.

Design: runs sglang server + bench_serving inside the same Modal container,
hitting localhost:8000. No Modal web_server proxy, no network hops, no
408/500 connection errors.

Usage:
  modal run eval_eviction.py                          # runs both policies, prints comparison
  modal run eval_eviction.py --policy lru             # single policy
"""

import json
import subprocess
import sys
import time
from pathlib import Path

import modal

MINUTES = 60

# ---------------------------------------------------------------------------
# Image: patch sglang v0.5.9 in-place to add WLFUStrategy
# ---------------------------------------------------------------------------
LOCAL_SGLANG = Path(__file__).parent / "sglang"
LOCAL_PATCH = str(Path(__file__).parent / "_patch_sglang.py")
SGLANG_SRT = "/sgl-workspace/sglang/python/sglang/srt"

sglang_image = (
    modal.Image.from_registry("lmsysorg/sglang:v0.5.9-cu129-amd64-runtime")
    .entrypoint([])
    .env({
        "HF_HUB_CACHE": "/root/.cache/huggingface",
        "HF_XET_HIGH_PERFORMANCE": "1",
        "SGLANG_ENABLE_JIT_DEEPGEMM": "0",
    })
    .add_local_file(
        str(LOCAL_SGLANG / "python" / "sglang" / "srt" / "mem_cache" / "evict_policy.py"),
        remote_path=f"{SGLANG_SRT}/mem_cache/evict_policy.py",
        copy=True,
    )
    .add_local_file(LOCAL_PATCH, remote_path="/tmp/_patch.py", copy=True)
    .run_commands(f"python3 /tmp/_patch.py {SGLANG_SRT} && rm /tmp/_patch.py")
)

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
# Pure transformer model — prefix caching works correctly.
# Qwen3.5-35B-A3B is a Mamba hybrid: MambaRadixCache ignores our eviction patch.
MODEL_NAME = "Qwen/Qwen2.5-7B-Instruct"
MODEL_REVISION = None  # use latest
PORT = 8000
MEM_FRACTION = 0.85  # A100-40GB: ~34GB model → ~6GB free → ~85% gives ~5GB KV cache
                     # Per-token KV: 28 layers × 2 × 4 heads × 128 dim × 2B = 57KB
                     # → ~87K token capacity

# Workload: 128 groups × 1024 tokens = 131K prefix tokens > 87K cache
# With --tokenize-prompt: exact token IDs sent, guaranteed prefix reuse + eviction.
BENCH_PARAMS = {
    "num_prompts": 2048,
    "gsp_num_groups": 128,
    "gsp_prompts_per_group": 16,
    "gsp_system_prompt_len": 1024,
    "gsp_question_len": 64,
    "gsp_output_len": 64,
    "request_rate": 16,
}

HF_CACHE_VOL = modal.Volume.from_name("huggingface-cache", create_if_missing=True)
HF_CACHE_PATH = "/root/.cache/huggingface"

app = modal.App("eval-eviction-policy")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _wait_ready(process, timeout=8 * MINUTES):
    import requests as req
    deadline = time.time() + timeout
    while time.time() < deadline:
        try:
            if (rc := process.poll()) is not None:
                raise subprocess.CalledProcessError(rc, cmd=process.args)
            req.get(f"http://127.0.0.1:{PORT}/health").raise_for_status()
            return
        except Exception:
            time.sleep(5)
    raise TimeoutError(f"SGLang not ready after {timeout}s")


def _get_cache_capacity():
    import requests as req
    text = req.get(f"http://127.0.0.1:{PORT}/metrics").text
    for line in text.splitlines():
        if "max_total_num_tokens" in line and not line.startswith("#"):
            return int(float(line.split()[-1]))
    return None


def _get_eviction_count():
    import requests as req
    text = req.get(f"http://127.0.0.1:{PORT}/metrics").text
    # eviction_duration_seconds histogram appears only if evictions occurred
    has_evictions = "eviction_duration_seconds_count" in text
    for line in text.splitlines():
        if "eviction_duration_seconds_count" in line and not line.startswith("#"):
            return int(float(line.split()[-1]))
    return 0


def _get_cache_hit_rate():
    import requests as req
    text = req.get(f"http://127.0.0.1:{PORT}/metrics").text
    for line in text.splitlines():
        if "sglang:cache_hit_rate{" in line:
            return float(line.split()[-1])
    return 0.0


# ---------------------------------------------------------------------------
# Core experiment function — runs server + bench in same container
# ---------------------------------------------------------------------------
with sglang_image.imports():
    import requests as _req


@app.function(
    image=sglang_image,
    gpu="A100",
    volumes={HF_CACHE_PATH: HF_CACHE_VOL},
    timeout=90 * MINUTES,
)
def run_policy(policy: str) -> dict:
    """Start sglang with the given policy, run bench_serving locally, return results."""
    print(f"\n{'='*60}")
    print(f"  Running policy: {policy}")
    print(f"  Cache capacity target: ~246K tokens")
    print(f"  Workload: {BENCH_PARAMS['gsp_num_groups']} groups × "
          f"{BENCH_PARAMS['gsp_system_prompt_len']} tokens = "
          f"{BENCH_PARAMS['gsp_num_groups'] * BENCH_PARAMS['gsp_system_prompt_len']:,} prefix tokens")
    print(f"{'='*60}\n")

    # Start sglang server
    revision_args = ["--revision", MODEL_REVISION] if MODEL_REVISION else []
    cmd = [
        "python", "-m", "sglang.launch_server",
        "--model-path", MODEL_NAME,
        *revision_args,
        "--served-model-name", MODEL_NAME,
        "--host", "0.0.0.0",
        "--port", str(PORT),
        "--tp", "1",
        "--mem-fraction", str(MEM_FRACTION),
        "--radix-eviction-policy", policy,
        "--enable-metrics",
        "--decode-log-interval", "50",
    ]
    print(f"[server] Starting with --radix-eviction-policy {policy}", flush=True)
    server = subprocess.Popen(cmd, stdout=sys.stdout, stderr=sys.stderr)
    _wait_ready(server)

    capacity = _get_cache_capacity()
    print(f"[server] Ready. Cache capacity: {capacity:,} tokens", flush=True)

    # Run bench_serving against localhost — no proxy, no network issues
    output_file = f"/tmp/results_{policy}.json"
    bench_cmd = [
        "python", "-m", "sglang.bench_serving",
        "--backend", "sglang",
        "--base-url", f"http://127.0.0.1:{PORT}",
        "--model", MODEL_NAME,
        "--dataset-name", "generated-shared-prefix",
        "--num-prompts", str(BENCH_PARAMS["num_prompts"]),
        "--request-rate", str(BENCH_PARAMS["request_rate"]),
        "--gsp-num-groups", str(BENCH_PARAMS["gsp_num_groups"]),
        "--gsp-prompts-per-group", str(BENCH_PARAMS["gsp_prompts_per_group"]),
        "--gsp-system-prompt-len", str(BENCH_PARAMS["gsp_system_prompt_len"]),
        "--gsp-question-len", str(BENCH_PARAMS["gsp_question_len"]),
        "--gsp-output-len", str(BENCH_PARAMS["gsp_output_len"]),
        "--output-file", output_file,
        "--seed", "42",
    ]
    print(f"\n[bench] Starting benchmark...", flush=True)
    subprocess.run(bench_cmd, check=True, stderr=subprocess.STDOUT)

    # Wait for in-flight requests to drain, then check tree state
    time.sleep(10)
    pool_usage = None
    metrics_text = _req.get(f"http://127.0.0.1:{PORT}/metrics").text
    for line in metrics_text.splitlines():
        if "sglang:token_usage{" in line:
            pool_usage = float(line.split()[-1])
            break
    print(f"[metrics] Post-benchmark pool usage: {pool_usage:.3f} "
          f"(>0 means tree holds cached tokens)", flush=True)

    # Collect metrics
    evictions = _get_eviction_count()
    hit_rate = _get_cache_hit_rate()
    print(f"\n[metrics] Evictions: {evictions} | Cache hit rate: {hit_rate:.3f}", flush=True)

    server.terminate()
    server.wait()

    with open(output_file) as f:
        # file is JSONL, take last line (most recent run)
        results = json.loads(f.readlines()[-1])

    results["eviction_count"] = evictions
    results["cache_hit_rate_final"] = hit_rate
    results["cache_capacity_tokens"] = capacity
    results["eviction_policy"] = policy
    return results


# ---------------------------------------------------------------------------
# Local entrypoint: run both policies and print comparison
# ---------------------------------------------------------------------------
@app.local_entrypoint()
def main(policy: str = "both"):
    policies = ["lru", "wlfu"] if policy == "both" else [policy]

    all_results = {}
    for p in policies:
        result = run_policy.remote(p)
        all_results[p] = result
        # Save locally
        with open(f"results_{p}.json", "w") as f:
            json.dump(result, f, indent=2)
        print(f"\nSaved results_{p}.json")

    if len(all_results) == 2:
        lru = all_results["lru"]
        wlfu = all_results["wlfu"]
        print(f"\n{'='*60}")
        print(f"  COMPARISON SUMMARY")
        print(f"{'='*60}")
        print(f"{'Metric':<35} {'LRU':>12} {'WLFU':>12} {'Delta':>10}")
        print(f"{'-'*70}")

        metrics = [
            ("Output throughput (tok/s)",    "output_throughput"),
            ("Request throughput (req/s)",    "request_throughput"),
            ("Mean TTFT (ms)",               "mean_ttft_ms"),
            ("P99 TTFT (ms)",                "p99_ttft_ms"),
            ("Mean E2E latency (ms)",         "mean_e2e_latency_ms"),
            ("Cache hit rate (final)",        "cache_hit_rate_final"),
            ("Eviction count",               "eviction_count"),
        ]
        for label, key in metrics:
            lv = lru.get(key, 0) or 0
            wv = wlfu.get(key, 0) or 0
            if lv != 0:
                delta = f"{(wv - lv) / lv * 100:+.1f}%"
            else:
                delta = "N/A"
            print(f"{label:<35} {lv:>12.2f} {wv:>12.2f} {delta:>10}")
        print(f"{'='*60}\n")
