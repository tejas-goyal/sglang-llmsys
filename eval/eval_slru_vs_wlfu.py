"""
Eval: compare SLRU vs WLFU KV cache eviction strategies.

SLRU (Segmented LRU) is the best built-in policy in sglang — it divides
nodes into a probationary segment (hit_count < 2) and a protected segment
(hit_count >= 2), always evicting probationary before protected.

WLFU (Weighted LFU-LRU) is our policy — it uses a continuous frequency
bonus: priority = last_access_time - alpha * log(1 + hit_count).

This eval runs both back-to-back in the same container for a fair comparison.

Usage:
  modal run eval_slru_vs_wlfu.py            # runs slru then wlfu, prints comparison
  modal run eval_slru_vs_wlfu.py --policy slru
  modal run eval_slru_vs_wlfu.py --policy wlfu
"""

import json
import subprocess
import sys
import time
from pathlib import Path

import modal

MINUTES = 60

# ---------------------------------------------------------------------------
# Image: overlay evict_policy.py (has WLFUStrategy) + patch for wlfu dispatch
# ---------------------------------------------------------------------------
LOCAL_SGLANG = Path(__file__).parent / "sglang"
LOCAL_PATCH = str(Path(__file__).parent / "_patch_sglang_slru.py")
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
# Config — identical to eval_eviction.py for direct comparability
# ---------------------------------------------------------------------------
MODEL_NAME = "Qwen/Qwen2.5-7B-Instruct"
MODEL_REVISION = None
PORT = 8000
MEM_FRACTION = 0.85

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

app = modal.App("eval-slru-vs-wlfu")


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
# Core experiment function
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
    print(f"  Workload: {BENCH_PARAMS['gsp_num_groups']} groups × "
          f"{BENCH_PARAMS['gsp_system_prompt_len']} tokens = "
          f"{BENCH_PARAMS['gsp_num_groups'] * BENCH_PARAMS['gsp_system_prompt_len']:,} prefix tokens")
    print(f"{'='*60}\n")

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

    time.sleep(10)
    metrics_text = _req.get(f"http://127.0.0.1:{PORT}/metrics").text
    pool_usage = None
    for line in metrics_text.splitlines():
        if "sglang:token_usage{" in line:
            pool_usage = float(line.split()[-1])
            break
    print(f"[metrics] Post-benchmark pool usage: {pool_usage:.3f}", flush=True)

    evictions = _get_eviction_count()
    hit_rate = _get_cache_hit_rate()
    print(f"[metrics] Evictions: {evictions} | Cache hit rate: {hit_rate:.3f}", flush=True)

    server.terminate()
    server.wait()

    with open(output_file) as f:
        results = json.loads(f.readlines()[-1])

    results["eviction_count"] = evictions
    results["cache_hit_rate_final"] = hit_rate
    results["cache_capacity_tokens"] = capacity
    results["eviction_policy"] = policy
    return results


# ---------------------------------------------------------------------------
# Local entrypoint
# ---------------------------------------------------------------------------
@app.local_entrypoint()
def main(policy: str = "both"):
    policies = ["slru", "wlfu"] if policy == "both" else [policy]

    all_results = {}
    for p in policies:
        result = run_policy.remote(p)
        all_results[p] = result
        with open(f"results_{p}.json", "w") as f:
            json.dump(result, f, indent=2)
        print(f"\nSaved results_{p}.json")

    if len(all_results) == 2:
        slru = all_results["slru"]
        wlfu = all_results["wlfu"]
        print(f"\n{'='*60}")
        print(f"  COMPARISON SUMMARY  (SLRU vs WLFU)")
        print(f"{'='*60}")
        print(f"{'Metric':<35} {'SLRU':>12} {'WLFU':>12} {'Delta':>10}")
        print(f"{'-'*70}")

        metrics = [
            ("Output throughput (tok/s)",  "output_throughput"),
            ("Request throughput (req/s)", "request_throughput"),
            ("Mean TTFT (ms)",             "mean_ttft_ms"),
            ("P99 TTFT (ms)",              "p99_ttft_ms"),
            ("Mean E2E latency (ms)",       "mean_e2e_latency_ms"),
            ("Cache hit rate (final)",      "cache_hit_rate_final"),
            ("Eviction count",             "eviction_count"),
        ]
        for label, key in metrics:
            sv = slru.get(key, 0) or 0
            wv = wlfu.get(key, 0) or 0
            delta = f"{(wv - sv) / sv * 100:+.1f}%" if sv != 0 else "N/A"
            print(f"{label:<35} {sv:>12.2f} {wv:>12.2f} {delta:>10}")
        print(f"{'='*60}\n")
