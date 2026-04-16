"""Generic Modal runner for KV cache eviction policy ablations.

Runs N policies back-to-back in a single container so results are directly
comparable (same cold-start state, same HF cache, same GPU).

Usage:
    modal run sglang/eval/runner.py --policies wlfu,wslru --tag wslru_vs_wlfu
    modal run sglang/eval/runner.py --policies lru,slru,wlfu --tag baseline_sweep

Results land in sglang/eval/results/<timestamp>_<tag>/ with:
    config.json        -- bench params, model, capacity, sglang git SHA
    <policy>.json      -- raw bench_serving output + eviction metrics
    summary.md         -- comparison table (generated via summarize.py)
"""

from __future__ import annotations

import datetime as _dt
import json
import subprocess
import sys
import time
from pathlib import Path

import modal

# --------------------------------------------------------------------------- #
# Paths                                                                       #
# --------------------------------------------------------------------------- #
EVAL_DIR = Path(__file__).parent
REPO_ROOT = EVAL_DIR.parent.parent  # .../project (eval lives at project/sglang/eval)
LOCAL_EVICT_POLICY = (
    EVAL_DIR.parent / "python" / "sglang" / "srt" / "mem_cache" / "evict_policy.py"
)
RESULTS_DIR = EVAL_DIR / "results"

SGLANG_SRT = "/sgl-workspace/sglang/python/sglang/srt"
MINUTES = 60

# --------------------------------------------------------------------------- #
# Image: overlay our evict_policy.py, then run the registry-aware patcher.    #
# --------------------------------------------------------------------------- #
sglang_image = (
    modal.Image.from_registry("lmsysorg/sglang:v0.5.9-cu129-amd64-runtime")
    .entrypoint([])
    .env(
        {
            "HF_HUB_CACHE": "/root/.cache/huggingface",
            "HF_XET_HIGH_PERFORMANCE": "1",
            "SGLANG_ENABLE_JIT_DEEPGEMM": "0",
        }
    )
    .add_local_file(
        str(LOCAL_EVICT_POLICY),
        remote_path=f"{SGLANG_SRT}/mem_cache/evict_policy.py",
        copy=True,
    )
    .add_local_file(str(EVAL_DIR / "registry.py"), remote_path="/tmp/eval/registry.py", copy=True)
    .add_local_file(str(EVAL_DIR / "patcher.py"), remote_path="/tmp/eval/patcher.py", copy=True)
    .run_commands(f"python3 /tmp/eval/patcher.py {SGLANG_SRT}")
)

# --------------------------------------------------------------------------- #
# Bench config — identical to eval_eviction.py for historical comparability   #
# --------------------------------------------------------------------------- #
MODEL_NAME = "Qwen/Qwen2.5-7B-Instruct"
MODEL_REVISION: str | None = None
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

app = modal.App("eviction-eval")


# --------------------------------------------------------------------------- #
# Helpers (run inside container)                                              #
# --------------------------------------------------------------------------- #
def _wait_ready(process, timeout=8 * MINUTES):
    import requests as req

    deadline = time.time() + timeout
    while time.time() < deadline:
        if (rc := process.poll()) is not None:
            raise subprocess.CalledProcessError(rc, cmd=process.args)
        try:
            req.get(f"http://127.0.0.1:{PORT}/health").raise_for_status()
            return
        except Exception:
            time.sleep(5)
    raise TimeoutError(f"SGLang not ready after {timeout}s")


def _metric(pattern: str, cast=float, default=0.0):
    import requests as req

    text = req.get(f"http://127.0.0.1:{PORT}/metrics").text
    for line in text.splitlines():
        if pattern in line and not line.startswith("#"):
            return cast(float(line.split()[-1]))
    return default


# --------------------------------------------------------------------------- #
# Container entrypoint                                                        #
# --------------------------------------------------------------------------- #
@app.function(
    image=sglang_image,
    gpu="A100",
    volumes={HF_CACHE_PATH: HF_CACHE_VOL},
    timeout=90 * MINUTES,
)
def run_policies(policies: list[str]) -> dict:
    """Run each policy end-to-end in this container. Return {policy: results_dict}."""
    import requests as req  # noqa: F401  (keeps imports adjacent to remote context)

    out: dict[str, dict] = {}
    capacity_tokens: int | None = None

    for policy in policies:
        print(f"\n{'=' * 60}\n  Running policy: {policy}\n{'=' * 60}\n", flush=True)

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
        server = subprocess.Popen(cmd, stdout=sys.stdout, stderr=sys.stderr)
        try:
            _wait_ready(server)
            capacity = int(_metric("max_total_num_tokens", default=0))
            capacity_tokens = capacity_tokens or capacity
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
            subprocess.run(bench_cmd, check=True, stderr=subprocess.STDOUT)
            time.sleep(10)

            evictions = int(_metric("eviction_duration_seconds_count", default=0))
            hit_rate = _metric("sglang:cache_hit_rate{", default=0.0)
            print(f"[metrics] Evictions: {evictions} | hit_rate: {hit_rate:.3f}", flush=True)

            with open(output_file) as f:
                res = json.loads(f.readlines()[-1])
            res.update(
                eviction_policy=policy,
                eviction_count=evictions,
                cache_hit_rate_final=hit_rate,
                cache_capacity_tokens=capacity,
            )
            out[policy] = res
        finally:
            server.terminate()
            server.wait()

    out["_capacity_tokens"] = capacity_tokens
    return out


# --------------------------------------------------------------------------- #
# Local entrypoint                                                            #
# --------------------------------------------------------------------------- #
def _git_sha() -> str:
    try:
        return subprocess.check_output(
            ["git", "-C", str(EVAL_DIR.parent), "rev-parse", "HEAD"],
            stderr=subprocess.DEVNULL,
        ).decode().strip()
    except Exception:
        return "unknown"


@app.local_entrypoint()
def main(policies: str = "lru,slru,wlfu", tag: str = "sweep") -> None:
    names = [p.strip() for p in policies.split(",") if p.strip()]
    if not names:
        raise SystemExit("--policies must be a non-empty CSV (e.g. 'lru,wlfu')")

    ts = _dt.datetime.now().strftime("%Y-%m-%dT%H-%M")
    run_dir = RESULTS_DIR / f"{ts}_{tag}"
    run_dir.mkdir(parents=True, exist_ok=True)

    print(f"[runner] Policies: {names}")
    print(f"[runner] Results dir: {run_dir}")

    all_results = run_policies.remote(names)
    capacity = all_results.pop("_capacity_tokens", None)

    config = {
        "timestamp": ts,
        "tag": tag,
        "policies": names,
        "model": MODEL_NAME,
        "mem_fraction": MEM_FRACTION,
        "bench_params": BENCH_PARAMS,
        "cache_capacity_tokens": capacity,
        "sglang_git_sha": _git_sha(),
    }
    (run_dir / "config.json").write_text(json.dumps(config, indent=2))
    for policy, res in all_results.items():
        (run_dir / f"{policy}.json").write_text(json.dumps(res, indent=2))

    # Emit summary.md
    sys.path.insert(0, str(EVAL_DIR))
    from summarize import render_summary  # noqa: E402

    (run_dir / "summary.md").write_text(render_summary(run_dir))
    print(f"\n[runner] Done. See {run_dir}/summary.md")
    print((run_dir / "summary.md").read_text())
