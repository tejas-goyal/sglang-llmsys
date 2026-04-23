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
import os
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
SGLANG_BENCH_SERVING = "/sgl-workspace/sglang/python/sglang/bench_serving.py"
MINUTES = 60

# --------------------------------------------------------------------------- #
# Pluggable human-readable corpus sources.                                    #
# `generated-shared-prefix` draws random tokens from the vocab by default;    #
# setting --dataset <name> flips SGLANG_EVAL_GEN_PROMPT_SOURCE=human inside   #
# the bench subprocess and points SGLANG_EVAL_HUMAN_CORPUS at a pre-baked     #
# text file on the huggingface-cache Modal volume. One dict entry per HF      #
# source; files are cached to <HF_CACHE_PATH>/eval/<name>.txt so A/B between  #
# sources on the same volume never reuses a stale corpus.                    #
# --------------------------------------------------------------------------- #
HUMAN_CORPUS_SOURCES: dict[str, dict] = {
    "wikitext": {
        "path": "wikitext",
        "name": "wikitext-103-raw-v1",
        "split": "train[:20000]",
        "text_field": "text",
    },
    # Add more sources here; one dict entry per new HF dataset. Example:
    # "c4":          {"path": "allenai/c4",             "name": "en",   "split": "train[:20000]", "text_field": "text"},
    # "openwebtext": {"path": "Skylion007/openwebtext", "name": None,   "split": "train[:20000]", "text_field": "text"},
}
HUMAN_ALIASES = {"human": "wikitext"}  # readable CLI aliases

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
    # Human-readable prompt source: install `datasets` + patch gen_prompt in
    # v0.5.9's monolithic bench_serving.py. Corpus files are downloaded lazily
    # at runtime into the huggingface-cache volume; nothing gets baked in.
    .pip_install("datasets")
    .add_local_file(
        str(EVAL_DIR / "human_dataset_patcher.py"),
        remote_path="/tmp/eval/human_dataset_patcher.py",
        copy=True,
    )
    .run_commands(f"python3 /tmp/eval/human_dataset_patcher.py {SGLANG_BENCH_SERVING}")
)

# --------------------------------------------------------------------------- #
# Bench config — identical to eval_eviction.py for historical comparability   #
# --------------------------------------------------------------------------- #
MODEL_NAME = "Qwen/Qwen2.5-7B-Instruct"
MODEL_REVISION: str | None = None
PORT = 8000
MEM_FRACTION = 0.85
# Read at module-import time so `@app.function(gpu=...)` below is bound before
# Modal serializes the function. Bash callers (see sweep.sh) set EVAL_GPU per
# `modal run` invocation to target H100 / H200 / B200 without editing source.
# Default preserves historical A100-40GB behaviour for every example in README.
DEFAULT_GPU = os.environ.get("EVAL_GPU", "A100-40GB")

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
HUMAN_CORPUS_DIR = f"{HF_CACHE_PATH}/eval"


def _corpus_path(name: str) -> str:
    return f"{HUMAN_CORPUS_DIR}/{name}.txt"


def _resolve_dataset(dataset: str) -> str:
    """Canonicalize CLI dataset name. 'synthetic' -> 'synthetic'; aliases -> registry key."""
    if dataset == "synthetic":
        return dataset
    name = HUMAN_ALIASES.get(dataset, dataset)
    if name not in HUMAN_CORPUS_SOURCES:
        valid = sorted({"synthetic", *HUMAN_ALIASES, *HUMAN_CORPUS_SOURCES})
        raise ValueError(f"--dataset must be one of {valid}, got {dataset!r}")
    return name


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


def _ensure_corpus(name: str) -> str:
    """Lazily download the selected HF text source onto the huggingface-cache volume.

    First call per volume downloads and commits; subsequent calls are no-ops
    that just return the path. Each source caches to its own `<name>.txt` so
    switching --dataset between sources never reuses stale text.
    """
    import pathlib

    spec = HUMAN_CORPUS_SOURCES[name]
    p = pathlib.Path(_corpus_path(name))
    if p.exists() and p.stat().st_size > 0:
        print(
            f"[dataset] Reusing cached {name} corpus at {p} "
            f"({p.stat().st_size // 1024} KB)",
            flush=True,
        )
        return str(p)

    print(
        f"[dataset] Downloading {name} ({spec['path']}"
        + (f"/{spec['name']}" if spec.get("name") else "")
        + f", split={spec['split']}) to {p}...",
        flush=True,
    )
    from datasets import load_dataset

    ds = load_dataset(
        spec["path"],
        spec.get("name"),
        split=spec["split"],
        cache_dir=f"{HF_CACHE_PATH}/datasets",
    )
    field = spec["text_field"]
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text("".join(r.get(field, "") or "" for r in ds))
    HF_CACHE_VOL.commit()
    print(
        f"[dataset] Wrote {p.stat().st_size // 1024} KB to {p} and committed volume.",
        flush=True,
    )
    return str(p)


# --------------------------------------------------------------------------- #
# Container entrypoint                                                        #
# --------------------------------------------------------------------------- #
@app.function(
    image=sglang_image,
    gpu=DEFAULT_GPU,
    volumes={HF_CACHE_PATH: HF_CACHE_VOL},
    timeout=90 * MINUTES,
)
def run_policies(
    policies: list[str],
    dataset: str = "synthetic",
    model: str = MODEL_NAME,
    mem_fraction: float = MEM_FRACTION,
) -> dict:
    """Run each policy end-to-end in this container. Return {policy: results_dict}."""
    import requests as req  # noqa: F401  (keeps imports adjacent to remote context)

    resolved = _resolve_dataset(dataset)
    bench_env: dict[str, str] | None = None
    if resolved != "synthetic":
        corpus = _ensure_corpus(resolved)
        bench_env = {
            **os.environ,
            "SGLANG_EVAL_GEN_PROMPT_SOURCE": "human",
            "SGLANG_EVAL_HUMAN_CORPUS": corpus,
        }
        print(f"[dataset] Active human corpus: {resolved} -> {corpus}", flush=True)

    out: dict[str, dict] = {}
    capacity_tokens: int | None = None

    for policy in policies:
        print(f"\n{'=' * 60}\n  Running policy: {policy}\n{'=' * 60}\n", flush=True)

        revision_args = ["--revision", MODEL_REVISION] if MODEL_REVISION else []
        cmd = [
            "python", "-m", "sglang.launch_server",
            "--model-path", model,
            *revision_args,
            "--served-model-name", model,
            "--host", "0.0.0.0",
            "--port", str(PORT),
            "--tp", "1",
            "--mem-fraction", str(mem_fraction),
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
            # Warmup + flush removes position-in-list bias across policies:
            # a burst of warmup requests forces JIT compilation of every prefill /
            # attention / sampling kernel shape the main run will hit, then
            # --flush-cache clears the radix tree so measurement starts from an
            # empty cache for every policy (same state as a cold server) while
            # disk-level Triton/inductor caches stay warm.
            bench_cmd = [
                "python", "-m", "sglang.bench_serving",
                "--backend", "sglang",
                "--base-url", f"http://127.0.0.1:{PORT}",
                "--model", model,
                "--dataset-name", "generated-shared-prefix",
                "--num-prompts", str(BENCH_PARAMS["num_prompts"]),
                "--request-rate", str(BENCH_PARAMS["request_rate"]),
                "--gsp-num-groups", str(BENCH_PARAMS["gsp_num_groups"]),
                "--gsp-prompts-per-group", str(BENCH_PARAMS["gsp_prompts_per_group"]),
                "--gsp-system-prompt-len", str(BENCH_PARAMS["gsp_system_prompt_len"]),
                "--gsp-question-len", str(BENCH_PARAMS["gsp_question_len"]),
                "--gsp-output-len", str(BENCH_PARAMS["gsp_output_len"]),
                "--warmup-requests", str(BENCH_PARAMS["gsp_num_groups"]),
                "--flush-cache",
                "--output-file", output_file,
                "--seed", "42",
            ]
            subprocess.run(bench_cmd, check=True, stderr=subprocess.STDOUT, env=bench_env)
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
def main(
    policies: str = "lru,slru,wlfu",
    tag: str = "sweep",
    dataset: str = "synthetic",
    model: str = MODEL_NAME,
    gpu: str = DEFAULT_GPU,
    mem_fraction: float = MEM_FRACTION,
) -> None:
    names = [p.strip() for p in policies.split(",") if p.strip()]
    if not names:
        raise SystemExit("--policies must be a non-empty CSV (e.g. 'lru,wlfu')")
    resolved_dataset = _resolve_dataset(dataset)

    ts = _dt.datetime.now().strftime("%Y-%m-%dT%H-%M")
    run_dir = RESULTS_DIR / f"{ts}_{tag}"
    run_dir.mkdir(parents=True, exist_ok=True)

    print(f"[runner] Policies: {names}")
    print(f"[runner] Dataset:  {dataset} (resolved: {resolved_dataset})")
    print(f"[runner] Model:    {model}")
    print(f"[runner] GPU:      {gpu}  mem_fraction={mem_fraction}")
    print(f"[runner] Results dir: {run_dir}")

    # `gpu` arg is logged in config.json for traceability. The ACTUAL GPU
    # binding was resolved at module import from EVAL_GPU; warn loudly on
    # mismatch so a CLI user never thinks they got a different GPU than they
    # actually did.
    if gpu != DEFAULT_GPU:
        raise SystemExit(
            f"--gpu={gpu!r} does not match EVAL_GPU env ({DEFAULT_GPU!r}). "
            f"To bind a different GPU, prepend `EVAL_GPU={gpu}` to the modal "
            f"run invocation."
        )
    all_results = run_policies.remote(names, resolved_dataset, model, mem_fraction)
    capacity = all_results.pop("_capacity_tokens", None)

    config = {
        "timestamp": ts,
        "tag": tag,
        "policies": names,
        "dataset": resolved_dataset,
        "model": model,
        "gpu": gpu,
        "mem_fraction": mem_fraction,
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
