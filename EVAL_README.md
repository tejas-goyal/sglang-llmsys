# KV Cache Eviction Policy Eval — CMU LLM Systems Project

We benchmark alternative KV cache eviction strategies in sglang's radix trie
against the LRU baseline under concurrent shared-prefix workloads.

## What we changed

`python/sglang/srt/mem_cache/evict_policy.py` — added `WLFUStrategy`:

```
priority = last_access_time - alpha * log(1 + hit_count)
```

Frequently-hit nodes (e.g. shared system prompts) get a virtual recency bonus,
keeping them in cache longer during eviction pressure.

Two other files are patched at image-build time by `_patch_sglang.py`:
- `server_args.py` — adds `"wlfu"` to `--radix-eviction-policy` choices
- `mem_cache/radix_cache.py` — registers `WLFUStrategy` in the dispatch block

## Setup

```bash
# requires Modal account (modal.com) and uv
uv venv --python=3.12 && source .venv/bin/activate
uv pip install modal sglang
modal setup   # links your Modal account
```

All eval code lives in the repo root (one level above this directory):
- `eval_eviction.py` — Modal function: starts sglang + runs bench inside same container
- `_patch_sglang.py` — in-place patch for the Docker image's sglang install
- `run_bench.py` — thin wrapper if you want to hit a deployed web server instead

## Run the eval

```bash
# Run both LRU and WLFU sequentially on the same GPU, print comparison table
modal run eval_eviction.py

# Run a single policy
modal run eval_eviction.py --policy lru
modal run eval_eviction.py --policy wlfu
```

Results are saved to `results_lru.json` and `results_wlfu.json` in the working directory.

## Key results (Qwen2.5-7B-Instruct, A100-40GB, 2048 requests)

| Metric          | LRU     | WLFU    | Delta   |
|-----------------|---------|---------|---------|
| Mean TTFT       | 104 ms  | 70 ms   | **-33%** |
| P99 TTFT        | 1039 ms | 410 ms  | **-61%** |
| Mean E2E        | 1655 ms | 1502 ms | -9%     |
| Output tok/s    | 996     | 997     | 0%      |
| Evictions       | 574     | 576     | ~same   |

## Swapping model or GPU

Edit these two lines at the top of `eval_eviction.py`:

```python
MODEL_NAME = "Qwen/Qwen2.5-7B-Instruct"   # any HF model ID
MEM_FRACTION = 0.85                         # fraction of GPU RAM for KV cache
```

```python
@app.function(
    gpu="A100",     # "H100", "A10G", "L4", etc. — see modal.com/pricing
    ...
)
```

**Model requirements**: must be a **pure transformer** (no Mamba/SSM layers).
Hybrid SSM models use `MambaRadixCache` which has its own hardcoded LRU and
ignores the `--radix-eviction-policy` flag.

**Workload calibration**: evictions only fire when the workload exceeds the
cache capacity. After changing the model/GPU, check
`max_total_num_tokens` in `/metrics` and set:

```python
BENCH_PARAMS = {
    "gsp_num_groups": N,              # N × system_prompt_len > max_total_num_tokens
    "gsp_system_prompt_len": 1024,
    ...
}
```

## Other eviction policies to try

All of these exist in `evict_policy.py` and are selectable via `--radix-eviction-policy`:

| Policy | Idea | Expected behaviour |
|--------|------|--------------------|
| `lfu`  | Evict least-frequently used first | Protects hot nodes; can strand one-shot nodes |
| `slru` | Two-segment LRU (probationary + protected) | Good default; threshold is fixed |
| **`wlfu`** | **LRU + log(freq) bonus (this work)** | **Continuous frequency weighting** |
| Depth-weighted LRU | Penalise shallow nodes (cheap to recompute) | Protects deep, expensive prefixes |
| Adaptive SLRU | Dynamic threshold based on observed hit counts | Self-tunes to workload |

The next two (depth-weighted, adaptive SLRU) are the natural follow-ons.
They require only a new `EvictionStrategy` subclass — no other changes.
