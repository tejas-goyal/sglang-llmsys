# KV Cache Eviction Eval Harness

Single-branch, registry-driven ablation runner for SGLang radix-cache eviction
policies. Every new policy is: one class in `evict_policy.py` + one line in
`registry.py`. No per-policy scripts.

## Layout

```
sglang/eval/
├── registry.py         # POLICY_REGISTRY: name -> strategy class
├── patcher.py          # AST-based, idempotent v0.5.9 container patch
├── runner.py           # Modal app — runs N policies back-to-back
├── summarize.py        # Renders summary.md from a results dir
└── results/
    └── <timestamp>_<tag>/
        ├── config.json       # bench params, model, capacity, git SHA
        ├── <policy>.json     # raw bench_serving + eviction metrics
        └── summary.md        # comparison table
```

## Run

```bash
# Sanity check: baseline + the two policies we already shipped
modal run sglang/eval/runner.py --policies lru,slru,wlfu --tag baseline_sweep

# A/B a new policy against WLFU
modal run sglang/eval/runner.py --policies wlfu,wslru --tag wslru_vs_wlfu

# Single policy
modal run sglang/eval/runner.py --policies wlfu --tag wlfu_only
```

Results land in `sglang/eval/results/<timestamp>_<tag>/`. The first policy in
the CSV is the baseline for delta columns in `summary.md`.

## Add a new policy

1. Add a subclass of `EvictionStrategy` in
   `sglang/python/sglang/srt/mem_cache/evict_policy.py`:

   ```python
   class WSLRUStrategy(EvictionStrategy):
       def __init__(self, alpha: float = 2.0, threshold: int = 2):
           self.alpha = alpha
           self.threshold = threshold

       def get_priority(self, node):
           segment = 1 if node.hit_count >= self.threshold else 0
           return (segment, node.last_access_time - self.alpha * math.log1p(node.hit_count))
   ```

2. Add one line to `registry.py`:

   ```python
   "wslru": {"class_name": "WSLRUStrategy", "description": "Hybrid SLRU×WLFU"},
   ```

3. Run:

   ```bash
   modal run sglang/eval/runner.py --policies wlfu,wslru --tag wslru_vs_wlfu
   ```

Stock v0.5.9 policies (`lru`, `lfu`, `fifo`, `mru`, `filo`, `priority`) need no
registration — they're already in the container.

## How the patcher works

`patcher.py` runs once at Modal image build time. It uses `ast` (not string
search-and-replace) to:

- Append missing policy names to `RADIX_EVICTION_POLICY_CHOICES` in
  `server_args.py`.
- Insert missing `elif self.eviction_policy == "<name>":` branches in
  `RadixCache.__init__` in `radix_cache.py`, plus any missing imports from
  `evict_policy`.

It re-parses both files after editing and aborts the image build on syntax
errors. Idempotent: re-running adds nothing if the policy is already wired.

## Gotchas

- Must use a **pure transformer model** (Qwen2.5-7B). Mamba hybrids route to
  `MambaRadixCache` which bypasses our dispatch.
- `SGLANG_ENABLE_JIT_DEEPGEMM=0` must stay in the image env or startup times
  out on FP8 kernel compilation.
- `cache_hit_rate` reads 0 at benchmark end (server idle); check server logs
  for `#cached-token` during prefill to confirm hits.
- Workload should overflow the cache by ~50% to force meaningful evictions.
  At `mem_fraction=0.85` on 1×A100-40GB with Qwen2.5-7B, cache ≈ 87K tokens;
  128 groups × 1024-tok system prompts = 131K prefix tokens → ~33% evicted.
