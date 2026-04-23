"""Registry of eviction policies available to the eval harness.

Adding a new policy:
    1. Implement a subclass of EvictionStrategy in
       python/sglang/srt/mem_cache/evict_policy.py
    2. Add one entry below mapping the CLI name -> strategy class.

The patcher reads this registry and injects each entry into the
v0.5.9 container's server_args.py (CLI choices) and radix_cache.py
(dispatch branch) at image build time.

Policies that ship with stock v0.5.9 ("lru", "lfu", "fifo", "mru",
"filo", "priority") are NOT listed here — they need no patching.
"""

POLICY_REGISTRY = {
    "slru": {
        "class_name": "SLRUStrategy",
        "description": "Segmented LRU: probationary (hit<2) evicted before protected.",
    },
    "wlfu": {
        "class_name": "WLFUStrategy",
        "description": "Weighted LFU-LRU: priority = last_access - alpha*log1p(hit_count).",
    },
    "wslru": {
        "class_name": "WSLRUStrategy",
        "description": "Hybrid SLRU×WLFU: segment gate (hit>=2) with WLFU ordering within segment.",
    },
    "calru": {
        "class_name": "CALRUStrategy",
        "description": "Cost-Aware LRU: priority = last_access + alpha*log1p(hit_count*len(node)).",
    },
    "scalru": {
        "class_name": "SCALRUStrategy",
        "description": "Size-gated CALRU: protect iff hit_count*len >= cost_threshold, CALRU tiebreak.",
    },
}

STOCK_POLICIES = {"lru", "lfu", "fifo", "mru", "filo", "priority"}


def all_policies() -> list[str]:
    return sorted(STOCK_POLICIES | POLICY_REGISTRY.keys())
