"""Patch sglang v0.5.9 to add both SLRU and WLFU policies to RadixCache.

The Docker image's v0.5.9 RadixCache only supports:
  ['lru', 'lfu', 'fifo', 'mru', 'filo', 'priority']

Our overlaid evict_policy.py contains both SLRUStrategy and WLFUStrategy,
but radix_cache.py's __init__ dispatch block and server_args.py's CLI
choices list need to be patched to register them.
"""
import pathlib
import sys

SGLANG_SRT = pathlib.Path(sys.argv[1])

# 1. server_args.py — add "slru" and "wlfu" to CLI choices
sa = SGLANG_SRT / "server_args.py"
t = sa.read_text()
changed = False
if "slru" not in t:
    t = t.replace(
        "RADIX_EVICTION_POLICY_CHOICES = [",
        'RADIX_EVICTION_POLICY_CHOICES = ["slru", ',
    )
    changed = True
if "wlfu" not in t:
    t = t.replace(
        "RADIX_EVICTION_POLICY_CHOICES = [",
        'RADIX_EVICTION_POLICY_CHOICES = ["wlfu", ',
    )
    changed = True
if changed:
    sa.write_text(t)
    print("Updated server_args.py CLI choices")

# 2. radix_cache.py — register SLRUStrategy and WLFUStrategy
rc = SGLANG_SRT / "mem_cache" / "radix_cache.py"
t = rc.read_text()

# Add imports if missing
imports_to_add = []
if "SLRUStrategy" not in t:
    imports_to_add.append("SLRUStrategy")
if "WLFUStrategy" not in t:
    imports_to_add.append("WLFUStrategy")

if imports_to_add:
    import_line = ",\n    ".join(imports_to_add)
    t = t.replace(
        "from sglang.srt.mem_cache.evict_policy import (",
        f"from sglang.srt.mem_cache.evict_policy import (\n    {import_line},",
    )

# Add elif branches before the final else
if '"slru":' not in t and "SLRUStrategy" in imports_to_add:
    t = t.replace(
        "        else:\n            raise ValueError(",
        '        elif self.eviction_policy == "slru":\n'
        "            self.eviction_strategy = SLRUStrategy()\n"
        "        else:\n            raise ValueError(",
    )

if '"wlfu":' not in t and "WLFUStrategy" in imports_to_add:
    t = t.replace(
        "        else:\n            raise ValueError(",
        '        elif self.eviction_policy == "wlfu":\n'
        "            self.eviction_strategy = WLFUStrategy()\n"
        "        else:\n            raise ValueError(",
    )

if imports_to_add:
    rc.write_text(t)
    print(f"Registered {imports_to_add} in radix_cache.py")
else:
    print("SLRU and WLFU already registered in radix_cache.py")

print("Patch complete: slru and wlfu available via --radix-eviction-policy")
