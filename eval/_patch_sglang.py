"""Patch sglang v0.5.9 to recognize the 'wlfu' eviction policy."""
import pathlib
import sys

SGLANG_SRT = pathlib.Path(sys.argv[1])

# 1. server_args.py: add "wlfu" to allowed CLI choices
sa = SGLANG_SRT / "server_args.py"
t = sa.read_text()
if "wlfu" not in t:
    t = t.replace(
        "RADIX_EVICTION_POLICY_CHOICES = [",
        'RADIX_EVICTION_POLICY_CHOICES = ["wlfu", ',
    )
    sa.write_text(t)

# 2. radix_cache.py: add WLFUStrategy import + elif branch
rc = SGLANG_SRT / "mem_cache" / "radix_cache.py"
t = rc.read_text()
if "WLFUStrategy" not in t:
    t = t.replace(
        "from sglang.srt.mem_cache.evict_policy import (",
        "from sglang.srt.mem_cache.evict_policy import (\n    WLFUStrategy,",
    )
    t = t.replace(
        "        else:\n            raise ValueError(",
        '        elif self.eviction_policy == "wlfu":\n'
        "            self.eviction_strategy = WLFUStrategy()\n"
        "        else:\n            raise ValueError(",
    )
    rc.write_text(t)

print("Patched server_args.py and radix_cache.py for wlfu support")
