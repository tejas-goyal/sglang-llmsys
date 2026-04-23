from __future__ import annotations

import math
from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Tuple, Union

if TYPE_CHECKING:
    from sglang.srt.mem_cache.radix_cache import TreeNode


class EvictionStrategy(ABC):
    @abstractmethod
    def get_priority(self, node: "TreeNode") -> Union[float, Tuple]:
        pass


class LRUStrategy(EvictionStrategy):
    def get_priority(self, node: "TreeNode") -> float:
        return node.last_access_time


class LFUStrategy(EvictionStrategy):
    def get_priority(self, node: "TreeNode") -> Tuple[int, float]:
        return (node.hit_count, node.last_access_time)


class FIFOStrategy(EvictionStrategy):
    def get_priority(self, node: "TreeNode") -> float:
        return node.creation_time


class MRUStrategy(EvictionStrategy):
    def get_priority(self, node: "TreeNode") -> float:
        return -node.last_access_time


class FILOStrategy(EvictionStrategy):
    def get_priority(self, node: "TreeNode") -> float:
        return -node.creation_time


class PriorityStrategy(EvictionStrategy):
    """Priority-aware eviction: lower priority values evicted first, then LRU within same priority."""

    def get_priority(self, node: "TreeNode") -> Tuple[int, float]:
        # Return (priority, last_access_time) so lower priority nodes are evicted first
        return (node.priority, node.last_access_time)


class SLRUStrategy(EvictionStrategy):
    def __init__(self, protected_threshold: int = 2):
        self.protected_threshold = protected_threshold

    def get_priority(self, node: "TreeNode") -> Tuple[int, float]:
        # Priority Logic:
        # Smaller value = Evicted earlier.
        #
        # Segment 0 (Probationary): hit_count < threshold
        # Segment 1 (Protected): hit_count >= threshold
        #
        # Tuple comparison: (segment, last_access_time)
        # Nodes in segment 0 will always be evicted before segment 1.
        # Inside the same segment, older nodes (smaller time) are evicted first.

        is_protected = 1 if node.hit_count >= self.protected_threshold else 0
        return (is_protected, node.last_access_time)


class WLFUStrategy(EvictionStrategy):
    """Weighted LFU-LRU hybrid: frequency gives nodes a virtual recency bonus.

    priority = last_access_time - alpha * log(1 + hit_count)

    Nodes with higher hit_count appear "more recent" and resist eviction.
    Under concurrent workloads with shared prefixes (system prompts, few-shot),
    these nodes accumulate hits and survive bursts of unique requests that
    would flush them under pure LRU.
    """

    def __init__(self, alpha: float = 2.0):
        self.alpha = alpha

    def get_priority(self, node: "TreeNode") -> float:
        return node.last_access_time - self.alpha * math.log1p(node.hit_count)


class WSLRUStrategy(EvictionStrategy):
    """Hybrid SLRU × WLFU.

    Outer segment gate from SLRU (nodes with hit_count >= threshold always
    survive over probationary nodes), smooth WLFU ordering within segment.
    """

    def __init__(self, alpha: float = 2.0, threshold: int = 2):
        self.alpha = alpha
        self.threshold = threshold

    def get_priority(self, node: "TreeNode") -> Tuple[int, float]:
        segment = 1 if node.hit_count >= self.threshold else 0
        return (
            segment,
            node.last_access_time - self.alpha * math.log1p(node.hit_count),
        )


def _node_len(node: "TreeNode") -> int:
    """Safe token-length accessor; root and half-constructed nodes may have key=None."""
    return len(node.key) if node.key is not None else 0


class CALRUStrategy(EvictionStrategy):
    """Cost-Aware LRU.

    The prefix cache's job is to avoid re-prefill work, not to maximise hits.
    The tokens saved if we *keep* a node past one more hit is roughly
    ``hit_count * len(node)`` — long frequently-hit nodes are worth many
    token-saves, short accidentally-shared stubs are worth very few even at
    high hit_count.

    priority = last_access_time + alpha * log1p(hit_count * len(node))

    Higher priority = evicted later (heap pops smallest first). A hit=0 leaf
    collapses exactly to LRU; a long-hot node gets a large positive bump.
    Unlike WLFU the sign is such that *more valuable* nodes look *more
    recent*, which is the behaviour the WLFU docstring claims but inverts.
    """

    def __init__(self, alpha: float = 2.0):
        self.alpha = alpha

    def get_priority(self, node: "TreeNode") -> float:
        return node.last_access_time + self.alpha * math.log1p(
            node.hit_count * _node_len(node)
        )


class SCALRUStrategy(EvictionStrategy):
    """Size-gated Cost-Aware LRU (segmented analogue of WSLRU).

    WSLRU's ``hit_count >= 2`` gate was designed against synthetic random
    tokens, where every shared-prefix node is a full group system prompt.
    On real text (wikitext et al.) accidental short cross-group prefixes
    (" = = ", "\n", " The") also blow past hit_count >= 2, so the gate
    stops discriminating and policies collapse to LRU.

    Replace the gate with an explicit re-prefill-cost threshold:

        protected iff hit_count * len(node) >= cost_threshold

    A 1024-token system prompt earns protection after one re-hit
    (1024 >= 512); a 4-token accidental stub needs hit_count >= 128.
    Within-segment tiebreak uses the sign-corrected CALRU score.
    """

    def __init__(self, cost_threshold: int = 512, alpha: float = 2.0):
        self.cost_threshold = cost_threshold
        self.alpha = alpha

    def get_priority(self, node: "TreeNode") -> Tuple[int, float]:
        cost = node.hit_count * _node_len(node)
        segment = 1 if cost >= self.cost_threshold else 0
        return (
            segment,
            node.last_access_time + self.alpha * math.log1p(cost),
        )
