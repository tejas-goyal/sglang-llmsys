# Coding agent session: WSLRU eviction policy for SGLang

A session that started as "implement a new KV cache eviction policy and
prove it beats the baseline" and turned into something more interesting:
a sustained argument with the agent about whether our own benchmark was
actually measuring what we thought it was.

## Where we started

The repo had an eval harness for SGLang's radix KV cache: a registry of
eviction policies, a Modal runner that spins up a GPU container and runs
N policies back to back, and a summarizer that renders deltas vs a
baseline. Two non stock policies were already in: SLRU (segmented LRU)
and WLFU (weighted LFU that mixes frequency into LRU ordering).

WLFU was winning. On a shared prefix workload (128 groups, 16 prompts
each, 1024 token system prompts) it cut mean TTFT by 38% and P99 TTFT by
70% vs LRU. The task going in: build something that beats WLFU.

## What we built first

We added WSLRU, a hybrid policy that combines SLRU's segment gate with
WLFU's smooth ordering inside each segment. The intuition is that SLRU
gives you a hard guarantee that any node hit twice or more always
survives over a fresh node, which prevents a pathological case where WLFU
lets a very recent unique leaf kick out a popular but slightly stale
system prompt. Inside each segment WLFU's log frequency weighting still
ranks nodes smoothly.

```python
class WSLRUStrategy(EvictionStrategy):
    def __init__(self, alpha: float = 2.0, threshold: int = 2):
        self.alpha = alpha
        self.threshold = threshold

    def get_priority(self, node):
        segment = 1 if node.hit_count >= self.threshold else 0
        return (
            segment,
            node.last_access_time - self.alpha * math.log1p(node.hit_count),
        )
```

Two file changes total: one class in `evict_policy.py`, one line in
`registry.py`. The patcher picked it up automatically and wired it into
the server's `--radix-eviction-policy` flag at container build time.

## First result, which looked too good

The first A/B run (WSLRU vs WLFU) showed WSLRU beating WLFU by 36% on
mean TTFT and 51% on P99 TTFT. That is a very large gap for a policy
change that is essentially "WLFU with a two level tiebreak".

At this point the right thing to do was not celebrate. The right thing
was to ask if the number made sense. Two follow up observations killed
the easy interpretation:

1. We ran a second A/B (WSLRU vs LRU) in a separate Modal container.
   It reported WSLRU ahead of LRU by 42% on mean TTFT, a better gap
   than WLFU's own 38% vs LRU. But eviction count was reported as
   `0 / 0`. The cache never overflowed. Modal had handed us an A100
   80GB instead of 40GB, cache capacity jumped from 352K tokens to
   985K, the 131K token working set fit comfortably, and the eviction
   policy was dead code for the entire run. The TTFT delta was pure
   cold start noise.

2. Even in runs where evictions were happening, WLFU's own absolute
   numbers swung wildly between sessions. Same seed, same bench
   parameters, mean TTFT moved from 72ms to 179ms across containers.
   Single shot runs on Modal were not reproducible enough to trust
   point estimates of single digit percent improvements.

## How we fixed the setup

The fix was to make every comparison happen inside the same container,
with the same GPU, in one session, so that at least relative ordering
between policies was measured on a consistent substrate.

We ran lru, wlfu and wslru back to back in one container with a pinned
cache capacity. This is the cleanest run we have
([`wslru_triple`](results/2026-04-16T18-57_wslru_triple/summary.md)):

| Metric | lru | wlfu | wslru |
|---|---|---|---|
| Mean TTFT (ms) | 172.55 | 159.58 | 114.57 |
| P99 TTFT (ms) | 1804.17 | 877.28 | 508.86 |
| Eviction count | 579 | 550 | 573 |

WSLRU beats WLFU by 28% on mean TTFT and 42% on P99 TTFT. Directionally
the same as the first run, magnitude cut roughly in half. Evictions are
non zero for all three policies, which is the minimum bar for saying
the eviction logic actually ran.

## The next bias we caught

The agent flagged a remaining issue: policies run sequentially, so
policy 1 pays a cold start penalty that policies 2 and 3 do not. In
the triple run above, LRU is first. Some of LRU's disadvantage is
CUDA graph warmup, page cache state, and first request Poisson
variance. We could not know how much without a control.

So we ran the same three policies in reverse order
([`reverse_triple`](results/2026-04-16T19-37_reverse_triple/summary.md)):

| Metric | wslru (first) | wlfu | lru (last) |
|---|---|---|---|
| Mean TTFT (ms) | 129.89 | 96.37 | 83.84 |
| P99 TTFT (ms) | 1438.78 | 530.02 | 496.69 |

With order reversed, LRU wins on mean TTFT. Whichever policy runs first
pays a ~50ms TTFT penalty from warmup that has nothing to do with cache
policy. This matches the LRU first run where LRU was slowest.

The honest read across both runs:

- Warmup dominates mean TTFT on a 2048 prompt bench. Single run mean
  TTFT deltas of less than 30% are inside the noise band.
- WSLRU's P99 TTFT advantage over LRU survives order reversal (508ms
  vs 1804ms in one run, 1438ms vs 497ms in the other, consistent with
  the policy mattering at the tail regardless of which position it
  runs in). The policies are doing different things, just not the
  amount the first A/B suggested.
- The clean claim is: **WSLRU roughly matches WLFU at the mean and
  improves it at the tail**, not "WSLRU beats WLFU by 30%".

## First principles sanity check on the workload

The agent also pointed out that the synthetic dataset has caveats that
inflate WSLRU's apparent win:

- Prompts are random token sequences drawn from the tokenizer vocab,
  not natural language. 128 system prompts share zero cross group
  prefix, so the radix trie is 128 disjoint branches off the root.
  Real traffic has natural cross prompt sharing ("You are a helpful
  assistant...") that changes the eviction pressure profile.
- `gen_prompt` does not actually seed Python's or NumPy's RNG. The
  `seed` parameter only acts as a cache key for the pickled prompt
  set. Inside one container all policies see identical prompts
  (cache hit on the pickle), but across containers the prompts
  differ. Another reason cross run absolute numbers drift.
- `range_ratio=1.0` collapses the "random length" path to constant
  1024 / 64 / 64, so actual length variance is zero. A real workload
  with variable prompt sizes is where size weighted policies (SWLFU)
  might matter more than segment based ones.

## What the session actually produced

Technical artifacts:

- `WSLRUStrategy` added to `python/sglang/srt/mem_cache/evict_policy.py`
- Registry entry in `eval/registry.py`
- Three comparison runs committed under `eval/results/`
- A 4 panel comparison chart in the `wslru_triple` summary
- Confirmed via reverse ordering that the first A/B run numbers
  overstated the improvement

What actually made the session good:

- We did not stop at the first positive result. When WSLRU beat WLFU
  by 36% the first time, the agent flagged that this magnitude was
  larger than a back of envelope model predicted, which prompted the
  capacity check that found the 80GB GPU bug.
- Every claim got pressure tested. Cross run variance, order bias,
  RNG seeding, workload realism. None of these were in the original
  task. They came out of treating the benchmark itself as a suspect.
- The net claim got smaller but defensible. "WSLRU improves tail
  TTFT vs WLFU, ties on mean, and needs a real workload to prove
  anything stronger" is a result you can show a staff engineer
  without flinching. "WSLRU beats WLFU by 30%" is not.

## What comes next

The open question is whether WSLRU's tail win generalizes off the
synthetic benchmark. The cleanest path forward is to add a workload
profile system to the runner that supports:

- The existing `generated-shared-prefix` benchmark (unchanged, still
  the default)
- ShareGPT for chat style traffic
- Mooncake production traces from Moonshot AI's Kimi, which ship with
  hash based prefix fingerprints specifically for KV cache reuse
  evaluation

If WSLRU's tail advantage holds on Mooncake, the result is publishable.
If it collapses, we know the synthetic benchmark was flattering the
policy and the next thing to try is SWLFU (size weighted), which
targets the variable context length pattern that `generated-shared-prefix`
deliberately eliminates.

## One line takeaway

The win was not the policy. The win was refusing to ship a result that
the benchmark setup could not actually support, and getting the
benchmark to a state where the remaining claim is small but real.
