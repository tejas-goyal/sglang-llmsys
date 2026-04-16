# Eviction eval: baseline_sweep

- Run: `2026-04-16T14-32`
- Model: `Qwen/Qwen2.5-7B-Instruct`
- Cache capacity: 352642 tokens
- sglang SHA: `74b4f890780b`
- Bench params: `{"num_prompts": 2048, "gsp_num_groups": 128, "gsp_prompts_per_group": 16, "gsp_system_prompt_len": 1024, "gsp_question_len": 64, "gsp_output_len": 64, "request_rate": 16}`
- Baseline (for deltas): **lru**

| Metric | lru | slru | wlfu | Δ vs lru | Δ vs lru |
|---|---|---|---|---|---|
| Output throughput (tok/s) | 996.14 | 996.96 | 997.55 | +0.1% | +0.1% |
| Request throughput (req/s) | 15.56 | 15.58 | 15.59 | +0.1% | +0.1% |
| Mean TTFT (ms) | 116.63 | 74.95 | 71.80 | -35.7% | -38.4% |
| P99 TTFT (ms) | 1384.74 | 431.49 | 415.11 | -68.8% | -70.0% |
| Mean E2E latency (ms) | 1685.84 | 1514.75 | 1533.51 | -10.1% | -9.0% |
| P99 E2E latency (ms) | 9659.61 | 6403.19 | 6447.39 | -33.7% | -33.3% |
| Cache hit rate (final) | 0.000 | 0.000 | 0.000 | — | — |
| Eviction count | 576 | 591 | 576 | +2.6% | +0.0% |
