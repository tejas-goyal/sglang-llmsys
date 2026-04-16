# Eviction eval: smoke_lru

- Run: `2026-04-16T14-25`
- Model: `Qwen/Qwen2.5-7B-Instruct`
- Cache capacity: 352642 tokens
- sglang SHA: `74b4f890780b`
- Bench params: `{"num_prompts": 2048, "gsp_num_groups": 128, "gsp_prompts_per_group": 16, "gsp_system_prompt_len": 1024, "gsp_question_len": 64, "gsp_output_len": 64, "request_rate": 16}`
- Baseline (for deltas): **lru**

| Metric | lru |
|---|---|
| Output throughput (tok/s) | 996.20 |
| Request throughput (req/s) | 15.57 |
| Mean TTFT (ms) | 118.84 |
| P99 TTFT (ms) | 1330.53 |
| Mean E2E latency (ms) | 1698.56 |
| P99 E2E latency (ms) | 9680.06 |
| Cache hit rate (final) | 0.000 |
| Eviction count | 584 |
