#!/usr/bin/env bash
# Parallel 3-model x 3-GPU eviction-policy sweep.
#
# For each of 3 models (different families), we spin up 3 Modal containers in
# parallel - one per GPU (H100 / H200 / B200) - and run the 4 core policies
# (lru, wlfu, wslru, scalru) back-to-back inside each container. A `wait`
# barrier between models keeps logs readable and caps peak Modal concurrency
# at 3 GPUs at a time instead of 9.
#
# mem_fraction per cell was calibrated against empirical Qwen2.5-7B / A100-40GB
# runs in results/2026-04-20T15-20_cost_aware_synth/ (cache=352K tokens ->
# ~500 evictions on a ~393K-token workload). Every cell below targets a cache
# capacity << 393K so that eviction counts are unambiguously non-zero and
# policies actually get exercised.
#
# Usage:
#   bash sglang/eval/sweep.sh
#   TAG_SUFFIX=rerun bash sglang/eval/sweep.sh   # optional suffix on run tags
#
# Outputs land in sglang/eval/results/<timestamp>_<tag>/ as usual.

set -euo pipefail

cd "$(dirname "$0")/../.."

POLICIES="lru,wlfu,wslru,scalru"
DATASET="synthetic"
TAG_SUFFIX="${TAG_SUFFIX:-}"

# model_key | hf_name | mf_h100 | mf_h200 | mf_b200
# Derivation (see README "Gotchas" + sweep.sh header comment):
#   cache_GB = GPU_GB * mf - model_bf16_GB - ~3GB overhead
#   target cache tokens 20K-75K (workload = 131K prefix, ~393K total tree)
#   Phi-4         : 14B weights (~28 GB bf16), 200 KB/tok  (40L, GQA-10, hd=128)
#   Mistral-24B   : 24B weights (~48 GB bf16), 160 KB/tok  (40L, GQA-8,  hd=128)
#   DeepSeek-R1-32B: 32B weights (~64 GB bf16), 256 KB/tok (64L, GQA-8,  hd=128)
# DeepSeek-32B on H100 is intentionally tight (~22K-tok cache) to stress
# every policy; if it OOMs, drop mf to 0.90 and rerun that single cell.
MATRIX=(
  "phi4|microsoft/phi-4|0.56|0.32|0.23"
  "mistral24b|mistralai/Mistral-Small-24B-Instruct-2501|0.78|0.44|0.33"
  "deepseek32b|deepseek-ai/DeepSeek-R1-Distill-Qwen-32B|0.92|0.60|0.43"
)

GPUS=("H100" "H200" "B200")

launch_cell() {
  local model_key="$1" hf_name="$2" gpu="$3" mf="$4"
  local tag="${model_key}_${gpu}${TAG_SUFFIX:+_${TAG_SUFFIX}}"
  echo "[sweep] launching $model_key on $gpu (mf=$mf) -> tag=$tag"
  # EVAL_GPU is read by runner.py at module-import time to bind the
  # @app.function decorator's gpu=... kwarg; --gpu is passed purely so the
  # value shows up in config.json for the result dir.
  #
  # --detach keeps the Modal app alive even if this local bash process / SSH
  # session dies. The remote run continues on Modal's infra; logs + results
  # are still fetchable via `modal app logs <app-id>` or the Modal dashboard.
  # Without --detach, Ctrl-C or a dropped terminal would kill the GPU job.
  EVAL_GPU="$gpu" modal run --detach sglang/eval/runner.py \
    --policies "$POLICIES" \
    --dataset "$DATASET" \
    --model "$hf_name" \
    --gpu "$gpu" \
    --mem-fraction "$mf" \
    --tag "$tag"
}

for row in "${MATRIX[@]}"; do
  IFS='|' read -r MODEL_KEY HF_NAME MF_H100 MF_H200 MF_B200 <<<"$row"
  echo ""
  echo "======================================================================"
  echo "  Phase: $MODEL_KEY  ($HF_NAME)"
  echo "  H100 mf=$MF_H100 | H200 mf=$MF_H200 | B200 mf=$MF_B200"
  echo "======================================================================"

  # Parallel fan-out across 3 GPUs. Each backgrounded job writes to its own
  # results dir; no shared mutable state. `wait` is a hard barrier - we do
  # not start the next model's phase until all 3 GPU cells finish.
  pids=()
  mfs=("$MF_H100" "$MF_H200" "$MF_B200")
  for i in 0 1 2; do
    launch_cell "$MODEL_KEY" "$HF_NAME" "${GPUS[$i]}" "${mfs[$i]}" &
    pids+=($!)
  done

  # Collect exit statuses; continue sweeping even if one cell fails so a
  # single OOM doesn't torch the whole run. Failures are surfaced at the end.
  fail=0
  for pid in "${pids[@]}"; do
    if ! wait "$pid"; then
      echo "[sweep] WARN: pid $pid for $MODEL_KEY failed (see its log above)"
      fail=$((fail + 1))
    fi
  done
  echo "[sweep] $MODEL_KEY phase done ($((3 - fail))/3 cells succeeded)"
done

echo ""
echo "[sweep] all phases complete. Results under sglang/eval/results/"
