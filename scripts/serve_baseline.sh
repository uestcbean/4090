set -eu
cd /mnt/d/bean/code/cs336/inference_lab
source scripts/project_env.sh
export HF_HUB_OFFLINE=1
export VLLM_NO_USAGE_STATS=1
export DO_NOT_TRACK=1
exec "$HOME/.venvs/4090-serving/bin/vllm" serve \
  /mnt/d/ai/models/Qwen3-8B/b968826d9c46dd6066d109eabc6255188de91218 \
  --served-model-name qwen3-8b --host 127.0.0.1 --port 8000 \
  --dtype bfloat16 --max-model-len 4096 --max-num-seqs 4 \
  --gpu-memory-utilization 0.85 --enable-prefix-caching \
  --enable-chunked-prefill --max-num-batched-tokens 2048 \
  --generation-config vllm
