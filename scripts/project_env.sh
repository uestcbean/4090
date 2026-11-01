# Source from a WSL Bash shell: source scripts/project_env.sh
# Ubuntu VHD is at D:\WSL\Ubuntu on this machine; Linux home is physically on D:.
export INFERENCE_LAB_DATA="$HOME/inference-lab-data"
export HF_HOME="$INFERENCE_LAB_DATA/huggingface"
export HF_HUB_CACHE="$HF_HOME/hub"
export XDG_CACHE_HOME="$INFERENCE_LAB_DATA/cache"
export PIP_CACHE_DIR="$XDG_CACHE_HOME/pip"
export UV_CACHE_DIR="$XDG_CACHE_HOME/uv"
export TORCH_HOME="$XDG_CACHE_HOME/torch"
export TORCHINDUCTOR_CACHE_DIR="$XDG_CACHE_HOME/torchinductor"
export TRITON_CACHE_DIR="$XDG_CACHE_HOME/triton"
export VLLM_CACHE_ROOT="$XDG_CACHE_HOME/vllm"
export TMPDIR="$INFERENCE_LAB_DATA/tmp"
mkdir -p "$HF_HUB_CACHE" "$PIP_CACHE_DIR" "$UV_CACHE_DIR" "$TORCH_HOME" \
    "$TORCHINDUCTOR_CACHE_DIR" "$TRITON_CACHE_DIR" "$VLLM_CACHE_ROOT" "$TMPDIR"
