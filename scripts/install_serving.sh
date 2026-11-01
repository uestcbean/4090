set -eu
cd /mnt/d/bean/code/cs336/inference_lab
source scripts/project_env.sh
python3 -m venv --without-pip "$HOME/.venvs/4090-serving"
python3 /mnt/d/bean/code/cs336/tmp/4090-pip.pyz --python "$HOME/.venvs/4090-serving/bin/python" install 'vllm==0.11.2' 'transformers==4.57.1' 'numpy<3' --timeout 60 --retries 3
