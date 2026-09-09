#!/bin/bash
# Run using the separate VLM environment, either directly or via the launcher.
set -euo pipefail
exec "${VLLM_PYTHON:-python}" -m vllm.entrypoints.openai.api_server \
    --model "${VLLM_MODEL:-Qwen/Qwen3-VL-30B-A3B-Thinking}" \
    --served-model-name qwen3-vl-30b \
    --host 0.0.0.0 --port "${PORT:-8000}" "$@"
