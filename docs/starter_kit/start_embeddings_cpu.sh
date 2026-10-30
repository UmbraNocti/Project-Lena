#!/usr/bin/env bash
# nomic-embed-text-v1.5 on the CPU: Lena's embedding model (embedding_backend 127.0.0.1:8082).
# Optional but recommended. About 150 MB; runs fine on any CPU and leaves the GPU to the main model.
. "$(dirname "$0")/_common.sh"
MODEL="${MODEL:-$(first_existing nomic-embed-text-v1.5.Q8_0.gguf nomic-embed-text-v1.5.Q4_K_M.gguf)}"
[ -n "$MODEL" ] || missing "nomic-embed-text v1.5" "https://huggingface.co/nomic-ai/nomic-embed-text-v1.5-GGUF"
run "$LLAMA" -m "$MODEL" --embedding --pooling mean -c 2048 -b 2048 -ub 2048 \
  --device none --host 127.0.0.1 --port 8082
