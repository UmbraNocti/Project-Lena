#!/usr/bin/env bash
# Gemma 4 E4B with vision: the small model that runs on almost anything. Lena's main model (default_backend 127.0.0.1:8080).
# Uses the GPU if there is one and fits as much as it can there; the rest runs on the CPU (slower, but it works).
# About 6 GB of memory for the model and its vision part, plus the context.
. "$(dirname "$0")/_common.sh"
MODEL="${MODEL:-$(first_existing gemma-4-E4B-it-ultra-uncensored-heretic-Q4_K_M.gguf \
  gemma-4-E4B_q4_0-it.gguf gemma-4-E4B-it-Q4_0.gguf)}"
MMPROJ="${MMPROJ:-$(first_existing gemma-4-E4B-it-mmproj-BF16.gguf gemma-4-E4B-it-mmproj.gguf mmproj-gemma-4-E4B-it-Q8_0.gguf)}"
[ -n "$MODEL" ]  || missing "Gemma 4 E4B" "https://huggingface.co/llmfan46/gemma-4-E4B-it-ultra-uncensored-heretic-GGUF"
vision_args "$MMPROJ"
run "$LLAMA" -m "$MODEL" "${VISION[@]}" \
  -c 16384 --parallel 1 --jinja --host 127.0.0.1 --port 8080
