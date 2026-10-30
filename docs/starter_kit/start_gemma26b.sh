#!/usr/bin/env bash
# Gemma 4 26B-A4B with vision, on a GPU with 16 GB of memory. Lena's main model (default_backend 127.0.0.1:8080).
# Tested on a Tesla P100 16 GB: about 16.0 GB used at a 20,000-token context, ~550 tokens/s reading, ~55 tokens/s writing.
# The GPU is nearly full at these settings, so don't raise -c or -ub.
. "$(dirname "$0")/_common.sh"
MODEL="${MODEL:-$(first_existing \
  gemma-4-26B-A4B-it-qat-q4_0-unquantized-uncensored-heretic-v2.i1-Q4_0.gguf \
  gemma-4-26B_q4_0-it.gguf)}"
MMPROJ="${MMPROJ:-$(first_existing \
  gemma-4-26B-A4B-it-qat-q4_0-unquantized-uncensored-heretic-v2.mmproj-Q8_0.gguf \
  gemma-4-26B-it-mmproj.gguf)}"
[ -n "$MODEL" ]  || missing "Gemma 4 26B" "https://huggingface.co/google/gemma-4-26B-A4B-it-qat-q4_0-gguf"
vision_args "$MMPROJ"
run "$LLAMA" -m "$MODEL" "${VISION[@]}" \
  -c 20000 --parallel 1 --kv-unified -b 2048 -ub 512 -fa on -ngl 99 \
  --jinja --cache-ram 0 --ctx-checkpoints 0 --host 127.0.0.1 --port 8080
