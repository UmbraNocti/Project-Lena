#!/usr/bin/env bash
# Shared by the start_*.sh scripts (Linux and macOS). Finds llama-server and the models folder.
# Override anything by setting it before running, e.g.  LLAMA_SERVER=/path/to/llama-server ./start_gemma26b.sh
KIT="$(cd "$(dirname "$0")" && pwd)"
MODELS="${MODELS:-$KIT/models}"

pause_exit() { echo; read -r -p "Press Enter to close." _; exit 1; }

find_llama() {
  if [ -n "$LLAMA_SERVER" ]; then echo "$LLAMA_SERVER"; return; fi
  for c in "$KIT/llama.cpp/llama-server" "$KIT"/llama.cpp/*/llama-server "$KIT"/llama.cpp/*/*/llama-server \
           "$KIT"/llama.cpp/*/*/*/llama-server; do
    if [ -x "$c" ]; then echo "$c"; return; fi
  done
  command -v llama-server
}
LLAMA="$(find_llama)"
if [ -z "$LLAMA" ]; then
  echo "Can't find llama-server."
  echo "Unpack a llama.cpp release into the llama.cpp folder next to this script (see README.md),"
  echo "or install it (macOS: brew install llama.cpp) so llama-server is on your PATH."
  pause_exit
fi

# first_existing <file>...  prints the first file that exists in the models folder
first_existing() {
  for f in "$@"; do
    if [ -f "$MODELS/$f" ]; then echo "$MODELS/$f"; return; fi
  done
}

# missing <what> <link>
missing() {
  echo "Missing model file for $1 in: $MODELS"
  echo "Download it from: $2"
  echo "(See README.md for the exact file names.)"
  pause_exit
}

# vision_args <mmproj path or empty>  the --mmproj option, or a note that the model runs without vision
vision_args() {
  if [ -n "$1" ]; then
    VISION=(--mmproj "$1")
  else
    VISION=()
    echo "No vision file (mmproj) found in $MODELS: starting without image support."
    echo "Lena still saves images you attach; set default_vision = false in her settings.txt until you add it."
    echo
  fi
}

run() {
  echo "Starting: $*"
  echo
  if [ -n "$DRYRUN" ]; then exit 0; fi
  "$@"
  status=$?
  echo
  echo "llama-server stopped (exit code $status). If that was unexpected, the messages above say why."
  read -r -p "Press Enter to close." _
}
