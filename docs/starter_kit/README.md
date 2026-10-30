# Lena Starter Kit

Start scripts for the models Lena's author actually uses, so you can skip the sliders and get talking.
You need three things: **llama.cpp** (the program that runs the model), **a model file**, and **Lena**.

> Tested on Linux (Bazzite, Tesla P100 and GTX 1070). The Windows (`.bat`) and macOS scripts are written
> to work, but **untested**. If one fails, the message in its window usually says why.

## 1. Pick a main model

| | Gemma 4 26B-A4B | Gemma 4 E4B |
|---|---|---|
| For | A graphics card with **16 GB** of memory | **Almost anything**, including no graphics card |
| Feels like | The author's daily driver: fast, answers well, sees images, thinks before answering | Small and quick (about 38 words/s on a GTX 1070); slow but usable on a CPU; sees images |
| Download | about 15.3 GB | about 6.3 GB |
| Script | `start_gemma26b` | `start_gemma_e4b` |

Then, optional but recommended: the **embedding model**, which makes Lena's recall sharper. It is tiny
(150 MB) and runs on the CPU, so it doesn't compete with the main model (`start_embeddings_cpu`).

## 2. Get llama.cpp

Download a release from **https://github.com/ggml-org/llama.cpp/releases** (pick the newest, then the
file for your system), and unpack it into the `llama.cpp` folder of this kit.

| System | File to pick |
|---|---|
| Windows, any GPU (NVIDIA, AMD, Intel) | `llama-…-bin-win-vulkan-x64.zip` |
| Windows, NVIDIA (often faster) | `llama-…-bin-win-cuda-…-x64.zip` **and** the matching `cudart-llama-bin-win-cuda-…-x64.zip`, both into the same folder |
| Windows, no GPU | `llama-…-bin-win-cpu-x64.zip` |
| Linux, any GPU | `llama-…-bin-ubuntu-vulkan-x64.tar.gz` |
| Linux, no GPU | `llama-…-bin-ubuntu-x64.tar.gz` |
| macOS | `brew install llama.cpp` (easiest), or `llama-…-bin-macos-arm64.tar.gz` (Apple Silicon) |

Older NVIDIA cards (GTX 10-series, Tesla P100): if the CUDA build refuses your card, use the Vulkan one.

The scripts find `llama-server` anywhere inside the `llama.cpp` folder, or on your PATH.

## 3. Download the model files into `models/`

Put the files straight into the `models` folder (no subfolders). Click a link and the download starts.

**Gemma 4 26B-A4B**, either version (both files of one row):

| Version | Model | Vision part |
|---|---|---|
| The author's (an uncensored community version) | [gemma-4-26B-A4B-it-qat-q4_0-unquantized-uncensored-heretic-v2.i1-Q4_0.gguf](https://huggingface.co/mradermacher/gemma-4-26B-A4B-it-qat-q4_0-unquantized-uncensored-heretic-v2-i1-GGUF/resolve/main/gemma-4-26B-A4B-it-qat-q4_0-unquantized-uncensored-heretic-v2.i1-Q4_0.gguf?download=true) (14.5 GB) | […heretic-v2.mmproj-Q8_0.gguf](https://huggingface.co/mradermacher/gemma-4-26B-A4B-it-qat-q4_0-unquantized-uncensored-heretic-v2-GGUF/resolve/main/gemma-4-26B-A4B-it-qat-q4_0-unquantized-uncensored-heretic-v2.mmproj-Q8_0.gguf?download=true) (0.8 GB) |
| Google's official | [gemma-4-26B_q4_0-it.gguf](https://huggingface.co/google/gemma-4-26B-A4B-it-qat-q4_0-gguf/resolve/main/gemma-4-26B_q4_0-it.gguf?download=true) (14.4 GB) | [gemma-4-26B-it-mmproj.gguf](https://huggingface.co/google/gemma-4-26B-A4B-it-qat-q4_0-gguf/resolve/main/gemma-4-26B-it-mmproj.gguf?download=true) (1.2 GB) |

**Gemma 4 E4B**, either version (both files of one row):

| Version | Model | Vision part |
|---|---|---|
| The author's (an uncensored community version) | [gemma-4-E4B-it-ultra-uncensored-heretic-Q4_K_M.gguf](https://huggingface.co/llmfan46/gemma-4-E4B-it-ultra-uncensored-heretic-GGUF/resolve/main/gemma-4-E4B-it-ultra-uncensored-heretic-Q4_K_M.gguf?download=true) (5.3 GB) | [gemma-4-E4B-it-mmproj-BF16.gguf](https://huggingface.co/llmfan46/gemma-4-E4B-it-ultra-uncensored-heretic-GGUF/resolve/main/gemma-4-E4B-it-mmproj-BF16.gguf?download=true) (1.0 GB) |
| Google's official | [gemma-4-E4B_q4_0-it.gguf](https://huggingface.co/google/gemma-4-E4B-it-qat-q4_0-gguf/resolve/main/gemma-4-E4B_q4_0-it.gguf?download=true) (5.2 GB) | [gemma-4-E4B-it-mmproj.gguf](https://huggingface.co/google/gemma-4-E4B-it-qat-q4_0-gguf/resolve/main/gemma-4-E4B-it-mmproj.gguf?download=true) (1.0 GB) |

The vision part is optional for both models. Without it the model starts without image support; Lena
still saves images you attach and describes them later, once a model that can see is running.

**Embedding model**:
[nomic-embed-text-v1.5.Q8_0.gguf](https://huggingface.co/nomic-ai/nomic-embed-text-v1.5-GGUF/resolve/main/nomic-embed-text-v1.5.Q8_0.gguf?download=true) (0.15 GB)

The model pages: [Gemma 4 26B (Google)](https://huggingface.co/google/gemma-4-26B-A4B-it-qat-q4_0-gguf),
[Gemma 4 26B (uncensored)](https://huggingface.co/mradermacher/gemma-4-26B-A4B-it-qat-q4_0-unquantized-uncensored-heretic-v2-i1-GGUF),
[Gemma 4 E4B (Google)](https://huggingface.co/google/gemma-4-E4B-it-qat-q4_0-gguf),
[Gemma 4 E4B (uncensored)](https://huggingface.co/llmfan46/gemma-4-E4B-it-ultra-uncensored-heretic-GGUF),
[nomic-embed-text v1.5](https://huggingface.co/nomic-ai/nomic-embed-text-v1.5-GGUF). All are Apache-2.0.

## 4. Start them

Each model runs in its own window. Leave the windows open while you use Lena; close one to stop it.

- **Windows:** double-click `start_gemma26b.bat` (or `start_gemma_e4b.bat`), then `start_embeddings_cpu.bat`.
- **Linux / macOS:** open a terminal in this folder and run `./start_gemma26b.sh` (or `./start_gemma_e4b.sh`),
  then, in a second terminal, `./start_embeddings_cpu.sh`.

The model is ready when its window says `listening on http://127.0.0.1:8080` (the embedding model: `:8082`). The 26B takes a
minute or two to load.

## 5. Point Lena at them

In Lena's `settings.txt` (type `!settings` in Lena's chat box to open it):

```ini
default_backend = 127.0.0.1:8080
default_vision = true

embedding_backend = 127.0.0.1:8082
embedding_doc_prefix = search_document:
embedding_query_prefix = search_query:
```

Then start Lena from her folder with `python controller.py` (`python3` on Linux and macOS), or type
`!reload` if she is already running. If you added the embedding model to a Lena that already has memories,
run `python db_setup.py --embed` once in her folder so the old memories get embedded too.

## If something goes wrong

- **Lena says "I can't reach my language model".** The model window is closed, still loading, or crashed.
  Look at that window.
- **"out of memory" in the model window.** The model doesn't fit your graphics card. Use the E4B, or close
  other programs that use the GPU.
- **"address already in use".** That model (or another program) is already running on that port.
- **macOS says the program can't be opened.** Run `xattr -dr com.apple.quarantine llama.cpp` in this folder,
  or install with `brew install llama.cpp` instead.
- **Thinking.** The 26B thinks before it answers by default; the E4B doesn't. To change that, add
  `--reasoning off` (or `--reasoning on`) to the end of the `llama-server` line in the script.
- **Changing things.** Every script lists its settings in one `llama-server` line. `-c` is how much text the
  model can hold at once (Lena needs at least about 12,000), `--port` is where Lena finds it.
