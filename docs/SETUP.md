# Setting up Lena

Everything past the README's quick start: models, embeddings, vision, the settings that matter, importing stories, bridges and commands. Every setting lives in `settings.txt`, and each one has a comment explaining it. `!settings` in the chat box opens the file; `!reload` re-reads it without a restart.

---

## 1. Models

Lena talks to any server with an OpenAI-compatible `/v1/chat/completions` endpoint: llama.cpp, KoboldCPP, Ollama, LM Studio, vLLM, or a hosted API. One backend in the `default` slot is enough. If you have never run a local model, the [Starter Kit](starter_kit/README.md) gets you a working model server and embedding server with start scripts and a step-by-step walkthrough.

```ini
default_backend = 127.0.0.1:8080   # IP:port or URL
default_api_key =                  # blank for local inference
default_model =                    # blank if the server already has a model loaded
```

There are three slots: `small`, `default` and `large`. Requests are routed by size. Small jobs (gates, classifiers) go to `small` if it's set, big ones (dream writing) go to `large` if it's set, and everything else goes to `default`. The cut-offs are `small_threshold` and `large_threshold`. Leave `small` and `large` blank and everything goes to `default`. This is how you mix a cheap local model for the repetitive calls with a bigger local or API model for the replies.

**What works:**

* **Author's setup:** Gemma 4 26B-A4B (with its vision projector) alone in `default` on a 16 GB P100.
* **Smallest used day to day:** Qwen3.8-4B. Conversation is good; Director Mode is noticeably weaker at that size.
* **Benchmarked:** Gemma 4 E4B scored 79.5 of 95 on the story benchmark, more than a standard RAG setup on the 26B (see [docs/BENCHMARK.md](BENCHMARK.md)).
* Use a general or instruct model. A model that won't follow instructions makes things up no matter what memory you hand it.

## 2. Embedding model (optional, recommended)

Recall works without one. With the dream's portraits in place it loses only a point or two out of 95 on the benchmark. Before those portraits exist, and for questions that share no words with the memory that answers them, a small embedding model helps. The tested one is **nomic-embed-text-v1.5** (Q8_0, about 150 MB). It runs fine on a CPU or a small second GPU:

```bash
llama-server -m nomic-embed-text-v1.5.Q8_0.gguf --embedding --pooling mean -c 2048 -b 2048 -ub 2048 --port 8082
```

```ini
embedding_backend = 127.0.0.1:8082
embedding_doc_prefix = search_document:
embedding_query_prefix = search_query:
```

New memories are embedded as they're stored. To embed an existing database right away, run `python3 db_setup.py --embed`. If you change embedding models, Lena notices and rebuilds the embeddings. `recall_vector_hits` (default 2) is how many closest matches recall takes; more of them only crowds out better results.

## 3. Vision (optional)

If one of your models can see images, flag its slot:

```ini
default_vision = true
```

If none of your slots can see, a fourth route, `specialist_backend` (with `specialist_api_key` and `specialist_model`), can point at a multimodal API used only for images. Without any vision route, Lena still remembers images by what was said around them; she just can't look at them.

Recalled images go to the model as memories. You only see one when a separate image gate decides it would actually help: a picture of the temple you asked about, yes; the dungeon map she used to keep the rooms straight, no. With a vision route, Lena also writes a short description of each image, so an image can be found by what's in it. Related: `vision_max_images`, `vision_max_px` (needs Pillow), `specialist_send_recalled`.

## 4. Dreaming

When you stop talking, Lena dreams. She groups recent memories in time, writes a portrait (a "gist") of each person, place, thing or idea that has come up at least 10 times, notices when two names are the same person, and keeps each conversation partner's dossier current. It runs in the background and never blocks you: send a message and she finishes the one step in progress, then answers.

```ini
idle_dream_check_seconds = 1800    # idle time before a dream is considered; 0 or less turns automatic dreaming off
dream_min_interval_seconds = 7200  # minimum time between dreams
dream_min_new_memories = 20        # minimum new memories since the last dream
dream_pause_seconds = 0            # rest before each of the dream's model calls, for a hot or shared GPU
```

`!dream` starts one by hand. On a paid API, turn automatic dreaming off and run it when you choose.

**How long does it take?** Everyday use: minutes, because a dream only folds in what's new. A big import is the expensive case: the first dream over two stories totalling half a million words took about 15 hours of inference on one P100. She's usable the whole time; the dreaming only makes her better (see the curve in [docs/BENCHMARK.md](BENCHMARK.md)).

**What gets a portrait** is decided by a gate: at least 10 mentions; never a word you've marked as "no" (the `ToGist` column in the `Keywords` table, 0 = never, 1 = always); names pass; other words go through a small classifier. During the dream, a model breaks the uncertain ties ("Biscuit": the dog or the pastry?), once per word (`gist_judge_margin`, 0 turns that off).

## 5. Canon

The **Canon** box in the UI scopes a conversation, a story or a bridge to one named world. Leave it blank and Lena sees everything, every canon included. Set it and that session only sees, and only writes to, that canon. It works one way: open chat can draw on your stories, but a story never sees your private life. A Discord bridge locked to a `Discord` canon keeps your family out of Discord, while Lena still gets to know the people she talks to there.

## 6. Writing stories (Director Mode)

Director Mode is the storyteller. Set the Canon box (it's the story's name), then write as your character (`I stab him`, `"We should go," I said`) or as the author (`let's have her arrive late`). Lena works out which of five kinds of input it is, drafts the next part, and you comment, revise or commit. Only what you commit becomes canon; your directions aren't remembered as part of the story. When you write in character, put the character's name in **Speaking as**. The dossier button rebuilds a character's portrait and dossier from scratch.

**Importing an existing story:** drop the text file into the file bridge's folder (`file_bridge_settings.txt`), set the Canon box to the story's name, and run `!file listen`. Lena reads it, remembers it, and dreams on it when you're idle.

## 7. Bridges

A bridge gives Lena another place to exist. It only supplies input (and optionally a place for replies); the controller does everything else, the same way it does for normal chat.

| Bridge | Command | Needs |
|---|---|---|
| Email | `!email` | standard library; `email_settings.txt` |
| File | `!file` | standard library for `.txt`; `pypdf` / `python-docx` for PDF / Word; `file_bridge_settings.txt` |
| RSS / Atom | `!rss` | `beautifulsoup4`; `rss_settings.txt` |
| Reddit | `!reddit` | `praw`; `reddit_settings.txt` |
| Discord | `!discord` | standard library and a bot account |

`!<name>` starts a bridge in full mode (it may reply); `!<name> listen` reads and remembers without ever replying; `!quit` stops it. Email and File are enabled by default. For the others, install the package, fill in the settings file, and uncomment its `bridge_<name>` line in `settings.txt`. `file_bridge.py` is the simplest reference if you want to write your own.

**Safeties:** a reply gate decides whether a bridge comment deserves an answer, and when it's unsure, Lena stays silent. Listen mode never replies. Email signs its replies and locks out reply loops. Discord is bot-only. Reddit replies are delayed. The image gate never shows images on bridges.

## 8. Commands

| Command | Does |
|---|---|
| `!settings` | opens `settings.txt` in your default editor |
| `!reload` | re-reads the settings and reports problems |
| `!restart` | restarts Lena in place (needed for a new port or bridge script) |
| `!dream` | runs a dream cycle now |
| `!<bridge>` / `!<bridge> listen` / `!quit` | bridges, see above |

## 9. Logs

Two plain-text files sit next to `controller.py`: `LOG.txt`, every exchange in and out, and `CONSOLE.txt`, everything Lena prints to the terminal (startup, the dream, bridges, errors, with a timestamp on each line). That second one means you can run her without a console window and still have a trail if something goes wrong. They're for debugging and reading back; Lena's memory doesn't use them. `CONSOLE.txt` moves to `CONSOLE.old.txt` at startup once it passes 10 MB, so it can't grow without bound.

If you'd rather keep no logs, set `log_files = false` in `settings.txt` and `!reload`. Nothing more is written to either file; existing files are left alone, so delete them yourself if you want them gone.

## 10. The database

`MEM.db` is Lena. Back it up like you'd back up anything you care about. The repository ships it empty. If you start Lena on a database from an older version, she tells you; move it aside and she makes a fresh one. `python3 db_setup.py` creates or checks it by hand. There's no vector database: embeddings live in `MEM.db` next to their memories.

## Requirements

| | |
|---|---|
| **Required** | Python 3.10+, `flask`, `requests`, an OpenAI-compatible LLM backend |
| **Optional** | an embedding server, `numpy` (faster vector search), `praw`, `beautifulsoup4`, `pypdf`, `python-docx`, `Pillow` |
| **Tested on** | Linux |
