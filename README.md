# Lena

### A local AI that remembers. The LLM is her voice; the memory is Lena.

> **Status: Alpha.** It works, I use it daily, and I can't test every situation. If something breaks, tell me.

Lena is an assistant and co-author that runs on your own hardware and remembers everything you've ever told her: your life, your projects, and every story you've written together. No lorebooks to maintain, no conversations to sort into folders, no "please remind me who that is." You talk; she remembers and connects it.

She does it without a giant model or a giant context window. A deterministic controller decides what Lena recalls, using a plain SQLite database of linked memories. The LLM never decides what to remember; Memory is imposed upon it to reason over. Between sessions she "dreams," folding what happened into short portraits of the people, places and ideas that keep coming up. This is the mechanism that allows Lena to have an evolving and nuanced understanding of a topic.

## Why you'd want it

* **She actually remembers.** Swap models whenever you like. The memory is a database, Context is working memory.
* **Story memory without story cards.** Import a 450,000-word story and she knows it nearly as well as its author, small details included.
* **Small models punch above their weight.** Her recall does the heavy lifting, so a 4B-class model gets answers it couldn't get on its own. In a 95-question benchmark over a 450,000-word RP, Gemma 4 E4B running Lena beat Gemma 4 26B on a traditional vector system, 79.5 to 75.
* **Runs on modest, old hardware.** Built on an i5-3570K with DDR3, a Tesla P100 and a GTX 1070.
* **Any OpenAI-compatible backend.** llama.cpp, KoboldCPP, Ollama, LM Studio, or a hosted API.
* **Extendable.** Bridges connect her to email, RSS, Discord, and files. Teach her your product and she can answer an inbox, follow the news daily, or join your server. A reply gate controls whether she responds to outside messages at all, and a new bridge is just a script.
* **Auditable.** One SQLite file. Recall makes no model calls and gives traceable and repeatable data packets every time.

## The numbers

95 questions about two long, messy roleplay stories (512,000 words, longer than *The Lord of the Rings*), the same local models for every system:

| | Gemma 4 26B | Gemma 4 E4B (small) |
|---|---:|---:|
| **Lena** | **87.5** | **79.5** |
| Lena with no embedding model at all | 85 | 78 |
| A standard RAG setup | 75 | 66.5 |
| Story cards / lorebook | 45.5 | 38.5 |

* With the small model, she beats RAG running on the big one.
* She's usable the minute a story is imported (on par with RAG), and passes RAG within the first hour of dreaming.
* Recall takes about a quarter of a second, with zero LLM calls necessary.

How the test was set up, and what each system got: **[docs/BENCHMARK.md](docs/BENCHMARK.md)**.

## Quick start

**You need:** Python 3.10 or newer, and a model server that speaks the OpenAI `/v1/chat/completions` API.

```bash
git clone https://github.com/UmbraNocti/Project-Lena.git
cd Project-Lena
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

On Windows the activate line is `.venv\Scripts\activate` instead.

That `.venv` folder keeps Lena's dependencies inside the project, which matters for
two reasons: nothing here touches the rest of your machine, and a system Python
upgrade can't take her dependencies away. Activate it in any terminal you start
her from — if `python controller.py` ever dies on `ModuleNotFoundError: flask`,
you almost certainly forgot.

Installing without a virtual environment also works, but most current Linux
distributions will stop you with `error: externally-managed-environment`. Then
it's `pip install --user --break-system-packages -r requirements.txt`, and a
future system Python upgrade will put you back here.

**New to local models?** The **[Starter Kit](docs/starter_kit/README.md)** walks you through what to download and where to put it, with one-click start scripts for a 16 GB graphics card or a smaller gaming PC (Windows, Linux, macOS). Do that first, then come back here.

Point `settings.txt` at your model server:

```ini
default_backend = 127.0.0.1:8080
```

Start your model server, then Lena:

```bash
python controller.py
```

Open `http://localhost:5000` and say hello. She starts with an empty memory and learns from you.

NOTE: you can type "!Settings" as your message to Lena to open the settings file without diving through your directory. "!Restart" will apply all changes made and check the server backend again.

**Recommended:** a small embedding model next to your LLM (nomic-embed-text-v1.5, about 150 MB, runs on a CPU). The vectors, though not necessary, do improve results by providing additional entry points into the associative structure. Two lines in `settings.txt`, covered in **[SETUP.md](docs/SETUP.md)**.

## Learn more

* **[SETUP.md](docs/SETUP.md):** models, embeddings, vision, settings, bridges (email, files, RSS, Reddit, Discord) and commands.
* **[FAQ.md](docs/FAQ.md):** the hard questions. "Isn't this just RAG?" "Why not a bigger model?" "Did AI write this?"
* **[RECALL.md](docs/RECALL.md):** exactly how a memory gets found. **[CODEMAP.md](docs/CODEMAP.md):** where everything lives in the code.

## License

Copyright (C) 2026 James (UmbraNocti). **GNU Affero General Public License v3.0 or later**; see [`LICENSE`](LICENSE). Contact: ProjectLenaAI@gmail.com
