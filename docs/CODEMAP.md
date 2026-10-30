# Code Map

A map of where things happen, not a tutorial. Read it with the code open beside it. Functions are named as
`file.py: function()`. Line numbers are left out on purpose because they go stale.

## The files

| File | Lines | Job |
|---|---|---|
| `controller.py` | ~2,400 | Flask app, chat route, the conversational pipeline, bulk ingestion, bridges, Director Mode, dream scheduling, start-up. Also holds the whole web UI as one HTML string (`HTML`, about 600 lines). |
| `llm_client.py` | ~800 | Every LLM and embedding call goes through here: tier routing (small/default/large), thinking flags, vision routing and image attachment, `get_embeddings()`. |
| `recall.py` | ~600 | `recall()`: identity, vector lane, handle lane, fill, the canon lock. No LLM call. |
| `memory_index.py` | ~560 | What recall needs: settings defaults, `TableStore` (the walk over `Keywords` and `Nodes`), the embedding cache; canons, speakers, personas, the conversation-partner rule, finding a gist by its handles and canon, `text_handles()` / `link_handles()`. |
| `gistable.py` | ~190 | Which handles may found a gist: mention floor, `ToGist`, names (upper-case handles), a naive-Bayes gate for the rest, trained from `data/gist_labels.csv`. |
| `keyword_utils.py` | ~250 | One copy of the handle rules: stop list, tokenizer (hyphen compounds split, accents folded), names vs stems, `handle_keywords()`, part of speech, `ensure_keyword()`. |
| `porter2.py` | ~180 | The Porter2 (Snowball English) stemmer, plain Python. |
| `retag_handles.py` | ~140 | Development tool: converts a test database from the older handle form (literal words) to the current one, from the text. |
| `seed.py` | ~330 | Writes an exchange to memory: chunking, handles, speaker, canon, image memory. |
| `dream.py` | ~1,100 | The dream cycle, passes 1 to 7 (Pass 7 is in `persona.py`). |
| `persona.py` | ~230 | Dream Pass 7: one person gist and one dossier per conversation partner, per canon. |
| `db_setup.py` | ~200 | The only code that creates tables (fresh database); refuses an outdated one. `--embed`. |
| `director_helpers.py` | ~1,200 | Director Mode support: story files, input classification, action resolver, dialogue formatting, pacing grade, dossiers, stub gists. |
| `director_input_classifier.py` | ~170 | Naive Bayes shadow classifier for Director input; logs to `DirectorInputLog` for training. |
| `temporal_anchor.py` | ~170 | Cue check, then (only with a cue) one small LLM call that resolves a date. |
| `reply_gate.py` | ~80 | Bridges only: should Lena answer this comment? Fails to silence. |
| `image_gate.py` | ~110 | After the reply: should a recalled image be shown to the user? Fails closed. |
| `image_describe.py` | ~150 | Writes a text description and extra handles for an image (on upload and in dream Pass 6). |
| `*_bridge.py` | | Email, file, RSS, Reddit, Discord. All share `run(mode, canon, process_comment, stop_event)`. |
| `prompts/` | | Every prompt the system sends, one file each. `system.txt` is Lena's own system prompt. |
| `data/` | | `pos_lexicon.tsv.gz` (part of speech from WordNet 3.0) and its license; `gist_labels.csv` (hand labels for the gist gate; every "no" ships as `ToGist = 0`). |
| `tests/selftest_recall.py` | | Builds a throw-away database and checks storage, recall, the dream, upgrades. Never opens `MEM.db`. |

**Hot reloading.** `controller.py` loads `recall.py`, `seed.py`, `dream.py`, `temporal_anchor.py`, `reply_gate.py`,
`image_gate.py` and the bridges fresh from disk on every call (`importlib`). An edit to one of those files takes
effect on the next message. `memory_index.py`, `keyword_utils.py`, `llm_client.py`, `director_helpers.py`,
`persona.py` and `controller.py` itself are ordinary imports: changing those needs `!restart`.

## Start-up

`controller.py: main()`

1. `ensure_wal_mode()`, then `load_settings()` and `config_report()` (prints what is configured and what is missing).
2. `db_setup.run()`: creates `MEM.db` if missing, refuses one from an older Lena (amnesia mode with the
   reason), checks the embedding model.
3. `check_memory_scripts()`: if the memory scripts fail to load, Lena runs in amnesia mode.
4. `reset_idle_timer()` starts the idle-to-dream timer, then Flask serves on `flask_port` (0.0.0.0).

## A chat message

`controller.py: chat()` is the `/chat` route. Everything typed in the UI comes in here.

1. **Escape commands** (no LLM call): `!quit`, `!dream`, `!settings`, `!reload`, `!restart`, `!<bridge>` /
   `!<bridge> listen`.
2. **Locks** (`lock_ui_during_processing`, on by default): if a bridge is active, chat is refused. If a dream is running, it is
   stopped and waited for (`request_stop_dream_and_wait()`).
3. **Upload**: an attached image is saved to `images/`.
4. **Director Mode** (the Ghost Writer toggle) branches off here. See below.
5. **Long form**: longer than `listening_mode_char_threshold` goes to bulk ingestion. See below.
6. Otherwise `run_conversational_pipeline(message, speaker, canon, image, source="ui")`.

### The pipeline

`controller.py: run_conversational_pipeline()`, used by chat, bulk ingestion and bridges.

| Step | Where | LLM call? |
|---|---|---|
| Temporal anchor | `resolve_temporal_anchor()` -> `temporal_anchor.py: resolve_anchor()` | Only when the message (or, for strong cues, the previous one) mentions a time |
| Recall | `get_memories()` -> `recall.py: recall()` | No (one embedding call when an embedder is set) |
| Reply gate | `gate_reply()` -> `reply_gate.py: should_reply()` | Bridges only (`source="bridge"`) |
| Prompt | system prompt + date + recalled memories + image note; then the previous direct-chat exchange (`get_short_term()`), then the message | |
| Reply | `llm_client.get_completion(..., images=...)` | Yes |
| Logs | `append_log()` -> `LOG.txt` (every exchange); `consolelog.install()` -> `CONSOLE.txt` (the terminal) | one setting, `log_files` |
| Seed | `run_seed()` -> `seed.py: seed()` in a background thread | Only for an uploaded image, if a vision route exists |
| Image gate | `start_image_job()` -> `image_gate.py: should_show_image()` after the reply; the UI polls `/image/result/<token>` | Yes, when an image is a candidate |

The context window holds the recalled memories, the one previous exchange and the new message. There is no
longer conversation history; past turns come back through recall.

### Recall

`recall.py: recall()`. Full description with formulas in [RECALL.md](RECALL.md).

1. Identity: the speaker's person gist and dossier for this canon (`memory_index.find_identity()`).
2. Vector lane: the message embedded, top `recall_vector_hits` by cosine over all memories (`memory_index`
   holds the embedding cache).
3. Handle lane: the message's handles (`keyword_utils.handle_keywords()`) at full weight, plus handles mined
   from the identity gist and the vector hits at lower weights; scored by rarity, handle count and recency
   from the anchor.
4. Fill to the character budget, counted on the text as the model sees it. The canon lock (`Memory.CanonID`) is a hard filter throughout.
5. Each memory is shown with its speaker from the table (`memory_index.render()`: "James said: …", "I said: …", "Rebuilding Legacy: …").
6. Nothing readable: raises `RecallUnavailable`; the controller answers with `amnesia_note()` instead.

### Seeding

`seed.py: seed()`

1. Canon: `memory_index.canon_id(create=True)`, a row in `Canons`. Speaker: `memory_index.get_or_create_speaker()`,
   a row for every named speaker (a person, or a document from the file bridge). Director commits pass no speaker.
2. The message is chunked (`chunk_text()`), each chunk gets every one of its handles as a tag and is stored as bare text
   with its `SpeakerID` and `CanonID`. Lena's reply is stored under her own row (`Kind = 'self'`) with
   `PartnerID` = the person she answered. No "said:" label is ever stored; `memory_index.render()` adds it
   whenever a memory is shown to the model, the dream or the embedder.
3. Each memory is embedded on the spot (`memory_index.embed_memory()`); anything missed is caught by dream
   Pass 2.
4. An uploaded image becomes an image memory and gets a description (`image_describe.describe_and_link()`).

## A long paste (bulk ingestion)

`controller.py: handle_bulk_ingestion()`

1. Seed the whole paste with no reply (`run_seed_sync()`).
2. `extract_bulk_query()`: one small-tier call asks whether the paste contains a question or request.
3. None: acknowledge and stop. Otherwise the extracted question re-enters `run_conversational_pipeline()`
   with a note that the input was unusually large.

## A bridge comment

`controller.py: start_bridge()` runs the bridge's `run()` in a thread and hands it `process_bridge_comment()`.

- **Listen mode**: seed with no reply. Never creates a speaker.
- **Full mode**: `run_conversational_pipeline(..., include_short_term_context=False, source="bridge")`. The
  reply gate runs; a decline is still seeded. The bridge posts whatever text comes back.
- `!quit` -> `stop_bridge()`.

## Director Mode

All in `controller.py: chat()`, with the work in `handle_director_*` and `director_helpers.py`. State lives in
globals (`director_state`, `director_draft`, `director_canon`, `director_pulled_dossiers`).

**New block** (`handle_director_new_block()`)

1. `director_helpers.classify_director_input()`: DIALOGUE, ACTION, SCENE, DIRECTION or PROSE (LLM decision
   tree; the naive Bayes classifier shadows it and logs).
2. PROSE: the user's text is the draft. No call.
3. ACTION: `director_helpers.resolve_action()` decides the outcome from the raw story tail first.
4. `director_recall()`: 20k budget, vector hits spread over the story, dossiers of everyone named, nothing
   already in the prompt, recalled images for reasoning only (`director_vision()`).
5. The type's prompt (`prompts/director_*.txt`) + memories + raw tail paragraphs + pacing note
   (`director_helpers.get_pacing_state()`). DIALOGUE first formats the user's line
   (`director_helpers.format_dialogue()`).
6. Generation on the large tier. The draft is held in RAM.

**Next message while a draft is open**: `director_helpers.classify_director_command()` checks the
`DirectorCommands` table (no LLM call).

- RETRY: drop the draft.
- COMMIT: `director_helpers.commit_story_block()` appends it to `stories/<canon>.txt`;
  `run_seed_then_dossier_maintenance()` seeds it under the canon with no speaker, then
  `director_helpers.run_dossier_maintenance_pass()` updates the dossiers of everyone in the turn and founds
  new ones; `grade_story_pacing_async()` re-grades pacing.
- Anything else is a revision note: `handle_director_revision()` recalls again (note + draft) and rewrites.

## The dream cycle

Triggered by `controller.py: _idle_timer_fired()` (idle for `idle_dream_check_seconds` and
`dream.py: should_run_dream()` agrees), by `!dream`, or on shutdown. Runs in a thread; a new chat message
stops it at the next safe point (`stop_event`).

`dream.py: run()`

| Pass | What | Where |
|---|---|---|
| Gate | Which handles may found a gist (floor, ToGist, names, Bayes) | `gistable.founding_candidates()` |
| Aliases | Names that are the same subject: the Aliases table, one gist, one recall walk | `refresh_aliases()`, `annotate_aliases()` |
| 1 | Temporal consolidation: recent memories clustered in time (per canon) and summarized | `derive_temporal_cluster()` |
| 2 | Embeddings for everything missing one | `memory_index.embed_missing()` |
| 3 | Open symbol gists (open memories only) | `derive_gist()` |
| 4 | Canon symbol gists (one per keyword per canon) | `derive_gist(cid=...)` |
| 5 | Gist compression (person gists excluded) | `compress_gist()` |
| 6 | Image descriptions for older images | `image_describe.backfill()` |
| 7 | Person gists and dossiers per conversation partner per canon | `persona.run()` |

Passes 1 and 2 cannot be interrupted. Person gists and dossiers only go to conversation partners, counted
separately in each canon: `memory_index.is_conversation_partner()`, at least `persona_min_replies` answered
turns and `persona_min_reply_ratio` of their turns answered. A persona adopts the gist Lena already had of
that name there (`Personas.GistRef`), and from then on Passes 3 and 4 leave it alone.

## When something breaks

Lena explains problems in plain words instead of showing a stack trace. When the model can still talk, it
gets a note and explains the problem itself. When the model is the problem, the code writes the message.

| Problem | What the user sees | Where |
|---|---|---|
| Memory database unreadable | Lena answers in amnesia mode and can walk the user through the fix (`prompts/amnesia_help.txt`) | `get_memories()`, `amnesia_note()` |
| Embedding server down | Recall runs on handles alone; Lena gets `DEGRADED_MEMORY_NOTE` and mentions it if asked or if it matters | `recall.LAST_PROBLEMS` -> pipeline |
| Image attached, no vision model | Nothing changes: the image is saved with the conversation's handles and described by a later dream; the model is only told it can't see it this turn (`NO_VISION_NOTE`) | pipeline |
| Vision call failed | The call is retried blind and Lena is told why (`VISION_FAILED_NOTE`) | `llm_client.get_completion(vision_failure_note=...)` |
| Model unreachable, refused key, context too long, timeout | A message from the code: what failed, at which address, what to do (`llm_client.describe_failure()`). The user's message is still saved, and the outage itself is remembered once (`note_llm_outage()`) | pipeline |
| Model answered with nothing | `EMPTY_REPLY_MESSAGE` (usually a thinking model out of tokens) | pipeline |
| Same, in Director Mode | The explanation goes to the status line; the draft on screen is kept, nothing is committed (`BackendDown`) | `chat()` |
| Same, on a bridge | Nothing is posted; the comment is still remembered | pipeline (`source="bridge"`) |
| Same, on a long paste | The paste is saved; the acknowledgement says the question check failed and why | `handle_bulk_ingestion()` |

## Where the data lives

`MEM.db` (SQLite, WAL). Created and upgraded only by `db_setup.py`.

| Table | Holds |
|---|---|
| `Memory` | Every memory as bare text: verbatim chunks, gists (`Gist: ...`), derived summaries. A negative `RecallCount` marks a derived memory. `SpeakerID` (who said or wrote it), `PartnerID` (on Lena's lines: whom she answered), `CanonID` (empty = open memory), `Embedding`. A gist is tagged only with what it is about; everything else with the handles of its own text. |
| `Keywords` | Handles: `Key` (a name in upper case, or a lower-case Porter2 stem; unique), `ToGist` (a human's 1/0 say-so; NULL leaves it to the gate). |
| `Nodes` | Memory <-> keyword links. This is the associative web. |
| `Images` | Image memories (file name in `images/`, description, `CanonID`). |
| `Dossiers` | State per entity (goals, limitations, assets, knowledge, location, relationships, knowledge gaps), keyed by the entity's gist. |
| `Speakers` | One row per speaker, with a `Kind`: `self` (Lena), `person`, or `document` (a file from the file bridge). |
| `Canons` | One row per canon. The canon lock reads `Memory.CanonID`. |
| `Personas` | One person in one canon (empty = open chat): the adopted person gist (`GistRef`) and how far Pass 7 has read (`LastPersonaRef`). |
| `EmbeddingMeta` | Which embedding model built the stored vectors. A change clears them for rebuilding. |
| `GistableModel`, `GistableModelMeta` | The gist gate's naive-Bayes model, in plain SQL. |
| `DirectorCommands` | Commit / retry phrases. |
| `DirectorInput*` | Director input log and the naive Bayes model. |

Outside the database: `LOG.txt` (every exchange), `CONSOLE.txt` (the terminal), `stories/<canon>.txt` (committed Director text),
`images/`, the bridges' `*_settings.txt` and `*_seen.json`.

## Every place the model is called

| Call | Where | Tier |
|---|---|---|
| Reply | `run_conversational_pipeline()` | routed by size |
| Temporal anchor (cue present) | `temporal_anchor.py` | small |
| Reply gate (bridges) | `reply_gate.py` | routed by size |
| Image gate | `image_gate.py` | small |
| Image description | `image_describe.py` | vision route |
| Bulk question extraction | `extract_bulk_query()` | small |
| Director: input type, action resolver, dialogue format, generation, revision, pacing grade, dossier upkeep, new entities, stub gists | `director_helpers.py`, `handle_director_*` | small, except generation and revision (large) |
| Dream: temporal summaries, gists, compression, person gists, dossiers | `dream.py`, `persona.py` | routed by size |

Recall, seeding (handles), gist eligibility, the commit/retry gate and the conversation-partner rule make no
model call.

## Settings and prompts

`settings.txt`: one file, commented. `controller.py: load_settings()` and `memory_index.settings()` (recall,
embedding and persona keys, with defaults) read it. `!reload` re-reads it.

`prompts/`: what each LLM call is told. Changing a prompt needs no restart for the hot-reloaded modules.

## Rough spots

- `controller.py` does too much: routing, pipeline, bridges, Director, dream scheduling and the UI. The HTML
  string and Director Mode are the natural first things to move out.
- Director state is module globals, so only one Director session at a time.
- `director_helpers.py` carries its own copies of a few helpers (`get_connection()`, `load_prompt_file()`,
  `build_dossier_text()` also exists in `recall.py`).
