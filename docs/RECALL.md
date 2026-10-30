# Recall

Recall is deterministic: it makes **no LLM call**.

## One message, step by step

Controller pipeline: **temporal anchor -> recall -> reply gate -> reply -> seed**.

1. **Temporal anchor** (`temporal_anchor.py`). A broad cue list (yesterday, last week, months, holidays, dates, ...) is checked against the message; the user's previous message is checked only for unmistakable references ("last spring", "yesterday", a month, a year), so a follow-up like "what did we eat there?" still anchors. No cue: the anchor is now and no call is made. A cue: the small-tier LLM resolves a date (`YYYY-MM-DD` or `SAME`), with the previous message as context. Recall measures recency from this anchor.
2. **Identity.** The speaker's person gist **for this canon** (open chat when no canon is set) and its dossier: their persona's adopted gist (`Personas.GistRef`), or else the gist in this canon whose only handle is their name (what Lena knew about them secondhand). Always included, on top of the budget. In Director Mode the character being played is the speaker on the action, dialogue and scene paths (not on direction or prose).
3. **Vector lane.** The message is embedded and the top K memories (default 6) by cosine over **all** memories (verbatim, gists, derived) are taken. No embedder configured, or a failure, simply leaves the whole budget to the handle lane.
4. **Handle lane.** Associations are read from the database, not re-derived from text: a memory is linked to its handles by `Nodes`. There are seven entry points: the message's own handles at full weight, and the handles of each of the top vector hits (the 10 rarest of each of the top 5, at 0.2, `recall_mined_weight`), each of which leads back across `Nodes` to other memories. The identity gist's 14 rarest handles join at 0.4 (`recall_identity_weight`). A gist has a single tag (what it is about), so when the identity gist or a vector hit is a gist, its handles are taken from its text instead; that reads only the few memories already in hand.
   - `weight(h) = rarity(h) * (1 + 0.6 * (sources - 1))`, `rarity = ln((N + 1) / (df + 1))` where `df` is the number of memories carrying the stem; handles with rarity <= 0.5 are dropped.
   - `score(m) = sum(weights shared) / (number of tags on m)^0.25 * (0.6 + 0.4 * exp(-days_from_anchor / 120))`.
   - All of it is one query over `Nodes` and `Keywords` (`memory_index.TableStore`); nothing is cached between recalls, and recall never reads the text of the table (the Director's "already in the prompt" check looks only at the memories about to be taken).
5. **Fill.** Vector hits first, then the ranked memories, until the budget (10,000 characters) is used. A dossier rides with its gist and is charged against the budget. The identity gist and dossier come on top.
6. **Amnesia.** If the database cannot be read, or both lanes fail, recall raises `RecallUnavailable`. Controller then answers in amnesia mode: the model gets the amnesia notice, the troubleshooting guide in `prompts/amnesia_help.txt` and the actual errors, so it can tell the user why it cannot remember and what to check.

**Canon is a hard gate.** A canon is a row in the `Canons` table and every memory and image has a `CanonID` (empty = open memory). With a canon set, only memories of that canon are eligible, person gists included, so someone's open-chat gist never shows up inside a Discord canon. With no canon set, everything is eligible. Images are recalled through the keyword image lookup, driven by the message's words, behind the same lock. The reply gate reads the first 4,000 characters of the recalled block.

## Director Mode

`controller.director_recall()` serves both a new block and a revision, through the same `recall()`:

- **Budget:** the chat budget (10,000 characters) of recalled memory, plus a rolling context of the newest story text: whole paragraphs up to `director_tail_chars` (10,000 characters). Recall never repeats what the rolling context already shows.
- **Vector hits spread over the story:** the input together with the end of the newest text gets 2/3 of `recall_vector_hits`; then the three committed blocks before it, working up the document, get 1/3 each (at least one each: 1 + 1 + 1 + 1 with the default 2).
- **Revisions are recalled like a new section:** the note with the end of the draft, the note alone (an edit can bring in someone new), then the two latest committed blocks. No temporal anchor: story time is not wall-clock time.
- **Standing slice:** the dossier (with its gist) of every entity named in the input or the newest text, on top of the budget.
- **No repeats:** the newest text's rarest handles shape the handle lane, and anything already in the prompt (the raw tail paragraphs, the draft) is never recalled again.
- **Images:** when a vision route exists, recalled images (character pictures, maps, the ship) are attached to the generation and revision calls for the model to reason over; Director Mode never shows them to the user. The image lookup uses the input plus the end of the newest text.
- **Failure:** a failed recall is not passed into the story prompt; the draft is written without memories and the warning appears in the status line.

Dossiers pulled by a revision join the list the commit-time maintenance pass reviews.

## Handles and keywords

`keyword_utils.py` holds the one copy of everything storage and recall share: the stop list, `handles_of` / `handle_keywords`, the Porter2 stemmer (`porter2.py`), part of speech (`pos_of`, from `data/pos_lexicon.tsv.gz`, built from the WordNet 3.0 index files; see `data/WORDNET_LICENSE.txt`) and `ensure_keyword`. A handle is a symbol, not a word: a name is stored as itself in UPPER CASE (`GRETA`), every other word as its Porter2 stem in lower case (`begin` for begin, begins, beginning, beginnings). A word is a name when it is capitalized mid-sentence; at the start of a sentence when the same text also capitalizes it mid-sentence or it is already a known name; in lower case when it is a known name and not a dictionary word. So `SILVER` (a character) and `silver` (the colour) are different handles.

- Seeding tags each chunk with all of its handles, no LLM call; the same goes for image descriptions and the dream's derived memories. A tag is an association, so a passage about three characters is tagged with all three however common one of them is; telling common handles from rare ones is recall's job (rarity weighting), not the storage's. Hyphenated compounds are tagged whole and by part (`cryo-pod`, `cryo`, `pod`), and accents are folded (`Yucatán` is `YUCATAN`). A database written before this (when a chunk kept only its 12 rarest handles) can be brought up to date, with no LLM call and in seconds, by `python3 retag_handles.py MEM.db`; it keeps a copy first.
- **What may found a gist** is decided by `gistable.py` each dream cycle, per handle and per canon, with no reading of memory text: (1) at least `gist_min_mentions` (10) ordinary memories carry it; (2) `Keywords.ToGist` is a human's say-so: 0 means never, 1 means yes; (3) names (upper-case handles) pass; (4) every other handle goes to a naive Bayes classifier over its spelling and part of speech, kept in the `GistableModel` tables and trained on `data/gist_labels.csv` (hand labels, turned into handles with the current rules when loaded). Its prior is `gist_prior`, the share of such words that deserve a gist. (5) Only in the dream, for a handle with no say-so and no gist whose answer is uncertain (a name that is also an ordinary word, like BISCUIT or SUNDAY, or a Bayes score within `gist_judge_margin` of the line): a model reads a few sentences where it appears and answers whether it is a subject worth a standing note (prompts/dream_gist_judge.txt). The answer is stored in `ToGist` (-1 no, 2 yes; a human's 0/1 is never overwritten), so each handle is asked once. A new database ships with every handle the labels mark "no" set to `ToGist = 0`. A handle that already has a gist is always refreshed. A gist is the derived memory whose only tag is its subject.

## Dream cycle

1. Temporal consolidation (uninterruptible), clustered per canon; each summary belongs to its cluster's canon
2. **Embeddings** (uninterruptible): everything without an embedding
3. Open symbol gists: one per handle that may found one (see above), from open memories only; source text is folded in `gist_fold_chars` (24,000 characters) at a time, on the large model with thinking off. A gist is WHO or WHAT its subject is (nature, character, voice, the relationships that shape or reveal them, at most five defining events); the record stays in the memories and the current state in the dossier (prompts/dream_gist_prompt.txt)
4. Canon symbol gists: one per handle per canon, from that canon's memories only
5. Compression (person gists are excluded): a gist over 4,000 characters has drifted into a log and is rewritten back into gist shape (prompts/dream_compress_prompt.txt). The rewrite is kept only if it is at least 800 characters and still has the two names the old text mentions most, besides its own subject
   - **Aliases** (deterministic, at the start of each dream): names in one canon whose memories are nearly the same set both ways (`alias_overlap`, 0.85) and that are written side by side are recorded in the `Aliases` table (CanonID, KeyID, MainKeyID), the most-mentioned name as the main one. An alias never founds a gist and is no longer folded into one it already has; the main name's gist reads the whole group's memories (and, once, the old memories that use only the alias). Recall walks an alias's main name too, so "Keeton" reaches what is filed under MARLOWE. After compression each member's gist gets one readable line: `Alias: Marlowe, James and Keeton are the same: James "Marlowe" Keeton.` A wrong alias is undone by deleting its row.
6. Image descriptions
7. **Person gists and dossiers** (`persona.py`): for each person, in open chat and in every canon they talk to Lena in, once they are a conversation partner there (answered at least 10 times and for at least 75% of their turns there, so feeds and passing commenters never get one). Their persona first adopts the existing gist whose only handle is their name in that place (what Lena knew secondhand), then rewrites it in place from their new statements there and maintains its dossier. Passes 3 and 4 never touch an adopted gist.

Gists written or rewritten by passes 1, 3, 4 and 5 are re-embedded immediately.

## Database

`db_setup.py` is the only code that creates tables. Controller runs it once at start-up; it is also safe to run by hand (`python3 db_setup.py`, add `--embed` to embed every memory now instead of waiting for the dream cycle).

- No `MEM.db`: creates a fresh one with every table and the default Director commands.
- Existing `MEM.db`: checked; the gist gate's tables are added if missing. A database from before the `Canons` table, or with the older handle form (literal words with a `Stem` column), is refused with a message saying to move it aside; Lena then starts with a fresh one.
- A new database also gets the shipped hand labels: every word marked "no" is added with `ToGist = 0`, and the gist gate is trained from them.
- Every start: the embedding model check (a changed model or prefix clears all embeddings for rebuilding).

| Table | Holds |
|---|---|
| `Memory` | every memory as bare text; `SpeakerID` (who said or wrote it, empty for story text and Lena's own notes), `PartnerID` (on Lena's own lines: whom she answered), `CanonID` (empty = open memory), `Embedding` |
| `Speakers` | one row per speaker, `Kind` = `self` (Lena), `person` or `document` |
| `Canons` | one row per canon |
| `Personas` | one person in one canon (empty `CanonID` = open chat): the adopted gist and how far Pass 7 has read |
| `Keywords`, `Nodes` | handles and the memory/image links to them; canons are not keywords |
| `Images` | image memories, with their `CanonID` |
| `Dossiers` | state per entity, keyed by its gist |

Speakers are recorded everywhere, inside canons too, and only in the tables: memory text never carries a "said:" label. `memory_index.render()` adds the label from `Speakers` whenever a memory is shown (to the model, the dream, the embedder): "James said: …", Lena's own lines as "I said: …", a document as "Rebuilding Legacy: …". So the model sees exactly what it always did. Director Mode commits have no speaker: the story is just the story. The conversation-partner bar counts a person's turns and Lena's lines answering them (`PartnerID`).

`MEM.db` is tracked in this repository and ships empty; running Lena alters it.

## Configure (`settings.txt`)

```
embedding_backend = 127.0.0.1:8082          # OpenAI-compatible /v1/embeddings; empty = handles only
embedding_doc_prefix = search_document:     # nomic-embed-text-v1.5 (a trailing space is added automatically)
embedding_query_prefix = search_query:
recall_vector_hits = 2
recall_mined_weight = 0.2
persona_min_replies = 10
persona_min_reply_ratio = 0.75
persona_min_new_memories = 15
persona_block_size = 25
persona_max_blocks_per_cycle = 6
```

`numpy` speeds up the cosine search when installed (optional).

## Files

See [CODEMAP.md](CODEMAP.md) for the whole system. Recall itself: `recall.py` (the recall), `memory_index.py` (the table walk, embedding cache, canons, speakers), `gistable.py` (what may found a gist), `db_setup.py` (tables and upgrades), `persona.py` (dream Pass 7), `keyword_utils.py`, `temporal_anchor.py`, `seed.py`, `dream.py`, `image_describe.py`, `controller.py` (pipeline, amnesia note, start-up), `llm_client.py` (embedding route), `prompts/persona_gist_prompt.txt`, `prompts/amnesia_help.txt`, `data/`, `tests/selftest_recall.py`.

## Test

`python3 tests/selftest_recall.py` builds a throw-away database with `db_setup.py` in a temp directory (the real `MEM.db` is never opened), uses a fake embedder and a stub LLM, and checks database creation and upgrades, seeding, speakers, both recall lanes and their failures, canon gating, the temporal anchor's cue check, a full dream cycle (pass order, embeddings, person gist and dossier) and embedder changes.

## Not done yet

- Handles from the previous exchange and co-occurrence expansion are untested ideas.
- Shelved idea: **reference canons** for Director Mode. Writing in a fan-fiction canon while reading another canon (the source book) as reference material, without the two ever mixing.
- Questions about the database itself ("when did you first come online?") are not answerable from recall.
