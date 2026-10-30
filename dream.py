# Project Lena
# Copyright (C) 2026 James (UmbraNocti)
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU Affero General Public License as published
# by the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
# GNU Affero General Public License for more details.
#
# You should have received a copy of the GNU Affero General Public License
# along with this program. If not, see <https://www.gnu.org/licenses/>.
#
# Contact: [ProjectLenaAI@gmail.com]


"""
dream.py - Dream cycle / memory consolidation
Run on demand or via shell script on a schedule.

Consolidation passes, in order:

  Pass 1 - Temporal:
    Watermark = MAX(OriginTime) WHERE RecallCount < 0
    Pulls ALL memories (factual and derived) newer than watermark.
    Groups by canon first (open memories form their own group), then by
    time window, and gives each summary its group's canon. Hard facts and derived are equal here —
    anchor is time, not type. Produces episodic summaries.
    UNCHANGED from the prior design.

  Pass 2 - Embeddings:
    Embeds every memory that has no embedding yet (new derived memories from
    Pass 1, anything a seed could not embed, everything after an embedding
    model change). Fast, so like Pass 1 it is never interrupted.

  Pass 3 - Symbolic (open gist):
    For every keyword touched by an open (no canon) memory since the
    temporal watermark that may have a gist (gistable.py: enough
    mentions, not vetoed, a name or a concept the gate accepts, or
    already gisted), maintain a single standing derived memory ABOUT
    exactly that one keyword (its only tag), in open memory. Not a cluster — an identity. Rewritten in place each cycle from (its own
    prior text + whatever open memory is new for that keyword since its
    own last rewrite). The gist's own OriginTime IS the watermark for its
    next update — no separate tracking needed. Open chat can still see
    both a symbol's open gist and its canon gists side by side.

  Pass 4 - Symbolic (canon gist):
    Same mechanism as Pass 3, but per (symbol, canon): for every canon
    memory since the watermark, each of its keywords gets a standing gist
    inside that canon, built only from that canon's memories. This is what
    lets "Lena" mean something different inside a canon than it does in
    open memory — same symbol, second identity, behind the same canon lock
    recall.py enforces.

  Passes 3 and 4 work in priority order: the handles mentioned most since the last cycle come first, so an
  interrupted cycle leaves only the least-mentioned ones for later (they are picked up the next time they
  come up). Passes 3 and 4 never touch a gist a persona has adopted (Personas.GistRef):
  once someone is a conversation partner, Pass 7 owns their gist.

Cooperative interruption:
  run() accepts an optional stop_event (a threading.Event owned by
  controller.py — the same object it already tracks dream-cycle state
  with). Pass 1 and Pass 2 always run to completion once started; they're fast
  and there's no value in cutting them off mid-way. After Pass 2 finishes, and
  before starting each new gist inside Pass 3/4/5, the loop checks
  stop_event and exits cleanly if it's set — finishing whatever LLM call
  and DB write is already in flight for the current item, never pulling
  the next one. No checkpoint is saved and none is needed: the global
  temporal watermark only ever reflects what Pass 1 actually wrote, and
  any gist that didn't get revisited this cycle simply keeps its own
  older last-touched OriginTime — so it's still a valid, correctly-
  scoped candidate (reaching back far enough to catch what was missed)
  the next time its symbol comes up and a future cycle runs discovery
  again.

Neither symbolic pass clusters by overlap or similarity — that was the
old design (semantic_clusters, 60%/85% thresholds) and it's gone. A
symbol's gist has exactly one identity per (keyword, canon or open)
pair, found by exact keyword-set match, not fuzzy grouping. No pruning
pass is needed either: there's structurally only ever one row per
identity, so there's nothing to consolidate down.

  Pass 5 - Compression:
    Runs last, scoped over every gist in the DB that is over
    GIST_COMPRESS_THRESHOLD_CHARS and has not already been looked at since
    it last changed — not just what Pass 3/4 wrote this cycle. Bloat
    accumulates across many small fold-ins, so a gist can cross the
    threshold on a cycle where it wasn't otherwise touched; this is
    general upkeep. "Looked at" is tracked with Memory.CompressedAt: set
    whenever the LLM actually answered (even if it returned the text
    unchanged), compared against OriginTime, which Pass 3/4 refresh each
    time a gist folds in new sources. So a gist is reviewed once per
    change, not once per cycle. A failed or empty LLM call leaves
    CompressedAt alone, so it is retried next cycle. Deliberately a
    separate LLM call from gist formation: Pass 3/4's job is epistemic
    (fold in new facts, keep the standing understanding correct) and
    Pass 5's job is purely stylistic (say the same true thing in fewer
    characters). Asking one call to do both risks trading accuracy for
    brevity in unpredictable ways. Writes back through
    update_gist_text_only() rather than update_derived_memory() —
    compression changes no keywords and folds in no new source memories,
    so nothing but the Memory text itself should move. Also the safest
    pass to interrupt: since its scoping set is re-derived from a live
    query every cycle rather than carried over from what happened last
    time, there's no state a partial run could leave inconsistent.

  Pass 6 - Image descriptions (runs after Pass 5, only if a vision route exists):
    Describes up to IMAGE_DESCRIBE_BATCH_LIMIT images that have no
    Images.Description yet and links extra keywords drawn from it. Purely
    supplemental; see image_describe.py.

  Pass 7 - Person gists and dossiers:
    For each conversation partner, in each canon they talk to Lena in (and in
    open chat), rewrites their person gist from their new statements there and
    maintains its dossier. See persona.py.

Keyword handles are deterministic everywhere (keyword_utils.handle_keywords):
the temporal pass tags its derived memories with the summary's own handles,
and gist eligibility is decided from part of speech when a keyword is created
(the prep step only fills in keywords that predate that).

A gist is not founded off a single source memory (same MIN_CLUSTER_SIZE
floor Pass 1 already uses — it's the same "is this worth deriving" 
judgment applied to a different clustering mechanism). One memory
compressed into a "gist" is just a reworded duplicate: no synthesis
value, and it costs a recall-scoring competitor for information that's
already sitting in the original. The floor only applies at founding —
an existing gist still folds in a lone new memory rather than going
stale waiting for company.
"""

import sqlite3
import time
import re
from collections import Counter
from pathlib import Path
from datetime import datetime

from llm_client import get_completion, is_backend_alive
from keyword_utils import handle_keywords, handles_of, ensure_keyword, known_names, is_name
import gistable
import memory_index

# ── Paths ─────────────────────────────────────────────────────────────────────
ROOT = Path(__file__).parent.resolve()
DB_PATH = ROOT / "MEM.db"
DREAM_DERIVE_PROMPT_FILE    = ROOT / "prompts" / "dream_derive_prompt.txt"
DREAM_GIST_PROMPT_FILE      = ROOT / "prompts" / "dream_gist_prompt.txt"
DREAM_COMPRESS_PROMPT_FILE  = ROOT / "prompts" / "dream_compress_prompt.txt"

# ── Settings ──────────────────────────────────────────────────────────────────
MIN_CLUSTER_SIZE     = 2      # minimum memories to bother deriving (temporal pass only)
TEMPORAL_WINDOW_HRS  = 2      # hours that define a time session
SUB_CHUNK_SIZE        = 20    # max memories per derived-memory chunk within one temporal cluster
SUB_CHUNK_OVERLAP_PCT = 0.10  # fraction of chunk_size shared with the next chunk, for narrative continuity

IMAGE_DESCRIBE_BATCH_LIMIT = 10  # max images described per dream cycle (Pass 6)
# What may found a gist, and how much source text one gist call takes, are settings (gist_min_mentions, gist_prior,
# gist_fold_chars in settings.txt); gistable.py holds the gate itself.
WRITER_MAX_TOKENS = 4096     # room for a whole gist rewrite in one answer
WRITER_TIMEOUT = 600
GIST_COMPRESS_THRESHOLD_CHARS = 4000   # a gist this long has drifted into a log; a healthy one is about 2,000
COMPRESS_MIN_CHARS = 800               # a rewrite shorter than this is a summary, not a gist: rejected
COMPRESS_KEEP_TOP_NAMES = 2            # ...and so is one that drops one of the names the gist mentions most (besides its subject)

GIST_TEXT_PREFIX = "Gist: "  # marks a Memory row as Pass 3/4 output, not a Pass 1 "Derived: " summary


def stop_requested(stop_event):
    """
    True when the controller has asked this cycle to end at the next
    safe boundary. stop_event is owned by controller.py, not dream.py —
    dream.py only ever reads it. A None stop_event (e.g. run_dream_sync()
    at shutdown, or direct script invocation) means "never stop early."
    """
    return stop_event is not None and stop_event.is_set()


# ══════════════════════════════════════════════════════════════════════════════
# Prompt loading
# ══════════════════════════════════════════════════════════════════════════════

def load_prompt(path):
    if not path.exists():
        print(f"[DREAM][WARNING] Prompt file not found: {path}")
        return ""
    return path.read_text(encoding="utf-8").strip()


# ══════════════════════════════════════════════════════════════════════════════
# LLM calls — all routed through llm_client.get_completion
# ══════════════════════════════════════════════════════════════════════════════

def llm_derive(derive_prompt, memory_block):
    """Used by the temporal pass to synthesize an episodic summary."""
    try:
        memory_index.dream_pause()
        return get_completion(
            system_prompt=derive_prompt, user_text=memory_block,
            max_tokens=256, temperature=0.4,
        )
    except Exception as e:
        print(f"[DREAM][WARNING] Derive LLM call failed: {e}")
        return None


def llm_gist(gist_prompt, memory_block):
    """
    Used by both symbolic passes. Keywords are NOT extracted from the
    result — a gist's identity IS the keyword set it was looked up by,
    forced, not derived. Letting the LLM choose keywords here would risk
    drift away from the exact set gist_with_handles() searches on, which
    would silently orphan the old gist and spawn a duplicate next cycle.

    The large model with thinking off: on the story benchmark the gist
    prompt's portraits were as good or better without thinking, and about
    twice as fast. One empty block would lose the work of every block
    before it, so an empty answer is asked once more.
    """
    for attempt in (1, 2):
        try:
            memory_index.dream_pause()
            text = get_completion(
                system_prompt=gist_prompt, user_text=memory_block,
                max_tokens=WRITER_MAX_TOKENS, temperature=0.4,
                timeout=WRITER_TIMEOUT, preferred_tier="large", thinking=False,
            )
        except Exception as e:
            print(f"[DREAM][WARNING] Gist LLM call failed: {e}")
            return None
        if text and text.strip():
            return text
        if attempt == 1:
            print("[DREAM][WARNING] Gist call came back empty; asking once more.")
    return None


def llm_compress(compress_prompt, gist_text):
    """
    Used by Pass 5 only, against gists that have already crossed the
    bloat threshold. Same non-negotiable as llm_gist(): no keyword
    extraction. Compression doesn't touch identity, it only restates
    the same gist at minimum length — the keyword set that was already
    forced onto this row stays exactly as it is.
    """
    try:
        # Room to hand the whole text back unchanged (about 4 characters a token), so an answer is never cut off
        # mid-gist. Thinking off: restating is not reasoning, and thinking would spend the same token budget.
        memory_index.dream_pause()
        return get_completion(
            system_prompt=compress_prompt, user_text=gist_text,
            max_tokens=len(gist_text) // 3 + 256, temperature=0.2,
            timeout=WRITER_TIMEOUT, preferred_tier="large", thinking=False,
        )
    except Exception as e:
        print(f"[DREAM][WARNING] Compress LLM call failed: {e}")
        return None


# ══════════════════════════════════════════════════════════════════════════════
# Keyword processing (temporal pass only — symbolic passes force their keywords)
# Keyword helpers (handle_keywords, ensure_keyword) live in keyword_utils.py, imported above.
# ══════════════════════════════════════════════════════════════════════════════

# ══════════════════════════════════════════════════════════════════════════════
# Database operations
# ══════════════════════════════════════════════════════════════════════════════

def get_connection():
    conn = sqlite3.connect(DB_PATH)
    conn.execute("PRAGMA journal_mode=WAL")
    return conn


def get_temporal_watermark(conn):
    """
    Returns the OriginTime of the most recent derived memory (temporal
    summary OR symbolic gist — any RecallCount < 0 row). If none exist
    yet, returns 0 (process everything).
    """
    cursor = conn.execute(
        "SELECT MAX(OriginTime) FROM Memory WHERE RecallCount < 0"
        # Remember that <0 equals a derived memory
    )
    row = cursor.fetchone()
    return row[0] if row and row[0] is not None else 0


def should_run_dream(min_interval_seconds, min_new_memories):
    """
    Gate for the automatic idle-triggered dream cycle (called from
    controller.py's idle timer). Fails closed on any DB error. Two
    conditions, both required:

      1. At least min_interval_seconds since the watermark (the most
         recent derived-memory OriginTime) — reuses get_temporal_watermark(),
         same definition of "last dream pass" the rest of dream.py already
         uses. If no derived memory exists yet, the watermark is 0, so this
         condition passes trivially (elapsed time since epoch is huge).

      2. At least min_new_memories FACTUAL memories (RecallCount >= 0)
         written since that same watermark. Derived rows are deliberately
         excluded from this count — otherwise a dream cycle's own gist
         writes would count toward triggering its very next run.
    """
    try:
        conn = get_connection()
        try:
            last_dream_time = get_temporal_watermark(conn)
            elapsed = int(time.time()) - last_dream_time
            if elapsed < min_interval_seconds:
                return False

            cursor = conn.execute(
                "SELECT COUNT(*) FROM Memory WHERE OriginTime > ? AND RecallCount >= 0",
                (last_dream_time,)
            )
            new_factual_count = cursor.fetchone()[0]
            return new_factual_count >= min_new_memories
        finally:
            conn.close()
    except Exception as e:
        print(f"[DREAM][ERROR] should_run_dream() check failed: {e}")
        return False


def fetch_memories_since(conn, since_timestamp):
    """
    Fetch ALL memories (factual and derived) newer than since_timestamp.
    Used for the temporal pass — type doesn't matter, time is the anchor.
    Pass 3 and Pass 4 both derive their scoping sets from this same list
    rather than re-querying, since it's already a full, keyword-attached
    snapshot of everything touched since the last dream cycle.
    """
    cursor = conn.execute("""
        SELECT m.MemRef, m.Memory, m.OriginTime, m.RecallCount, m.CanonID, m.SpeakerID
        FROM Memory m
        WHERE m.OriginTime > ?
        ORDER BY m.OriginTime ASC
    """, (since_timestamp,))
    return _attach_keywords(conn, cursor.fetchall())


def _attach_keywords(conn, rows):
    """Attach keyword sets to a list of memory rows. The text is rendered with its speaker ("I said: ...")."""
    speakers = memory_index.speaker_table(conn)
    memories = []
    for row in rows:
        mem_ref = row[0]
        kw_cursor = conn.execute("""
            SELECT k.Key FROM Keywords k
            JOIN Nodes n ON k.KeyID = n.KeyID
            WHERE n.MemID = ?
        """, (mem_ref,))
        keywords = set(r[0] for r in kw_cursor.fetchall())
        memories.append({
            "mem_ref":      mem_ref,
            "memory":       memory_index.render(row[1], row[5], speakers),
            "origin_time":  row[2],
            "recall_count": row[3],
            "keywords":     keywords,
            "is_derived":   row[3] < 0,
            "canon_id":     row[4],
        })
    return memories


def write_derived_memory(conn, memory_text, keywords, source_recall_sum, cid=None, subject=False):
    """
    Write a brand new derived memory in canon cid (None = open memory). Returns MemRef. subject=True makes it a
    gist, tagged only with `keywords`, the handles it is ABOUT (that is its identity); otherwise it is a summary,
    tagged with the handles of its own text.
    """
    if not keywords:
        return None
    origin_time  = int(time.time())
    recall_count = -(max(1, source_recall_sum))
    try:
        cursor = conn.execute(
            "INSERT INTO Memory (OriginTime, Memory, RecallCount, CanonID) VALUES (?, ?, ?, ?)",
            (origin_time, memory_text, recall_count, cid)
        )
        mem_ref = cursor.lastrowid
        memory_index.link_handles(conn, mem_ref, keywords if subject else memory_index.text_handles(conn, memory_text))
        conn.commit()
        memory_index.embed_memory(conn, mem_ref)
        return mem_ref
    except Exception as e:
        print(f"[DREAM][ERROR] DB write failed: {e}")
        return None


def update_derived_memory(conn, mem_ref, memory_text, keywords, source_recall_sum):
    """
    Update an existing gist in place: text, OriginTime, RecallCount, and its tags (exactly `keywords`, what it is about).
    """
    origin_time  = int(time.time())
    recall_count = -(max(1, source_recall_sum))
    try:
        conn.execute("""
            UPDATE Memory
            SET Memory = ?, OriginTime = ?, RecallCount = ?, Embedding = NULL
            WHERE MemRef = ?
        """, (memory_text, origin_time, recall_count, mem_ref))
        memory_index.link_handles(conn, mem_ref, keywords)
        conn.commit()
        memory_index.embed_memory(conn, mem_ref)
        return mem_ref
    except Exception as e:
        print(f"[DREAM][ERROR] DB update failed: {e}")
        return None


def update_gist_text_only(conn, mem_ref, memory_text):
    """
    Rewrite only the Memory text of an existing row (and stamp
    CompressedAt). Used exclusively by Pass 5 compression, which folds in
    no new source memories and changes no identity — touching OriginTime,
    RecallCount, or Nodes here would be side effects with no
    corresponding cause. update_derived_memory() is for when a gist
    actually absorbs new sources; this is for when it's just being said
    more briefly.
    """
    try:
        conn.execute(
            "UPDATE Memory SET Memory = ?, CompressedAt = ?, Embedding = NULL WHERE MemRef = ?",
            (memory_text, int(time.time()), mem_ref)
        )
        conn.commit()
        memory_index.embed_memory(conn, mem_ref)
        return mem_ref
    except Exception as e:
        print(f"[DREAM][ERROR] Gist text-only update failed: {e}")
        return None


def mark_gist_compressed(conn, mem_ref):
    """Records that Pass 5 reviewed this gist as-is (the LLM answered but
    had nothing shorter to offer), so it isn't re-sent until it changes."""
    conn.execute(
        "UPDATE Memory SET CompressedAt = ? WHERE MemRef = ?",
        (int(time.time()), mem_ref)
    )
    conn.commit()


def find_exact_gist(conn, keyword_set, cid=None):
    """
    Find the derived memory whose keyword set is EXACTLY keyword_set, in
    canon cid (None = open memory) — not overlap, identity. A symbol
    gist's whole identity IS its keyword set plus its canon: {"ATLAS"} in
    open memory and {"ATLAS"} in the Bastion canon are two gists.
    Returns (mem_ref, memory_text, origin_time), or (None, None, None)
    if no exact match exists yet.
    """
    ref = memory_index.gist_with_handles(conn, keyword_set, cid)
    if ref is None:
        return None, None, None
    row = conn.execute("SELECT Memory, OriginTime FROM Memory WHERE MemRef = ?", (ref,)).fetchone()
    return ref, row[0], row[1]


def fetch_tagged_since(conn, keyword_set, since_timestamp, cid=None, exclude_mem_ref=None):
    """
    Fetch memories in canon cid (None = open memory) carrying ALL the handles
    of keyword_set (AND, not OR), newer than since_timestamp. This is the
    "what's new for this symbol since its gist was last rewritten" query.
    The CanonID match is what keeps an open gist free of canon material
    and a canon gist free of anything outside its canon.
    """
    keyword_set = sorted(set(keyword_set))
    n = len(keyword_set)
    placeholders = ",".join("?" for _ in keyword_set)
    cursor = conn.execute(f"""
        SELECT m.MemRef, m.Memory, m.OriginTime, m.RecallCount, m.CanonID, m.SpeakerID
        FROM Memory m
        JOIN Nodes n ON n.MemID = m.MemRef
        JOIN Keywords k ON k.KeyID = n.KeyID
        WHERE k.Key IN ({placeholders}) AND m.OriginTime > ? AND m.CanonID IS ?
        GROUP BY m.MemRef
        HAVING COUNT(DISTINCT k.Key) = ?
        ORDER BY m.OriginTime ASC
    """, tuple(keyword_set) + (since_timestamp, cid, n))
    rows = cursor.fetchall()
    if exclude_mem_ref is not None:
        rows = [r for r in rows if r[0] != exclude_mem_ref]
    return _attach_keywords(conn, rows)


def fetch_tagged_any(conn, keys, since_timestamp, cid=None, exclude_mem_ref=None):
    """Like fetch_tagged_since, but memories carrying ANY of the handles (OR): a main name and its aliases."""
    keys = sorted(set(keys))
    rows = conn.execute(f"""
        SELECT m.MemRef, m.Memory, m.OriginTime, m.RecallCount, m.CanonID, m.SpeakerID
        FROM Memory m JOIN Nodes n ON n.MemID = m.MemRef JOIN Keywords k ON k.KeyID = n.KeyID
        WHERE k.Key IN ({",".join("?" * len(keys))}) AND m.OriginTime > ? AND m.CanonID IS ?
        GROUP BY m.MemRef ORDER BY m.OriginTime ASC""", tuple(keys) + (since_timestamp, cid)).fetchall()
    if exclude_mem_ref is not None:
        rows = [r for r in rows if r[0] != exclude_mem_ref]
    return _attach_keywords(conn, rows)


def fetch_all_gists(conn):
    """
    Gists that Pass 5 still needs to look at: over the compression
    threshold AND never reviewed (CompressedAt NULL) or changed since the
    last review (OriginTime, refreshed by Pass 3/4 on every fold-in, is
    newer than CompressedAt). Deliberately independent of what Pass 3/4
    wrote this cycle: bloat accumulates across many small fold-ins, so a
    gist can cross the threshold on a cycle where it wasn't otherwise
    touched. RecallCount < 0 alone would also catch Pass 1's "Derived: "
    episodic summaries, which aren't gists and were never meant to go
    through the gist-compression prompt, so the GIST_TEXT_PREFIX check is
    what actually scopes this to Pass 3/4 output.
    """
    cursor = conn.execute(
        "SELECT MemRef, Memory FROM Memory "
        "WHERE RecallCount < 0 AND Memory LIKE ? AND LENGTH(Memory) > ? "
        "AND (CompressedAt IS NULL OR OriginTime > CompressedAt) "
        "AND MemRef NOT IN (SELECT GistRef FROM Personas WHERE GistRef IS NOT NULL)",
        (GIST_TEXT_PREFIX + "%", GIST_COMPRESS_THRESHOLD_CHARS)
    )
    return cursor.fetchall()


# ══════════════════════════════════════════════════════════════════════════════
# Which symbols may found a gist: gistable.founding_candidates (mention floor, your ToGist say-so, names by
# capitalization, everything else by the Bayes gate). A symbol that already has its gist is always refreshed.
# ══════════════════════════════════════════════════════════════════════════════

DREAM_GIST_JUDGE_PROMPT_FILE = ROOT / "prompts" / "dream_gist_judge.txt"
JUDGE_SAMPLES = 4


def judge_gistable(conn, key, cid):
    """
    The gist gate's tie-breaker (gistable step 5): a model reads a few sentences where the handle appears (the
    oldest, the newest and some between, in this canon) and says whether it is a subject worth a standing note.
    True / False, or None when the model gave no usable answer (the deterministic decision then stands).
    """
    refs = [r for (r,) in conn.execute(
        "SELECT m.MemRef FROM Memory m JOIN Nodes n ON n.MemID = m.MemRef JOIN Keywords k ON k.KeyID = n.KeyID "
        "WHERE k.Key = ? AND m.CanonID IS ? AND m.RecallCount >= 0 ORDER BY m.OriginTime", (key, cid))]
    if not refs:
        return None
    picked = sorted({refs[round(i * (len(refs) - 1) / max(1, JUDGE_SAMPLES - 1))] for i in range(JUDGE_SAMPLES)})
    pat = re.compile(r"\b" + re.escape(key) + (r"\b" if is_name(key) else r"\w*"), re.I)
    lines, surface = [], Counter()
    for (text,) in conn.execute(f"SELECT Memory FROM Memory WHERE MemRef IN ({','.join('?' * len(picked))})", picked):
        m = pat.search(text or "")
        if m:
            surface[m.group(0)] += 1
            lines.append("- " + " ".join(text[max(0, m.start() - 160):m.end() + 160].split()))
    if not lines:
        return None
    prompt = load_prompt(DREAM_GIST_JUDGE_PROMPT_FILE)
    try:
        memory_index.dream_pause()
        answer = get_completion(system_prompt=prompt, user_text=f"WORD: {surface.most_common(1)[0][0]}\n\nSENTENCES:\n" + "\n".join(lines),
                                max_tokens=8, temperature=0.0, preferred_tier="large", thinking=False)
    except Exception as e:
        print(f"[DREAM][WARNING] Gist judge call failed for {key}: {e}")
        return None
    m = re.search(r"\b(YES|NO)\b", (answer or "").upper())         # tolerant of **NO**, "No." and the like
    return None if not m else m.group(1) == "YES"


def worth_a_gist(conn, symbol, cid, candidates):
    """Whether `symbol` (a handle) may have a gist in this scope: it already has one, or it is a founding candidate there."""
    return (symbol, cid) in candidates or memory_index.gist_with_handles(conn, {symbol}, cid) is not None


def qualifying(conn, symbols, cid, candidates, mentions):
    """The symbols that may have a gist in this scope."""
    return {sym for sym in symbols if worth_a_gist(conn, sym, cid, candidates)}


# ══════════════════════════════════════════════════════════════════════════════
# Aliases: one deterministic line in the gists of names that are the same subject
# ══════════════════════════════════════════════════════════════════════════════

ALIAS_LINE = "Alias: "


def find_aliases(conn, cid, min_mentions, overlap):
    """
    [[name, ...]] groups, in canon cid (None = open memory), of name handles whose memories are nearly the same set
    in BOTH directions (each set at least `overlap` of the other) AND that are written together (within a few
    words, in most of the memories they share): JAMES, MARLOWE and KEETON when the story always writes
    James "Marlowe" Keeton; HASTA and AEOLI for the Hasta Aeoli. A person and the place they are always mentioned
    with overlap too, but are not written side by side, so they are not aliases. Most mentioned first in each group.
    """
    counts = [(k, n) for k, n in conn.execute(
        "SELECT k.Key, COUNT(DISTINCT n.MemID) FROM Nodes n JOIN Keywords k ON k.KeyID = n.KeyID JOIN Memory m ON m.MemRef = n.MemID "
        "WHERE m.RecallCount >= 0 AND m.CanonID IS ? GROUP BY k.KeyID HAVING COUNT(DISTINCT n.MemID) >= ?", (cid, min_mentions))
        if is_name(k)]
    counts.sort(key=lambda kn: (-kn[1], kn[0]))
    sets = {k: {r[0] for r in conn.execute(
        "SELECT DISTINCT n.MemID FROM Nodes n JOIN Keywords k ON k.KeyID = n.KeyID JOIN Memory m ON m.MemRef = n.MemID "
        "WHERE k.Key = ? AND m.CanonID IS ? AND m.RecallCount >= 0", (k, cid))} for k, _ in counts}
    parent = {k: k for k, _ in counts}
    def root(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x
    for i, (a, na) in enumerate(counts):
        for b, nb in counts[i + 1:]:
            if nb < overlap * na:                 # sorted by size: no smaller set can overlap enough any more
                break
            shared = sets[a] & sets[b]
            if len(shared) >= overlap * na and len(shared) >= overlap * nb and written_together(conn, a, b, shared):
                parent[root(b)] = root(a)
    groups = {}
    for k, _ in counts:
        groups.setdefault(root(k), []).append(k)
    return [g for g in groups.values() if len(g) > 1], sets


def written_together(conn, a, b, refs, sample=40, span=30):
    """Whether names a and b stand within `span` characters of each other in at least half of a sample of the memories
    they share (only those few rows are read)."""
    pa = re.compile(r"\b" + re.escape(a) + r"\b", re.I)
    pb = re.compile(r"\b" + re.escape(b) + r"\b", re.I)
    picked = sorted(refs)[-sample:]
    if not picked:
        return False
    close = 0
    for (text,) in conn.execute(f"SELECT Memory FROM Memory WHERE MemRef IN ({','.join('?' * len(picked))})", picked):
        sa = [m.start() for m in pa.finditer(text or "")]
        sb = [m.start() for m in pb.finditer(text or "")]
        close += any(abs(x - y) <= span for x in sa for y in sb)
    return close * 2 >= len(picked)


def full_name(conn, names, refs, limit=300, span=60):
    """
    The most common short phrase that contains every one of `names` (e.g. James "Marlowe" Keeton), read from the
    memories they share (at most `limit` of them, already known by MemRef), or None.
    """
    pats = [re.compile(r"\b" + re.escape(n) + r"\b", re.I) for n in names]
    counts = Counter()
    for (text,) in conn.execute(f"SELECT Memory FROM Memory WHERE MemRef IN ({','.join('?' * min(limit, len(refs)))})",
                                sorted(refs)[-limit:]):
        hits = sorted((m.start(), m.end(), i) for i, p in enumerate(pats) for m in p.finditer(text or ""))
        best = None
        for a in range(len(hits)):                 # shortest window holding every name
            seen = set()
            for b in range(a, len(hits)):
                seen.add(hits[b][2])
                if len(seen) == len(names):
                    if hits[b][1] - hits[a][0] <= span and (best is None or hits[b][1] - hits[a][0] < best[1] - best[0]):
                        best = (hits[a][0], hits[b][1])
                    break
        if best:
            counts[" ".join(text[best[0]:best[1]].split())] += 1
    return counts.most_common(1)[0][0] if counts else None


def refresh_aliases(conn):
    """
    Runs the alias detection in every canon and rewrites the Aliases table to match: each group's most-mentioned
    name is its main name, the others point to it. Runs at the start of a dream, before any gist is written, so an
    alias never gets a gist of its own. Returns the pairs that are new this cycle, [(alias, main, cid)]; a main name
    that already has a gist reads those aliases' memories in once (see fold_aliases_in).
    """
    cfg = memory_index.settings()
    new = []
    for (cid,) in conn.execute("SELECT DISTINCT CanonID FROM Memory").fetchall():
        before = memory_index.alias_groups(conn, cid)[0]
        groups, _ = find_aliases(conn, cid, int(cfg["gist_min_mentions"]), float(cfg["alias_overlap"]))
        conn.execute("DELETE FROM Aliases WHERE CanonID IS ?", (cid,))
        for g in groups:
            main = ensure_keyword(conn, g[0])
            for alias in g[1:]:
                conn.execute("INSERT INTO Aliases (CanonID, KeyID, MainKeyID) VALUES (?, ?, ?)", (cid, ensure_keyword(conn, alias), main))
                if before.get(alias) != g[0]:
                    new.append((alias, g[0], cid))
    conn.commit()
    return new


def fold_aliases_in(conn, pairs, gist_prompt):
    """
    For each newly found alias whose main name already has a gist: fold the memories that carry the alias but not
    the main name into the main gist, once (later memories come in through the normal fold, which reads the whole
    group). The alias's own gist, if it has one, is left as it is. Returns how many main gists were updated.
    """
    done = 0
    for alias, main, cid in pairs:
        ref, text, _ = find_exact_gist(conn, {main}, cid)
        if ref is None or ref in memory_index.adopted_gists(conn):
            continue
        rows = conn.execute("""
            SELECT m.MemRef, m.Memory, m.OriginTime, m.RecallCount, m.CanonID, m.SpeakerID FROM Memory m
            JOIN Nodes n ON n.MemID = m.MemRef JOIN Keywords k ON k.KeyID = n.KeyID
            WHERE k.Key = ? AND m.CanonID IS ? AND m.RecallCount >= 0
              AND m.MemRef NOT IN (SELECT n2.MemID FROM Nodes n2 JOIN Keywords k2 ON k2.KeyID = n2.KeyID WHERE k2.Key = ?)
            ORDER BY m.OriginTime ASC""", (alias, cid, main)).fetchall()
        memories = _attach_keywords(conn, rows)
        if not memories:
            continue
        cname = memory_index.canon_name(conn, cid) if cid is not None else ""
        body = text[len(GIST_TEXT_PREFIX):] if text.startswith(GIST_TEXT_PREFIX) else text
        folded = _fold(body, memories, f"{main} (also called {alias})", cname, gist_prompt, f"{main} <- {alias}")
        if folded:
            update_derived_memory(conn, ref, f"{GIST_TEXT_PREFIX}{folded.strip()}", {main}, sum(abs(m["recall_count"]) for m in memories))
            print(f"[DREAM] {main} — read in {len(memories)} memories that call it {alias}")
            done += 1
    return done


def annotate_aliases(conn):
    """
    For every alias group in the Aliases table: one line at the end of each member's gist, rewritten every cycle so
    a fold can never lose it: 'Alias: Marlowe, James and Keeton are the same: James "Marlowe" Keeton.' A gist whose
    names are no longer aliases loses the line. Deterministic, no LLM. Returns how many gists changed.
    """
    changed = 0
    wanted = {}                                   # gist MemRef -> alias line
    for (cid,) in conn.execute("SELECT DISTINCT CanonID FROM Memory").fetchall():
        for g in memory_index.alias_groups(conn, cid)[1].values():
            sets = [{r[0] for r in conn.execute(
                "SELECT DISTINCT n.MemID FROM Nodes n JOIN Keywords k ON k.KeyID = n.KeyID JOIN Memory m ON m.MemRef = n.MemID "
                "WHERE k.Key = ? AND m.CanonID IS ? AND m.RecallCount >= 0", (n, cid))} for n in g]
            shown = [n.title() for n in g]
            phrase = full_name(conn, g, set.intersection(*sets))
            line = (f"{ALIAS_LINE}{', '.join(shown[:-1])} and {shown[-1]} are the same" + (f": {phrase}." if phrase else "."))
            for n in g:
                ref = memory_index.gist_with_handles(conn, {n}, cid)
                if ref is not None:
                    wanted[ref] = line
    rows = conn.execute("SELECT MemRef, Memory FROM Memory WHERE RecallCount < 0 AND Memory LIKE ?", (f"%{ALIAS_LINE}%",)).fetchall()
    rows += [(r, t) for r, t in conn.execute(
        f"SELECT MemRef, Memory FROM Memory WHERE MemRef IN ({','.join('?' * len(wanted))})", list(wanted)).fetchall()] if wanted else []
    for ref, text in dict(rows).items():
        body = "\n".join(l for l in (text or "").split("\n") if not l.startswith(ALIAS_LINE)).rstrip()
        new = body + ("\n" + wanted[ref] if ref in wanted else "")
        if new != text:
            conn.execute("UPDATE Memory SET Memory = ?, Embedding = NULL WHERE MemRef = ?", (new, ref))
            conn.commit()
            memory_index.embed_memory(conn, ref)
            changed += 1
    return changed


def symbols_from_memories(memories):
    """Every distinct keyword on an open (no canon) memory in the list — Pass 3's candidate set."""
    symbols = set()
    for m in memories:
        if m["canon_id"] is None:
            symbols |= m["keywords"]
    return symbols


def canon_pairs_from_memories(memories):
    """
    Every (symbol, canon) pair touched since the watermark: each keyword
    of each canon memory, paired with that memory's canon. This is the
    scoping set for Pass 4 — only combinations that actually came up,
    not every symbol against every known canon.
    """
    return {(kw, m["canon_id"]) for m in memories if m["canon_id"] is not None for kw in m["keywords"]}


# ══════════════════════════════════════════════════════════════════════════════
# Temporal clustering (Pass 1 — unchanged)
# ══════════════════════════════════════════════════════════════════════════════

def canon_temporal_clusters(memories):
    """
    temporal_clusters() run separately for each canon (and once for untagged memories), so a summary
    never mixes story text with ordinary conversation. Returns [(canon_id_or_None, cluster)]
    in time order.
    """
    groups = {}
    for m in memories:
        groups.setdefault(m["canon_id"], []).append(m)
    out = [(c, cluster) for c, mems in groups.items() for cluster in temporal_clusters(mems)]
    out.sort(key=lambda x: x[1][0]["origin_time"])
    return out


def temporal_clusters(memories, window_hrs=TEMPORAL_WINDOW_HRS):
    """
    Group memories by time proximity. All within window_hrs of the
    earliest in the cluster belong together.
    """
    if not memories:
        return []
    sorted_mems = sorted(memories, key=lambda m: m["origin_time"])
    clusters = []
    current = [sorted_mems[0]]
    window_secs = window_hrs * 3600
    for mem in sorted_mems[1:]:
        if mem["origin_time"] - current[0]["origin_time"] <= window_secs:
            current.append(mem)
        else:
            if len(current) >= MIN_CLUSTER_SIZE:
                clusters.append(current)
            current = [mem]
    if len(current) >= MIN_CLUSTER_SIZE:
        clusters.append(current)
    return clusters


def sub_chunk_cluster(cluster, chunk_size=SUB_CHUNK_SIZE, overlap_pct=SUB_CHUNK_OVERLAP_PCT):
    """
    Break an oversized temporal cluster into fixed-size sub-chunks with
    index-based overlap at the edges, so adjacent derived memories share
    a few verbatim source memories for cohesion.
    Returns a list of sub-chunks (each a list of memory dicts, same shape
    as a normal cluster). If the cluster already fits, returns it unchanged
    as the sole sub-chunk.
    """
    if len(cluster) <= chunk_size:
        return [cluster]

    overlap = max(1, round(chunk_size * overlap_pct))
    step = chunk_size - overlap

    chunks = []
    i = 0
    n = len(cluster)
    while i < n:
        chunks.append(cluster[i:i + chunk_size])
        if i + chunk_size >= n:
            break
        i += step
    return chunks


def derive_temporal_cluster(conn, cluster, derive_prompt, label="", cid=None):
    """
    Synthesize an episodic summary from a temporal cluster and write it
    as a brand new derived memory. Never updates — each time window is
    its own unique event, so there's nothing to match against.

    cid: the canon every memory in the cluster belongs to (clusters are
    formed per canon); the summary belongs to it too, so canon-locked recall
    finds it and a story summary never passes for something said outside
    the story.
    """
    lines = []
    for m in cluster:
        date_str = datetime.fromtimestamp(m["origin_time"]).strftime("%Y-%m-%d")
        lines.append(f"[{date_str}] {m['memory']}")
    memory_block = "\n".join(lines)

    derived_text = llm_derive(derive_prompt, memory_block)
    if not derived_text:
        print(f"[DREAM] {label} — LLM returned nothing, skipping.")
        return None

    keywords = handle_keywords(derived_text, known_names(conn))
    if not keywords:
        print(f"[DREAM] {label} — No keywords extracted, skipping.")
        return None

    source_recall_sum = sum(abs(m["recall_count"]) for m in cluster)
    full_text = f"Derived: {derived_text}"

    mem_ref = write_derived_memory(conn, full_text, keywords, source_recall_sum, cid)
    if mem_ref:
        print(f"[DREAM] {label} — Wrote new derived memory (MemRef={mem_ref})")
        print(f"        Keywords: {keywords}")
        print(f"        Text: {derived_text[:120]}...")
    return mem_ref


# ══════════════════════════════════════════════════════════════════════════════
# Symbolic gist passes (Pass 3 and Pass 4)
# ══════════════════════════════════════════════════════════════════════════════

def _fold(gist_text, memories, handle, canon_label, gist_prompt, label):
    """
    Folds memories into a gist text in blocks that fit one call (a bulk-loaded book can tie thousands of memories
    to one name): each block rewrites the gist from the previous block's result. Returns the new text, or None if
    a call came back empty (the gist is then left as it was).
    """
    blocks, cur, size = [], [], 0
    for m in memories:
        line = f"[{datetime.fromtimestamp(m['origin_time']).strftime('%Y-%m-%d')}] {m['memory']}"
        if cur and size + len(line) > int(memory_index.settings()["gist_fold_chars"]):
            blocks.append(cur); cur, size = [], 0
        cur.append(line); size += len(line) + 1
    if cur:
        blocks.append(cur)
    for n, block in enumerate(blocks, 1):
        lines = [f"HANDLE: {handle}" + (f" (inside the canon {canon_label!r})" if canon_label else "")]
        lines.append(f"GIST: {gist_text if gist_text else '(none yet — this is the founding pass)'}")
        lines.append("MEMORIES:")
        lines.extend(block)
        result = llm_gist(gist_prompt, "\n".join(lines))
        if not result:
            print(f"[DREAM] {label} — LLM returned nothing (block {n}/{len(blocks)}), skipping.")
            return None
        gist_text = result.strip()
        if gist_text.startswith(GIST_TEXT_PREFIX):
            gist_text = gist_text[len(GIST_TEXT_PREFIX):]
    if len(blocks) > 1:
        print(f"[DREAM] {label} — folded in {len(memories)} memories over {len(blocks)} calls")
    return gist_text


def derive_gist(conn, keyword_set, gist_prompt, label="", cid=None, canon_label="", from_scratch=False):
    """
    Rewrite-in-place a standing gist for keyword_set in canon cid (None =
    open memory): {SYMBOL} for a Pass 3 open gist, {SYMBOL} with a canon
    for a Pass 4 canon gist — the mechanism is identical either way.
    A gist adopted by a persona is left alone (Pass 7 owns it).

    The gist's own OriginTime is the watermark: if it exists, only
    memories newer than its last rewrite get folded in. If it doesn't
    exist, since_timestamp is 0 and the founding pass pulls everything
    ever tagged with keyword_set — expected to be expensive exactly
    once per symbol.

    Returns (mem_ref, full_text) on a write/update, or None if there's
    nothing new to fold in (no-op) or a founding attempt didn't clear
    MIN_CLUSTER_SIZE (deferred, not lost — the raw memories stay
    recallable and since_ts stays 0 until enough accumulate).

    from_scratch=True (the Director's dossier button): the old gist's text is
    ignored and every tagged memory is read again, so a bad gist cannot pass
    its errors on; the row is still rewritten in place (same MemRef, so its
    dossier stays attached).
    """
    keyword_set = set(keyword_set)
    group = None
    if len(keyword_set) == 1:
        (key,) = keyword_set
        main_of, members = memory_index.alias_groups(conn, cid)
        if key in main_of:                          # an alias: its memories go to the main name's gist
            return None
        group = members.get(key)
    existing_ref, existing_text, existing_time = find_exact_gist(conn, keyword_set, cid)
    if existing_ref is not None and existing_ref in memory_index.adopted_gists(conn):
        return None
    since_ts = existing_time if existing_ref and not from_scratch else 0

    new_memories = (fetch_tagged_any(conn, group, since_ts, cid, exclude_mem_ref=existing_ref) if group
                    else fetch_tagged_since(conn, keyword_set, since_ts, cid, exclude_mem_ref=existing_ref))
    if not new_memories:
        return None

    # Founding floor: a gist derived from a single source memory is just
    # a reworded duplicate, not a synthesis — no value, and it costs a
    # recall-scoring competitor for information already sitting in the
    # original memory. Only gates founding (existing_ref is None); an
    # already-founded gist still folds in a lone new memory rather than
    # going stale. since_ts stays 0 either way, so the next cycle
    # re-checks this symbol from scratch until it earns founding.
    if not existing_ref and len(new_memories) < MIN_CLUSTER_SIZE:
        return None

    handle = " + ".join(sorted(keyword_set)) + (f" (also called {', '.join(group[1:])})" if group else "")
    gist_text = existing_text[len(GIST_TEXT_PREFIX):] if existing_text and existing_text.startswith(GIST_TEXT_PREFIX) else existing_text
    if from_scratch:
        gist_text = None
    gist_text = _fold(gist_text, new_memories, handle, canon_label, gist_prompt, label)
    if gist_text is None:
        return None

    # Weight reflects this update's fold-in, not accumulated history —
    # otherwise a symbol touched every cycle for months would carry an
    # ever-growing RecallCount unrelated to how active it actually is.
    source_recall_sum = sum(abs(m["recall_count"]) for m in new_memories)
    full_text = f"{GIST_TEXT_PREFIX}{gist_text.strip()}"

    if existing_ref:
        mem_ref = update_derived_memory(conn, existing_ref, full_text, keyword_set, source_recall_sum)
        print(f"[DREAM] {label} — Updated gist (MemRef={existing_ref}), "
              f"{len(new_memories)} new memories folded in")
    else:
        mem_ref = write_derived_memory(conn, full_text, keyword_set, source_recall_sum, cid, subject=True)
        print(f"[DREAM] {label} — Founded new gist (MemRef={mem_ref}), "
              f"{len(new_memories)} founding memories")
    print(f"        Text: {gist_text[:120]}...")
    if not mem_ref:
        return None
    return mem_ref, full_text


# ══════════════════════════════════════════════════════════════════════════════
# Gist compression (Pass 5)
# ══════════════════════════════════════════════════════════════════════════════

def compress_gist(conn, mem_ref, gist_text, compress_prompt, label=""):
    """
    Runs against the gists fetch_all_gists selects — general maintenance,
    not scoped to what Pass 3/4 touched this cycle. No-op if the gist is
    still under GIST_COMPRESS_THRESHOLD_CHARS.

    Once the LLM has answered, the gist is stamped as reviewed even if
    nothing got shorter (a tight gist is told to come back unchanged), so
    it isn't re-sent next cycle; only a Pass 3/4 fold-in makes it eligible
    again. The text is only overwritten when the result is strictly
    shorter. An empty/failed call stamps nothing, so it is retried.
    """
    if len(gist_text) <= GIST_COMPRESS_THRESHOLD_CHARS:
        return None

    # The alias line is the alias pass's, rewritten every cycle: kept out of the LLM's hands and put back as it was.
    lines = gist_text.split("\n")
    alias = [l for l in lines if l.startswith(ALIAS_LINE)]
    body = "\n".join(l for l in lines if not l.startswith(ALIAS_LINE)).rstrip()
    compressed = llm_compress(compress_prompt, body[len(GIST_TEXT_PREFIX):] if body.startswith(GIST_TEXT_PREFIX) else body)
    if not compressed:
        print(f"[DREAM] {label} — Compress LLM returned nothing, will retry next cycle.")
        return None

    new_body = f"{GIST_TEXT_PREFIX}{compressed.strip()}"
    full_text = "\n".join([new_body] + alias)
    if len(full_text) >= len(gist_text):
        mark_gist_compressed(conn, mem_ref)
        print(f"[DREAM] {label} — Nothing shorter found, marked reviewed.")
        return None
    # Deterministic checks that this is still the same gist: long enough to be one, and the names the old text
    # mentions most, besides its own subject, are still in it (Marlowe's gist keeps Greta; an ally it named a few
    # times may go, to the dossier and the memories).
    known = known_names(conn)
    own = {k for (k,) in conn.execute("SELECT k.Key FROM Nodes n JOIN Keywords k ON k.KeyID = n.KeyID WHERE n.MemID = ?", (mem_ref,))}
    cid = conn.execute("SELECT CanonID FROM Memory WHERE MemRef = ?", (mem_ref,)).fetchone()[0]
    main_of, members = memory_index.alias_groups(conn, cid)
    for k in list(own):
        own |= set(members.get(main_of.get(k, k), []))
    counts = Counter(h for h in handles_of(body, known) if is_name(h) and h not in own)
    kept = set(handles_of(new_body, known))
    lost = [n for n, c in counts.most_common(COMPRESS_KEEP_TOP_NAMES) if c >= 2 and n not in kept]
    if len(new_body) < COMPRESS_MIN_CHARS or lost:
        mark_gist_compressed(conn, mem_ref)
        print(f"[DREAM] {label} — Rejected the rewrite ({len(body)} -> {len(new_body)} chars"
              + (f", dropped {', '.join(lost)}" if lost else "") + "); kept the original.")
        return None
    result = update_gist_text_only(conn, mem_ref, full_text)
    if result:
        print(f"[DREAM] {label} — Compressed ({len(gist_text)} -> {len(full_text)} chars)")
    return result


# ══════════════════════════════════════════════════════════════════════════════
# Main dream cycle
# ══════════════════════════════════════════════════════════════════════════════

def run(stop_event=None):
    """
    stop_event: optional threading.Event, owned and set by controller.py.
    See the module docstring's "Cooperative interruption" section for the
    full shape. None (direct script invocation, or run_dream_sync() at
    shutdown) means this cycle always runs to completion.
    """
    print(f"\n[DREAM] Starting dream cycle — {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

    if not DB_PATH.exists():
        print(f"[DREAM][ERROR] MEM.db not found at {DB_PATH}. Aborting.")
        return

    derive_prompt   = load_prompt(DREAM_DERIVE_PROMPT_FILE)
    gist_prompt     = load_prompt(DREAM_GIST_PROMPT_FILE)
    compress_prompt = load_prompt(DREAM_COMPRESS_PROMPT_FILE)
    if not derive_prompt or not gist_prompt or not compress_prompt:
        print("[DREAM][ERROR] One or more prompt files missing. Aborting.")
        return

    # Pre-flight reachability check, routed through llm_client so backend
    # selection (default/small/large, per settings.txt) stays centralized —
    # dream.py never talks to an inference backend directly.
    if not is_backend_alive():
        print("[DREAM][ERROR] Default inference backend not reachable. Aborting.")
        return

    conn = get_connection()

    # ── Pass 1: Temporal ──────────────────────────────────────────────────────
    print("\n[DREAM] Pass 1 — Temporal consolidation")
    watermark = get_temporal_watermark(conn)
    watermark_str = datetime.fromtimestamp(watermark).strftime("%Y-%m-%d %H:%M:%S") \
        if watermark > 0 else "beginning of time"
    print(f"[DREAM] Watermark: {watermark_str}")

    recent = fetch_memories_since(conn, watermark)
    print(f"[DREAM] Memories since watermark: {len(recent)}")

    t_clusters = canon_temporal_clusters(recent)
    print(f"[DREAM] Temporal clusters found: {len(t_clusters)} "
          f"({sum(1 for c, _ in t_clusters if c)} inside a canon)")

    for i, (cluster_canon, cluster) in enumerate(t_clusters):
        earliest = datetime.fromtimestamp(
            cluster[0]["origin_time"]).strftime("%Y-%m-%d")
        sub_chunks = sub_chunk_cluster(cluster)
        if len(sub_chunks) > 1:
            print(f"[DREAM] Cluster {i+1}/{len(t_clusters)} has {len(cluster)} "
                  f"memories, splitting into {len(sub_chunks)} sub-chunks "
                  f"of ≤{SUB_CHUNK_SIZE}")
        for j, chunk in enumerate(sub_chunks):
            if len(sub_chunks) > 1:
                label = (f"Temporal {i+1}/{len(t_clusters)} "
                         f"sub-chunk {j+1}/{len(sub_chunks)} "
                         f"({len(chunk)} memories from {earliest})")
            else:
                label = (f"Temporal {i+1}/{len(t_clusters)} "
                         f"({len(chunk)} memories from {earliest})")
            print(f"\n[DREAM] Processing {label}")
            derive_temporal_cluster(conn, chunk, derive_prompt, label=label, cid=cluster_canon)

    # ── Pass 2: Embeddings ───────────────────────────────────────────────────
    # Everything without an embedding gets one: Pass 1's new derived memories,
    # seeds the embedder missed, everything after an embedding model change.
    # Fast, so like Pass 1 it always runs to completion.
    print("\n[DREAM] Pass 2 — Embeddings")
    embedded = 0
    while True:
        n = memory_index.embed_missing(conn, limit=400)
        embedded += n
        if not n:
            break
    left = conn.execute("SELECT COUNT(*) FROM Memory WHERE Embedding IS NULL").fetchone()[0]
    print(f"[DREAM] Pass 2 complete — {embedded} memories embedded"
          + (f", {left} still without one (embedder off or unreachable)" if left else ""))

    # Pass 1 and Pass 2 always run to completion (see module docstring) — the
    # only checkpoint is here, before committing to the more expensive passes.
    if stop_requested(stop_event):
        conn.close()
        print(f"\n[DREAM] Stop requested after Pass 2 — cycle ending early "
              f"(Pass 3-7 skipped this run) — "
              f"{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        return

    # ── Aliases: names that are the same subject, found before any gist is written ──
    new_aliases = refresh_aliases(conn)
    if new_aliases:
        print(f"[DREAM] Aliases — {len(new_aliases)} new: " + ", ".join(f"{a} -> {m}" for a, m, _ in new_aliases))
        fold_aliases_in(conn, new_aliases, gist_prompt)

    # ── Pass 3: Global symbol gists ──────────────────────────────────────────
    print("\n[DREAM] Pass 3 — Open symbol gists")
    recent_symbols = symbols_from_memories(recent)
    canon_pairs_preview = canon_pairs_from_memories(recent)  # needed early for the combined flag lookup below

    # One decision for Pass 3 AND Pass 4 about which handles may found a gist (see gistable.py).
    cfg = memory_index.settings()
    candidates = gistable.founding_candidates(conn, int(cfg["gist_min_mentions"]), float(cfg["gist_prior"]),
                                              judge=judge_gistable, margin=float(cfg["gist_judge_margin"]))
    pre_filter_count = len(recent_symbols)
    # Most-mentioned first: if the cycle is interrupted, what is left undone is what mattered least.
    mentions = Counter(k for m in recent if m["canon_id"] is None for k in m["keywords"])
    recent_symbols = qualifying(conn, recent_symbols, None, candidates, mentions)
    print(f"[DREAM] Symbols touched since watermark: {pre_filter_count} "
          f"({len(recent_symbols)} may have a gist: {len(candidates)} handles pass the gate overall, plus those already gisted)")

    updated_count = 0
    for i, symbol in enumerate(sorted(recent_symbols, key=lambda k: (-mentions[k], k))):
        if stop_requested(stop_event):
            print(f"[DREAM] Stop requested — Pass 3 halted after "
                  f"{i}/{len(recent_symbols)} symbols.")
            break
        label = f"Symbol gist {i+1}/{len(recent_symbols)} ({symbol})"
        result = derive_gist(conn, {symbol}, gist_prompt, label=label)
        if result:
            updated_count += 1
    print(f"[DREAM] Pass 3 complete — {updated_count} gists written/updated")

    if stop_requested(stop_event):
        conn.close()
        print(f"\n[DREAM] Stop requested — cycle ending early after Pass 3 "
              f"(Pass 4/5 skipped this run) — "
              f"{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        return

    # ── Pass 4: Canon-locked symbol gists ────────────────────────────────────
    print("\n[DREAM] Pass 4 — Canon symbol gists")
    # Reuses canon_pairs_preview computed above (same source list, `recent`,
    # hasn't changed) rather than re-deriving it.
    pair_mentions = Counter((k, m["canon_id"]) for m in recent if m["canon_id"] is not None for k in m["keywords"])
    canon_pairs = set()
    for pair_cid in {c for _, c in canon_pairs_preview}:
        per_canon = Counter({k: n for (k, c), n in pair_mentions.items() if c == pair_cid})
        canon_pairs |= {(sym, pair_cid) for sym in qualifying(conn, {s for s, c in canon_pairs_preview if c == pair_cid},
                                                              pair_cid, candidates, per_canon)}
    print(f"[DREAM] Symbol/canon pairs touched since watermark: {len(canon_pairs_preview)} "
          f"({len(canon_pairs)} may have a gist)")

    updated_count = 0
    for i, (symbol, pair_canon) in enumerate(sorted(canon_pairs, key=lambda p: (-pair_mentions[p], p[0], p[1]))):
        if stop_requested(stop_event):
            print(f"[DREAM] Stop requested — Pass 4 halted after "
                  f"{i}/{len(canon_pairs)} pairs.")
            break
        cname = memory_index.canon_name(conn, pair_canon)
        label = f"Canon gist {i+1}/{len(canon_pairs)} ({symbol} in {cname})"
        result = derive_gist(conn, {symbol}, gist_prompt, label=label, cid=pair_canon, canon_label=cname)
        if result:
            updated_count += 1
    print(f"[DREAM] Pass 4 complete — {updated_count} gists written/updated")

    if stop_requested(stop_event):
        conn.close()
        print(f"\n[DREAM] Stop requested — cycle ending early after Pass 4 "
              f"(Pass 5 skipped this run) — "
              f"{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        return

    # ── Pass 5: Gist compression ─────────────────────────────────────────────
    # Full sweep of every gist in the DB, not just what Pass 3/4 touched
    # this cycle — see fetch_all_gists() and the module docstring. Only
    # over-threshold gists that are new or changed since their last review
    # (Memory.CompressedAt vs OriginTime) are sent to the LLM.
    print("\n[DREAM] Pass 5 — Gist compression")
    all_gists = fetch_all_gists(conn)
    print(f"[DREAM] Gists in scope: {len(all_gists)}")

    compressed_count = 0
    stopped_early = False
    for i, (mem_ref, gist_text) in enumerate(all_gists):
        if stop_requested(stop_event):
            print(f"[DREAM] Stop requested — Pass 5 halted after "
                  f"{i}/{len(all_gists)} gists.")
            stopped_early = True
            break
        label = f"Compress {i+1}/{len(all_gists)} (MemRef={mem_ref})"
        result = compress_gist(conn, mem_ref, gist_text, compress_prompt, label=label)
        if result:
            compressed_count += 1
    print(f"[DREAM] Pass 5 {'halted' if stopped_early else 'complete'} — "
          f"{compressed_count} gists compressed "
          f"(threshold={GIST_COMPRESS_THRESHOLD_CHARS} chars)")

    # ── Aliases: names that are the same subject say so in their gists ───────
    print("\n[DREAM] Aliases")
    print(f"[DREAM] Aliases complete — {annotate_aliases(conn)} gists updated")

    # ── Pass 6: Image descriptions ───────────────────────────────────────────
    # Supplemental only: gives images that lack a Description one (plus extra
    # keywords drawn from it), when a vision route exists. Skipped entirely
    # on a blind setup. Bounded per cycle so a big backlog (or a metered API
    # specialist) is worked through gradually.
    if not stopped_early:
        print("\n[DREAM] Pass 6 — Image descriptions")
        try:
            import image_describe
            described = image_describe.backfill(limit=IMAGE_DESCRIBE_BATCH_LIMIT, stop_event=stop_event)
            print(f"[DREAM] Pass 6 complete — {described} images described")
        except Exception as e:
            print(f"[DREAM][WARNING] Pass 6 skipped: {e}")

    # ── Pass 7: Person gists and dossiers ────────────────────────────────────
    # One gist and one dossier per conversation partner per canon (and open
    # chat), rewritten from their new statements there. See persona.py.
    if not stopped_early and not stop_requested(stop_event):
        print("\n[DREAM] Pass 7 — Person gists and dossiers")
        try:
            import persona
            blocks = persona.run(stop_event)
            print(f"[DREAM] Pass 7 complete — {blocks} person-gist update(s)")
        except Exception as e:
            print(f"[DREAM][WARNING] Pass 7 skipped: {e}")

    conn.close()
    print(f"\n[DREAM] Dream cycle {'stopped early' if stopped_early else 'complete'} — "
          f"{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")


# ══════════════════════════════════════════════════════════════════════════════
# Entry point
# ══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    run()
