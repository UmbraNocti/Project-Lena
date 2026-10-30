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
director_helpers.py - Small, single-purpose Director Mode calls.

Pulled out of controller.py (see dossier-system-design.md §8) purely to
manage controller.py's size/navigability as these small helpers
accumulate — the dossier maintenance pass is expected to land here too
once it's built.

Everything in this module is stateless with ONE exception: pacing state
(see "Pacing state" below), which this module owns and exposes through
get_pacing_state() rather than handing back to controller.py as a global.

What stays in controller.py instead of moving here: reset_director_state()
(owns the director_state/director_draft/director_canon mode-machine
globals — that's session identity, not a narrow call) and
handle_director_new_block()/handle_director_revision() (the per-turn
orchestrators — each is one big dispatcher, not "many small helpers
accumulating," and each is wired into last_user_msg/last_ai_msg and
get_memories(), which are controller-level concerns). Those two call
into this module's functions rather than duplicating them.

Settings note: get_story_tail_paragraphs()/grade_story_pacing() take their
tail-length as an explicit parameter (n / tail_n) rather than reading
controller.py's SETTINGS dict directly — that dict is populated at
runtime by controller.load_settings(), and importing it here would mean
either a circular import or a stale copy. Callers in controller.py pass
int(SETTINGS["director_tail_paragraphs"]) explicitly; the default here
(3) only matters if some future caller doesn't.
"""

import re
import time
import sqlite3
import threading
from pathlib import Path

import llm_client
from keyword_utils import ensure_keyword, stem
import memory_index

# ── Paths ────────────────────────────────────────────────────────────────────
# Own copies, not imports from controller.py — same convention recall.py
# already uses for DB_PATH/ROOT. This file's ROOT resolves to the same
# directory as controller.py's as long as they live side by side.
ROOT = Path(__file__).parent.resolve()
PROMPTS_DIR = ROOT / "prompts"
STORIES_DIR = ROOT / "stories"
DB_PATH = ROOT / "MEM.db"

# Calls that WRITE a gist or a dossier go to the largest model with thinking on (preferred_tier="large");
# yes/no checks stay on "small". The thinking block counts against max_tokens, so writers get room for it.
WRITER_MAX_TOKENS = 4096
WRITER_TIMEOUT = 600
# The dossier button builds the card from the freshly rebuilt gist (the arc) plus this much of the
# character's most recent memories (the present), the same size as one dream gist call.
DOSSIER_RECENT_CHARS = 12000

STORY_SEPARATOR = "\n\n---\n\n"

# Hard backstop only (~4 chars/token) — mirrors controller.py's own
# comment on this constant. Single source of truth now lives here;
# controller.py reads it as director_helpers.DIRECTOR_TAIL_CHAR_CAP
# wherever it still needs it directly (handle_director_new_block).
DIRECTOR_TAIL_CHAR_CAP = 20000

DIRECTOR_PACING_LABELS = ("STALLED", "BALANCED", "FAST")

# unclear/failed classification -> bounded, not full authority
DIRECTOR_FALLBACK_TYPE = "SCENE"

# ── Dossier system ───────────────────────────────────────────────────────────
# The field-patch call is expected to emit these exact uppercase headers —
# see dossier_field_patch.txt. Maps the LLM-facing vocabulary
# (KNOWLEDGE_GAPS) to the actual Dossiers column (KnowledgeGaps) — the two
# don't share a mechanical transform for every field, hence the explicit map.
_FIELD_HEADER_TO_COLUMN = {
    "GOALS":          "Goals",
    "LIMITATIONS":    "Limitations",
    "ASSETS":         "Assets",
    "KNOWLEDGE":      "Knowledge",
    "LOCATION":       "Location",
    "RELATIONSHIPS":  "Relationships",
    "KNOWLEDGE_GAPS": "KnowledgeGaps",
}

# Fixed, arbitrary backdate for stub gists (design doc §7) — deliberately
# not "look up the last DreamCycle run": DreamCycle will see anything this
# old as stale and refine/replace it on its own schedule regardless, and
# if the entity never comes up again the stub just sits harmlessly unused.
STUB_GIST_BACKDATE = 0.0  # Unix epoch

# On-demand button (design doc §6) — a name shouldn't need more than this;
# anything longer is "anomalous input," not a name, per §6 step 1.
ENTITY_NAME_MAX_CHARS = 60

# ── Pacing state ─────────────────────────────────────────────────────────────
# The one piece of module-level state in this file. Previously a global
# in controller.py (director_pacing_state); moved here since
# grade_story_pacing() is the only thing that ever writes it. Read via
# get_pacing_state() rather than importing this variable directly, so
# the module keeps control of how it's mutated.
_pacing_state = "BALANCED"


def get_pacing_state():
    return _pacing_state


def get_connection():
    """Mirrors recall.py's get_connection() — own connection per call,
    same WAL-mode convention, no shared connection state across this
    module's functions."""
    conn = sqlite3.connect(DB_PATH)
    conn.execute("PRAGMA journal_mode=WAL")
    return conn


# ══════════════════════════════════════════════════════════════════════════════
# Prompt loading
# ══════════════════════════════════════════════════════════════════════════════

def load_prompt_file(filename):
    path = PROMPTS_DIR / filename
    if not path.exists():
        print(f"[DIRECTOR_HELPERS][WARNING] Missing prompt file: {path}")
        return ""
    return path.read_text(encoding="utf-8").strip()


# ══════════════════════════════════════════════════════════════════════════════
# Canon / story file access
# ══════════════════════════════════════════════════════════════════════════════

def normalize_canon(canon_raw):
    """Light normalization for the raw canon string."""
    if not canon_raw:
        return ""
    cleaned = canon_raw.strip()
    cleaned = re.sub(r'[^A-Za-z0-9 _-]', '', cleaned)
    return cleaned.strip()


def story_path_for_canon(canon):
    """Resolves which file in stories/ this canon reads from and writes to.
    Every story belongs to a canon; a blank canon has no story file and
    returns None."""
    canon = normalize_canon(canon)
    if not canon:
        return None
    STORIES_DIR.mkdir(exist_ok=True)
    return STORIES_DIR / f"{canon}.txt"


def read_story_full(canon=""):
    path = story_path_for_canon(canon)
    if path is None or not path.exists():
        return ""
    return path.read_text(encoding="utf-8")


def get_story_tail_block(canon=""):
    """Returns the most recently committed story block."""
    full = read_story_full(canon)
    if not full.strip():
        return ""
    blocks = full.split(STORY_SEPARATOR)
    return blocks[-1].strip()


def get_story_tail_paragraphs(canon="", n=3, char_cap=DIRECTOR_TAIL_CHAR_CAP):
    """
    Returns the last n paragraphs of this canon's story file.

    n: paragraph count. Callers in controller.py pass
       int(SETTINGS["director_tail_paragraphs"]) explicitly — see module
       docstring. The default of 3 here only applies if some future
       caller omits it.
    """
    full = read_story_full(canon)
    if not full.strip():
        return ""
    paragraphs = [p.strip() for p in full.split("\n\n") if p.strip() and p.strip() != "---"]
    tail = paragraphs[-n:]
    text = "\n\n".join(tail)
    if len(text) > char_cap:
        text = text[-char_cap:]
    return text


def get_story_tail_chars(canon="", chars=10000):
    """
    Director Mode's rolling context: the newest whole paragraphs of this canon's story, about `chars` characters
    in total (rounded to the paragraph: a paragraph that would cross the limit is left out, but the newest one is
    always included). Recall excludes anything that appears here, so nothing is shown twice.
    """
    full = read_story_full(canon)
    if not full.strip():
        return ""
    paragraphs = [p.strip() for p in full.split("\n\n") if p.strip() and p.strip() != "---"]
    tail, used = [], 0
    for p in reversed(paragraphs):
        if tail and used + len(p) > chars:
            break
        tail.insert(0, p)
        used += len(p) + 2
    return "\n\n".join(tail)


def commit_story_block(canon, text):
    """Appends an approved story block to this canon's story file."""
    path = story_path_for_canon(canon)
    if path is None:
        raise ValueError("commit_story_block requires a canon")
    existing = read_story_full(canon)
    with open(path, "a", encoding="utf-8") as f:
        f.write((STORY_SEPARATOR + text) if existing.strip() else text)


# ══════════════════════════════════════════════════════════════════════════════
# Pacing gate
# ══════════════════════════════════════════════════════════════════════════════

def grade_story_pacing(canon, tail_n=3):
    """
    Reads the same tail window used for generation context (no memory/recall
    pass, no branch info) and grades the current state of the story as
    STALLED / BALANCED / FAST. Purely advisory — never gates or blocks
    anything, just informs the next director generation's system prompt.

    Synchronous by design; callers wrap this in a background thread (see
    grade_story_pacing_async) so it never blocks the request that
    triggered it.

    tail_n: see get_story_tail_paragraphs()'s n parameter — same
    settings-injection note applies.
    """
    global _pacing_state
    tail = get_story_tail_paragraphs(canon, n=tail_n)
    if not tail:
        _pacing_state = "BALANCED"
        print("[DIRECTOR] Pacing gate -> BALANCED (no story yet)")
        return

    prompt = load_prompt_file("director_pacing_prompt.txt")
    if not prompt:
        print("[DIRECTOR] Pacing gate -> BALANCED (no prompt file loaded, fail-quiet)")
        _pacing_state = "BALANCED"
        return

    try:
        raw = llm_client.get_completion(
            system_prompt=prompt,
            user_text=tail,
            max_tokens=8,
            temperature=0.0,
            timeout=30,
            preferred_tier="small",
        ).strip().upper()
        for label in DIRECTOR_PACING_LABELS:
            if label in raw:
                _pacing_state = label
                print(f"[DIRECTOR] Pacing gate -> {label} (raw: {raw!r})")
                return
        print(f"[DIRECTOR] Pacing gate returned unparseable output ({raw!r}), fail-quiet to BALANCED.")
        _pacing_state = "BALANCED"
    except Exception as e:
        print(f"[DIRECTOR] Pacing gate failed, fail-quiet to BALANCED: {e}")
        _pacing_state = "BALANCED"


def grade_story_pacing_async(canon, tail_n=3):
    """Fire-and-forget wrapper — same background-thread pattern as controller's run_dream()."""
    threading.Thread(target=grade_story_pacing, args=(canon, tail_n), daemon=True).start()


# ══════════════════════════════════════════════════════════════════════════════
# Classification / satisfaction / resolution
# ══════════════════════════════════════════════════════════════════════════════

def normalize_command(text):
    """Lowercase, drop apostrophes, turn other punctuation into spaces,
    collapse whitespace. DirectorCommands.Phrase must be stored in this form."""
    text = (text or "").lower().replace("'", "").replace("\u2019", "")
    text = re.sub(r"[^a-z0-9]+", " ", text)
    return text.strip()


def classify_director_command(user_message):
    """
    Deterministic edit-loop gate: returns "COMMIT", "RETRY", or None (treat
    as a revision instruction). Whole-message match against the
    DirectorCommands table after normalization — "good" commits, "good but
    shorter" falls through to revision. Empty input commits. Queried per
    call (tiny table) so DBeaver edits take effect without a restart.
    Fails quiet to None on any DB error.
    """
    phrase = normalize_command(user_message)
    if not phrase:
        print("[DIRECTOR] Command gate -> COMMIT (empty input shortcut)")
        return "COMMIT"
    try:
        conn = get_connection()
        try:
            row = conn.execute(
                "SELECT Action FROM DirectorCommands WHERE Phrase = ?", (phrase,)
            ).fetchone()
        finally:
            conn.close()
    except Exception as e:
        print(f"[DIRECTOR] Command gate lookup failed, treating as revision: {e}")
        return None
    action = row[0] if row else None
    print(f"[DIRECTOR] Command gate -> {action} ({phrase!r})")
    return action


def format_dialogue(user_message, story_context):
    """
    Formatter for DIALOGUE branch — takes raw user input and formats it to
    match the established POV, voice, and attribution style of the story.
    Returns just the formatted dialogue line(s), ready to be prepended to
    the generation call's response.
    """
    prompt = load_prompt_file("director_dialogue_formatter.txt")
    if not prompt:
        print("[DIRECTOR] Dialogue formatter prompt missing, skipping format pass.")
        return f"\"{user_message}\""  # fallback: bare quoted line

    try:
        formatted = llm_client.get_completion(
            system_prompt=prompt,
            user_text=f"Story context:\n{story_context}\n\nRaw input to format:\n{user_message}",
            max_tokens=200,
            temperature=0.0,
            timeout=30,
        ).strip()
        print(f"[DIRECTOR] Formatted dialogue: {formatted!r}")
        return formatted
    except Exception as e:
        print(f"[DIRECTOR] Dialogue formatting failed ({e}), using bare quote fallback.")
        return f"\"{user_message}\""


def _ask_classifier_question(system_prompt, user_text, expected_answers, question_label,
                              preferred_tier=None):
    """
    Single decision-tree node — asks a narrow yes/no or binary question,
    returns the matched answer or None on failure (which triggers fallback).
    No memory, no large prompt, just the question and the text.

    preferred_tier: optional, one of None, "small", "large". Q1-Q3 pass
        "small" explicitly (hard-routes to the small backend, dumdum mode).
        Q0 is left at None — default payload-size routing for the backend,
        but no thinking flags sent, so the server's own default applies.
    """
    try:
        raw = llm_client.get_completion(
            system_prompt=system_prompt,
            user_text=user_text,
            max_tokens=8,
            temperature=0.0,
            timeout=30,
            preferred_tier=preferred_tier,
        ).strip().upper()
        for answer in expected_answers:
            if answer in raw:
                print(f"[DIRECTOR] {question_label} -> {answer} (raw: {raw!r})")
                return answer
        print(f"[DIRECTOR] {question_label} returned unparseable ({raw!r}), continuing tree.")
        return None
    except Exception as e:
        print(f"[DIRECTOR] {question_label} failed ({e}), continuing tree.")
        return None


def _classify_via_llm_tree(user_message, speaking_as):
    """
    Decision-tree classifier — chains narrow binary questions instead of
    asking the model to hold all five categories at once. Each question is
    focused enough for a 4B to answer reliably; failures at any node fall
    through to the next question rather than dying. Final fallback: SCENE.
    No memory, no large prompt — just the text and the question. Prompts
    loaded from separate files (director_classifier_q*.txt) for easy tuning.
    """
    # Q0: Is this finished narrative or a request?
    q0_prompt = load_prompt_file("director_classifier_q0.txt")
    if not q0_prompt:
        print("[DIRECTOR] Q0 prompt missing, treating as REQUEST -> continuing to Q1")
        q0_answer = "REQUEST"
    else:
        if len(user_message) < 400:
            q0_answer = "REQUEST"
        else:
            q0_answer = _ask_classifier_question(q0_prompt, user_message, ["PROSE", "DIRECTION"], "Q0 (finished or request)")
        if q0_answer == "PROSE":
            print(f"[DIRECTOR] Classification complete: PROSE")
            return "PROSE"

    # Q1: Is the user speaking as their character, or directing as author?
    q1_prompt = load_prompt_file("director_classifier_q1.txt")
    if not q1_prompt:
        print("[DIRECTOR] Q1 prompt missing, treating as CHARACTER -> continuing to Q2")
        q1_answer = "CHARACTER"
    else:
        if len(user_message) > 200:
            q1_answer = _ask_classifier_question(q1_prompt, f"Current character: {speaking_as}\n\nInput: {user_message}", ["CHARACTER", "AUTHOR"], "Q1 (character or author)", preferred_tier="small")
        else:
            q1_answer = "CHARACTER"
    if q1_answer == "AUTHOR":
        print(f"[DIRECTOR] Classification complete: DIRECTION")
        return "DIRECTION"

    # Q2: Does this contain an action with a real, uncertain outcome that needs resolving?
    q2_prompt = load_prompt_file("director_classifier_q2.txt")
    if not q2_prompt:
        print("[DIRECTOR] Q2 prompt missing, treating as NO -> continuing to Q3")
        q2_answer = "NO"
    else:
        q2_answer = _ask_classifier_question(
            q2_prompt,
            user_message,
            ["YES", "NO"],
            "Q2 (action needs resolving)",
            preferred_tier="small",
        )
    if q2_answer == "YES":
        print(f"[DIRECTOR] Classification complete: ACTION")
        return "ACTION"

    # Q3: Does this contain dialogue or speech meant to be spoken?
    q3_prompt = load_prompt_file("director_classifier_q3.txt")
    if not q3_prompt:
        print("[DIRECTOR] Q3 prompt missing, falling back to SCENE")
        q3_answer = "NO"
    else:
        q3_answer = _ask_classifier_question(
            q3_prompt,
            user_message,
            ["YES", "NO"],
            "Q3 (contains speech)",
            preferred_tier="small",
        )
    if q3_answer == "YES":
        print(f"[DIRECTOR] Classification complete: DIALOGUE")
        return "DIALOGUE"

    # Fallback: if we get here, no clear answer — everything else is SCENE.
    print(f"[DIRECTOR] Classification tree exhausted, falling back to {DIRECTOR_FALLBACK_TYPE}")
    return DIRECTOR_FALLBACK_TYPE


def classify_director_input(user_message, speaking_as, mode="shadow", min_confidence=0.90):
    """
    Entry point for Director input typing. Naive bayes (director_input_
    classifier.py) runs alongside the LLM decision tree:
      shadow - tree decides; NB verdict is only logged for later comparison.
      live   - NB decides when its confidence >= min_confidence; below that
               (or with no trained model) the tree runs as the fallback.
    Every call is logged to DirectorInputLog so it can be hand-labeled and
    trained on. All NB/logging failures are swallowed — they must never
    break the classification itself.
    """
    nb_label = nb_conf = None
    conn = None
    try:
        import director_input_classifier as nb
        conn = get_connection()
        model = nb.load_model(conn)
        if model:
            nb_label, nb_conf = nb.predict(model, user_message)
            print(f"[DIRECTOR] NB verdict: {nb_label} ({nb_conf:.2f})")
    except Exception as e:
        print(f"[DIRECTOR] NB classifier unavailable, using LLM tree: {e}")

    if mode == "live" and nb_label and nb_conf >= min_confidence:
        llm_label = None
        result = nb_label
        print(f"[DIRECTOR] Classification complete (NB live): {result}")
    else:
        llm_label = result = _classify_via_llm_tree(user_message, speaking_as)

    if conn is not None:
        try:
            nb.log_decision(conn, speaking_as, user_message, llm_label, nb_label, nb_conf)
        except Exception as e:
            print(f"[DIRECTOR] Could not log classification: {e}")
        finally:
            conn.close()
    return result


def strip_pronouns_to_name(text, name):
    """
    Resolver-only scrub — replaces first-person pronouns with the acting
    character's name so the resolver can't tell (or favor/punish) whichever
    side is the user. Not critical-path precision, just enough to keep the
    LLM's native helpfulness from tilting the outcome.
    """
    replacements = {
        r"\bI\b": name,
        r"\bme\b": name,
        r"\bmy\b": f"{name}'s",
        r"\bmine\b": f"{name}'s",
        r"\bmyself\b": name,
    }
    scrubbed = text
    for pattern, replacement in replacements.items():
        scrubbed = re.sub(pattern, replacement, scrubbed, flags=re.IGNORECASE)
    return scrubbed


def resolve_action(user_message, speaking_as, context_blob):
    """
    Second small LLM call — adjudicates a proposed ACTION before prose
    generation. Never told which character is the player. Defaults to
    UNCONTESTED on failure so an infra hiccup doesn't stall or unfairly
    punish the scene.
    """
    scrubbed_action = strip_pronouns_to_name(user_message, speaking_as)
    prompt = load_prompt_file("action_resolver.txt")
    if not prompt:
        return "UNCONTESTED", "The action proceeds without further complication."
    try:
        raw = llm_client.get_completion(
            system_prompt=prompt,
            user_text=f"{context_blob}\n\nProposed event: {scrubbed_action}",
            max_tokens=80,
            temperature=0.3,
            timeout=60,
            preferred_tier="small",
        ).strip()
        lines = [ln.strip() for ln in raw.splitlines() if ln.strip()]
        if not lines:
            raise ValueError("empty resolver output")
        label = lines[0].upper()
        justification = " ".join(lines[1:]) if len(lines) > 1 else ""
        if label not in ("UNCONTESTED", "SUCCESS", "PARTIAL", "FAILURE"):
            label = "UNCONTESTED"
        print(f"[DIRECTOR] Action resolver gate -> {label} ({justification!r})")
        return label, justification
    except Exception as e:
        print(f"[DIRECTOR] Action resolver failed, defaulting to UNCONTESTED: {e}")
        return "UNCONTESTED", "The action proceeds without further complication."


# ══════════════════════════════════════════════════════════════════════════════
# Dossier system — automatic per-turn maintenance pass (design doc §5)
# ══════════════════════════════════════════════════════════════════════════════
#
# Runs after a Director Mode commit AND after that commit's seed has
# finished writing — see controller.py's run_seed_then_dossier_maintenance,
# which is the only correct caller. Calling this before seed completes
# risks the new-entity step (below) racing seed.py's own keyword/memory
# writes for the same turn.
#
# Every step here is a narrow, single-purpose call (classifier or patch),
# same design principle as the classifier tree above — kept within reach
# of a local 4B/7B model rather than one large reasoning call per dossier.

def build_dossier_text(dossier):
    """
    Multi-line dossier block, headed by full_name. Duplicated here from
    recall.py deliberately, not imported — this module has no import
    relationship with recall.py (avoids a second script depending on
    recall.py's internals for something this small), and the two copies
    have no correctness constraint forcing them to match byte-for-byte.
    Keep them in sync by eye if the dossier field set ever changes.
    """
    if not dossier:
        return ""
    fields = [
        ("Goals", dossier.get("goals")),
        ("Limitations", dossier.get("limitations")),
        ("Assets", dossier.get("assets")),
        ("Knowledge", dossier.get("knowledge")),
        ("Location", dossier.get("location")),
        ("Relationships", dossier.get("relationships")),
        ("Knowledge Gaps", dossier.get("knowledge_gaps")),
    ]
    lines = [dossier.get("full_name") or "(unnamed)"]
    lines += [f"    {label}: {val}" for label, val in fields if val]
    return "\n".join(lines)


def get_dossier_snapshot(mem_ref):
    """Reads the current Dossiers row for mem_ref as a flat dict (same
    shape as recall.py's dossier records), or None if no row exists."""
    conn = get_connection()
    cursor = conn.execute(
        "SELECT FullName, Goals, Limitations, Assets, Knowledge, "
        "Location, Relationships, KnowledgeGaps FROM Dossiers WHERE MemRef = ?",
        (mem_ref,),
    )
    row = cursor.fetchone()
    conn.close()
    if not row:
        return None
    return {
        "full_name": row[0], "goals": row[1], "limitations": row[2],
        "assets": row[3], "knowledge": row[4], "location": row[5],
        "relationships": row[6], "knowledge_gaps": row[7],
    }


def write_dossier_fields(mem_ref, edits):
    """
    Targeted UPDATE — only the columns present in `edits` are touched;
    every other field on the row is left exactly as it was. `edits` keys
    must already be real column names (see _FIELD_HEADER_TO_COLUMN) —
    this function trusts its caller rather than validating again.
    """
    if not edits:
        return
    set_clause = ", ".join(f"{col} = ?" for col in edits)
    params = list(edits.values()) + [time.time(), mem_ref]
    conn = get_connection()
    conn.execute(f"UPDATE Dossiers SET {set_clause}, UpdatedAt = ? WHERE MemRef = ?", params)
    conn.commit()
    conn.close()


def ensure_dossier_shell(mem_ref, full_name):
    """
    Creates an empty Dossiers row (FullName only, every field NULL) if one
    doesn't already exist for this MemRef. Idempotent — safe to call on
    every pass through the new-entity step even if a prior turn already
    created the row.
    """
    conn = get_connection()
    cursor = conn.execute("SELECT 1 FROM Dossiers WHERE MemRef = ?", (mem_ref,))
    if not cursor.fetchone():
        conn.execute(
            "INSERT INTO Dossiers (MemRef, FullName, UpdatedAt) VALUES (?, ?, ?)",
            (mem_ref, full_name, time.time()),
        )
        conn.commit()
    conn.close()


def find_gist_mem_ref(conn, name, canon=None):
    """
    Deterministic keyword-existence check, no LLM involved. Returns the
    MemRef of THE dedicated per-entity gist for `name` under `canon` — a
    derived Memory row whose ENTIRE keyword set is exactly {name}, in that
    canon (or open memory when canon is blank) — or None if no such gist
    exists yet.

    Requiring the full keyword set to match, not just "contains", is what
    excludes DreamCycle time-block summaries that merely mention the name.
    A canon that has never been seeded can hold no gist, so that is None
    too, never a fallback to a gist from somewhere else.
    """
    cid = memory_index.canon_id(conn, canon)
    if memory_index.canon_key(canon) and cid is None:
        print(f"[DOSSIER] Canon {canon!r} has no memories seeded yet — no gist can match.")
        return None
    return memory_index.gist_with_handles(conn, {name}, cid)


def create_stub_gist(name, essentials_text, canon=None):
    """
    Lightweight backdated gist for a brand-new entity (§5 step 5, §7).
    Creates the keyword if it doesn't exist yet, inserts a derived
    (RecallCount < 0) Memory row backdated to STUB_GIST_BACKDATE, and
    links it via Nodes to the entity's own keyword, inside `canon` (open
    memory when blank) so canon-locked recall can find it. DreamCycle refines or
    replaces this properly on its own schedule; if the entity never
    comes up again, it just sits harmlessly unused.

    Returns the new MemRef.

    NOTE: assumes Memory.MemRef and Keywords.KeyID are SQLite rowid-alias
    INTEGER PRIMARY KEYs, so cursor.lastrowid gives the right value after
    each INSERT — true for every table this module has seen the schema
    for (Dossiers included), but this function is guessing at Memory's
    and Keywords' actual CREATE TABLE, which I haven't seen. Worth
    confirming against seed.py's own insert path before trusting this in
    production — if seed.py generates IDs differently, this needs to
    match it.
    """
    name = name if name.islower() else name.upper()      # a handle: a lower-case stem stays, a name is upper case
    conn = get_connection()
    cid = memory_index.canon_id(conn, canon, create=True)

    cursor = conn.execute(
        "INSERT INTO Memory (Memory, OriginTime, RecallCount, CanonID) VALUES (?, ?, ?, ?)",
        (essentials_text, STUB_GIST_BACKDATE, -1, cid),
    )
    mem_ref = cursor.lastrowid
    memory_index.link_handles(conn, mem_ref, [name])
    conn.commit()
    memory_index.embed_memory(conn, mem_ref)   # searchable by the vector lane right away
    conn.close()
    print(f"[DOSSIER] Stub gist created for {name!r} (MemRef={mem_ref}, canon={canon!r})")
    return mem_ref


def generate_stub_gist_text(name, committed_text):
    """
    Brief 'character card' fill for a stub gist — physical description,
    occupation, distilled basics only, not the full DreamCycle treatment.
    Falls back to a bare placeholder line on failure so a new dossier
    isn't blocked by an infra hiccup.
    """
    prompt = load_prompt_file("dossier_stub_gist.txt")
    if not prompt:
        print(f"[DOSSIER] Stub gist prompt missing for {name!r}, using bare placeholder.")
        return f"{name} — newly introduced, no further detail recorded yet."
    try:
        return llm_client.get_completion(
            system_prompt=prompt,
            user_text=f"Newly introduced entity: {name}\n\nText they appear in:\n{committed_text}",
            max_tokens=WRITER_MAX_TOKENS,
            temperature=0.3,
            timeout=WRITER_TIMEOUT,
            preferred_tier="large",
        ).strip()
    except Exception as e:
        print(f"[DOSSIER] Stub gist generation failed for {name!r}, using bare placeholder: {e}")
        return f"{name} — newly introduced, no further detail recorded yet."


def entity_mentioned(full_name, text):
    """
    Deterministic prefilter for the dossier change gate: True if any
    name token (3+ chars, whole-word, case-insensitive) of full_name
    appears in text. Fails open (True) when the name has no usable token.
    Pronoun-only references are deliberately not caught — the passage has
    to name the entity for its dossier to be reviewed.
    """
    tokens = [t for t in re.split(r"[^\w']+", full_name or "") if len(t) >= 3]
    if not tokens:
        return True
    return any(re.search(rf"\b{re.escape(t)}\b", text, re.IGNORECASE) for t in tokens)


def dossier_needs_update(full_name, dossier_fields, committed_text):
    """
    Narrow yes/no gate — "does anything just committed to canon change
    this entity's dossier?" Defaults to 'no' on any failure, same
    fail-quiet convention as classify_director_command: an infra hiccup
    should mean a missed update, not a crashed commit.
    """
    if not entity_mentioned(full_name, committed_text):
        print(f"[DOSSIER] Change gate for {full_name!r} -> NO (not mentioned, no LLM call)")
        return False
    prompt = load_prompt_file("dossier_change_classifier.txt")
    if not prompt:
        print("[DOSSIER] Change classifier prompt missing, skipping update check.")
        return False
    dossier_text = build_dossier_text({"full_name": full_name, **dossier_fields})
    user_text = f"Current dossier:\n{dossier_text}\n\nText just committed to canon:\n{committed_text}"
    try:
        answer = llm_client.get_completion(
            system_prompt=prompt,
            user_text=user_text,
            max_tokens=10,
            temperature=0.0,
            timeout=30,
            preferred_tier="small",
        ).strip().upper()
        result = answer.startswith("YES")
        print(f"[DOSSIER] Change gate for {full_name!r} -> {'YES' if result else 'NO'} (raw: {answer!r})")
        return result
    except Exception as e:
        print(f"[DOSSIER] Change gate failed for {full_name!r}, defaulting to NO: {e}")
        return False


def _parse_dossier_fields(raw):
    """
    Parses '<FIELD>: <value>' lines — see dossier_field_patch.txt for the
    exact header vocabulary the prompt is expected to produce
    (_FIELD_HEADER_TO_COLUMN's keys). A value of 'NONE' (case-insensitive)
    explicitly clears that field. Unrecognized lines are silently
    ignored rather than raising — a stray preamble line shouldn't sink an
    otherwise-good patch.
    """
    edits = {}
    for line in raw.splitlines():
        line = line.strip()
        if not line or ":" not in line:
            continue
        header, _, value = line.partition(":")
        column = _FIELD_HEADER_TO_COLUMN.get(header.strip().upper())
        if not column:
            continue
        value = value.strip()
        # "NONE", "None.", "**NONE**" and "N/A" all mean "nothing to record"
        edits[column] = None if value.strip("*_ ").rstrip(".").upper() in ("NONE", "N/A") else value
    return edits


def apply_dossier_patch(mem_ref, full_name, dossier_fields, committed_text):
    """
    Second call, only made when dossier_needs_update() said yes — asks for
    the actual field edits and writes them back. Only fields the LLM
    names get touched; everything else on the row is left alone.
    """
    prompt = load_prompt_file("dossier_field_patch.txt")
    if not prompt:
        print("[DOSSIER] Field patch prompt missing, skipping edit.")
        return
    dossier_text = build_dossier_text({"full_name": full_name, **dossier_fields})
    user_text = f"Current dossier:\n{dossier_text}\n\nText just committed to canon:\n{committed_text}"
    try:
        raw = llm_client.get_completion(
            system_prompt=prompt,
            user_text=user_text,
            max_tokens=WRITER_MAX_TOKENS,
            temperature=0.2,
            timeout=WRITER_TIMEOUT,
            preferred_tier="large",
        )
        edits = _parse_dossier_fields(raw)
        if not edits:
            print(f"[DOSSIER] Patch call for {full_name!r} returned no parseable field edits, raw: {raw!r}")
            return
        write_dossier_fields(mem_ref, edits)
        print(f"[DOSSIER] Updated {full_name!r} ({mem_ref}): {list(edits.keys())}")
    except Exception as e:
        print(f"[DOSSIER] Field patch failed for {full_name!r}, leaving dossier untouched: {e}")


def process_dossier_reasoning(mem_ref, full_name, dossier_snapshot, committed_text):
    """
    Shared classify-then-patch step, used identically for a dossier that
    was already pulled this turn and for one just created for a
    newly-named entity (whose snapshot is all-None going in — the
    classifier still gets a fair yes/no on whether the committed text
    tells us anything worth recording).
    """
    fields_only = {k: v for k, v in dossier_snapshot.items() if k not in ("full_name", "mem_ref")}
    if dossier_needs_update(full_name, fields_only, committed_text):
        apply_dossier_patch(mem_ref, full_name, fields_only, committed_text)


def scan_for_new_dossier_entities(committed_text):
    """
    Final call in the per-turn pass — "are there any new entities that
    need a dossier made or adjusted? Name them." Expects one name per
    line; a response of NONE (case-insensitive) means nothing new. Fails
    quiet to an empty list.
    """
    prompt = load_prompt_file("dossier_new_entity_scan.txt")
    if not prompt:
        print("[DOSSIER] New-entity scan prompt missing, skipping.")
        return []
    try:
        raw = llm_client.get_completion(
            system_prompt=prompt,
            user_text=committed_text,
            max_tokens=100,
            temperature=0.0,
            timeout=30,
            preferred_tier="small",
        )
        names = [ln.strip() for ln in raw.splitlines() if ln.strip()]
        names = [n for n in names if n.upper() != "NONE"]
        print(f"[DOSSIER] New-entity scan -> {names!r}")
        return names
    except Exception as e:
        print(f"[DOSSIER] New-entity scan failed, assuming none: {e}")
        return []


def run_dossier_maintenance_pass(dossiers_pulled, committed_text, canon=None):
    """
    Automatic per-turn maintenance pass (design doc §5). MUST run after
    the turn's text is both committed AND seeded — see the module-level
    note above this section. Call via controller.py's
    run_seed_then_dossier_maintenance, not directly from a request thread.

    dossiers_pulled: the dossier records already fetched during this
    turn's generation call (see recall.recall(..., return_dossiers=True)
    via controller.get_memories) — reused as-is, no re-fetch, per §5.

    canon: now used by the new-entity step (steps 4-5) below, to make
    sure a gist founded here gets tagged into the right canon partition
    and to make sure find_gist_mem_ref doesn't return a same-named gist
    from a different canon. dossiers_pulled itself needs no canon
    filtering here — recall() already applied the canon hard-gate before
    those records were ever pulled, so anything in the list is already
    canon-correct.
    """
    if not committed_text or not committed_text.strip():
        return

    # Steps 1-3: re-check every dossier already pulled into this turn.
    for record in dossiers_pulled or []:
        process_dossier_reasoning(
            record["mem_ref"], record["full_name"], record, committed_text,
        )

    # Steps 4-5: scan for newly-introduced entities and give each a dossier.
    already_seen = {r["full_name"] for r in (dossiers_pulled or [])}
    for name in scan_for_new_dossier_entities(committed_text):
        if name in already_seen:
            continue  # already handled in the loop above this same pass
        conn = get_connection()
        mem_ref = find_gist_mem_ref(conn, name, canon)
        conn.close()
        if mem_ref is None:
            essentials = generate_stub_gist_text(name, committed_text)
            mem_ref = create_stub_gist(name, essentials, canon)
        ensure_dossier_shell(mem_ref, name)
        snapshot = get_dossier_snapshot(mem_ref)
        if snapshot:
            process_dossier_reasoning(mem_ref, name, snapshot, committed_text)


# ══════════════════════════════════════════════════════════════════════════════
# Dossier system — on-demand entity/dossier creation (design doc §6)
# ══════════════════════════════════════════════════════════════════════════════
#
# The Director Mode attachment-button flow. Unlike the automatic pass
# above, there is NO LLM judgment call on whether to act — the user has
# already named the entity, so the only questions left are the
# deterministic keyword/gist-existence checks in §6 step 3. Putting
# "which entities need a dossier" on the user rather than an LLM is
# deliberate (see §6's note on the superseded pre-filter+classifier
# design): the user already knows who they've been discussing.
#
# Wiring note: this module has no UI of its own. The attachment button's
# icon-swap while in Director Mode, and the adjacent text box it reads
# from, are frontend concerns — is_valid_entity_name() below is what a
# Flask route should call on that text box's current contents before
# ever calling create_or_update_dossier_on_demand(), so the "empty or
# anomalous input -> ask for a name" branch never touches the DB.

def is_valid_entity_name(text):
    """
    Deterministic sanity check for the button's adjacent text box (§6
    step 1) — no LLM judgment call on whether to proceed. Empty,
    whitespace-only, longer than ENTITY_NAME_MAX_CHARS, or containing a
    newline (clearly not a single name — "way too much text, obviously
    malformed") all fail. A Flask route should treat a False here as
    "pop the 'what character?' prompt," not call the function below at
    all.
    """
    if not text or not text.strip():
        return False
    text = text.strip()
    if len(text) > ENTITY_NAME_MAX_CHARS:
        return False
    if "\n" in text:
        return False
    return True


def fetch_memory_texts_for_keyword(conn, name, canon=None):
    """
    Every existing Memory row's text currently linked to this keyword,
    oldest first. Only meaningful in the no-gist-yet branch of
    create_or_update_dossier_on_demand — by construction every row found
    here is factual (RecallCount >= 0), since a derived one would already
    have been caught by find_gist_mem_ref.

    canon: same lock as recall — only memories of that canon (open memory
    when blank), so a gist founded on demand is never synthesized from a
    different canon's material.
    """
    cid = memory_index.canon_id(conn, canon)
    if memory_index.canon_key(canon) and cid is None:
        return []  # canon requested, nothing seeded under it yet
    cursor = conn.execute(
        "SELECT m.Memory, m.SpeakerID FROM Memory m JOIN Nodes n ON m.MemRef = n.MemID "
        "JOIN Keywords k ON k.KeyID = n.KeyID WHERE k.Key = ? AND m.CanonID IS ? AND m.RecallCount >= 0 "
        "ORDER BY m.OriginTime ASC",
        (name, cid),
    )
    speakers = memory_index.speaker_table(conn)
    return [memory_index.render(text, spk, speakers) for text, spk in cursor.fetchall()]


def generate_gist_from_existing_memories(name, memory_texts):
    """
    Lightweight gist-founding fill for the button's no-gist-yet branch —
    same 'brief character card, not the full DreamCycle treatment' spirit
    as generate_stub_gist_text() (reuses the same prompt file), but
    synthesized from every existing memory already on record for this
    name rather than a single turn's text. The button flow starts from
    real prior context (that's the whole reason it can found a gist at
    all instead of hard-aborting) — generate_stub_gist_text()'s "one
    turn's text" framing doesn't fit here, hence the separate function
    rather than reusing it directly.
    """
    prompt = load_prompt_file("dossier_stub_gist.txt")
    if not prompt:
        print(f"[DOSSIER] Stub gist prompt missing for {name!r}, using bare placeholder.")
        return f"{name} — existing memories on record, no synthesized summary available."
    memory_block = "\n".join(memory_texts) if memory_texts else "(no prior memory text found)"
    try:
        return llm_client.get_completion(
            system_prompt=prompt,
            user_text=f"Entity: {name}\n\nExisting memories about them:\n{memory_block}",
            max_tokens=WRITER_MAX_TOKENS,
            temperature=0.3,
            timeout=WRITER_TIMEOUT,
            preferred_tier="large",
        ).strip()
    except Exception as e:
        print(f"[DOSSIER] Gist-from-memories generation failed for {name!r}, using bare placeholder: {e}")
        return f"{name} — existing memories on record, no synthesized summary available."


def fetch_recent_memory_texts(conn, name, cid, max_chars=DOSSIER_RECENT_CHARS):
    """
    The newest ordinary memories tagged with `name` in canon `cid` (open memory when None), up to
    max_chars, returned oldest first so they read in story order. Derived rows (gists, summaries) are
    left out.
    """
    rows = conn.execute(
        "SELECT m.Memory, m.SpeakerID FROM Memory m JOIN Nodes n ON m.MemRef = n.MemID "
        "JOIN Keywords k ON k.KeyID = n.KeyID WHERE k.Key = ? AND m.CanonID IS ? AND m.RecallCount >= 0 "
        "ORDER BY m.OriginTime DESC", (name, cid)).fetchall()
    speakers = memory_index.speaker_table(conn)
    out, size = [], 0
    for text, spk in rows:
        line = memory_index.render(text, spk, speakers)
        if out and size + len(line) > max_chars:
            break
        out.append(line)
        size += len(line) + 1
    return list(reversed(out))


def build_dossier(mem_ref, full_name, gist_text, recent_texts):
    """
    The button's dossier step: every field written from scratch (dossier_build.txt) from the gist and
    the character's most recent memories. Fields the model leaves out or marks NONE are cleared, so
    nothing stale survives.

    The build runs on the large model with thinking on. Thinking sometimes loops (one line repeated
    until max_tokens runs out, no answer), so a verification pass follows: every field must have been
    written (a NONE line counts, "the text says nothing" is an answer; an absent line does not). Any
    field still missing is asked for again, on the same model with thinking off, which does not loop.
    A build that produced nothing at all, even after that, leaves the old dossier untouched.
    Returns True when the dossier was written.
    """
    prompt = load_prompt_file("dossier_build.txt")
    if not prompt:
        print("[DOSSIER] Build prompt missing, leaving the dossier untouched.")
        return False
    user_text = (f"Character: {full_name}\n\nGIST (their story so far):\n{gist_text}\n\n"
                 "MOST RECENT MEMORIES (oldest first; the last ones describe the present):\n"
                 + "\n".join(recent_texts))
    wanted = {h: c for h, c in _FIELD_HEADER_TO_COLUMN.items()}

    def ask(text, thinking):
        # Temperature 0.7: at 0.2 a long source sent the thinking into a loop more often.
        return _parse_dossier_fields(llm_client.get_completion(
            system_prompt=prompt, user_text=text, max_tokens=WRITER_MAX_TOKENS, temperature=0.7,
            timeout=WRITER_TIMEOUT, preferred_tier="large", thinking=thinking))

    try:
        edits = ask(user_text, True)
        missing = [h for h, c in wanted.items() if c not in edits]
        if missing:
            print(f"[DOSSIER] Verification: {full_name!r} came back without {', '.join(missing)}; "
                  "asking for those again with thinking off.")
            extra = ask(user_text + "\n\nWrite only these fields: " + ", ".join(missing) + ".", False)
            edits.update({c: v for c, v in extra.items() if c in {wanted[h] for h in missing}})
            still = [h for h in missing if wanted[h] not in edits]
            if still:
                print(f"[DOSSIER] Verification: {full_name!r} still lacks {', '.join(still)}; those stay empty.")
    except Exception as e:
        print(f"[DOSSIER] Build failed for {full_name!r}, leaving the dossier untouched: {e}")
        return False
    if not any(edits.values()):
        print(f"[DOSSIER] Build for {full_name!r} produced no usable fields, leaving the dossier untouched.")
        return False
    write_dossier_fields(mem_ref, {col: edits.get(col) for col in wanted.values()})
    print(f"[DOSSIER] Built {full_name!r} ({mem_ref}): {[k for k, v in edits.items() if v]}")
    return True


def create_or_update_dossier_on_demand(name, canon=None, progress=None):
    """
    The Director's dossier button: a full rebuild of a character's gist and dossier, the cleanup for a
    bad or messy pair. Call only after is_valid_entity_name(name) has passed.

      1. No keyword anywhere for this name -> abort; nothing is invented from nothing.
      2. The gist is rebuilt from scratch from every memory tagged with the name in this canon, with
         the dream's own folding (dream.derive_gist(from_scratch=True)): the old gist's text is
         ignored, and the row is rewritten in place so its dossier stays attached. Fewer memories
         than the dream's founding floor falls back to the lightweight stub gist.
      3. The dossier is built from scratch (build_dossier) from the new gist plus the character's
         most recent memories (DOSSIER_RECENT_CHARS): the gist carries the arc, the recent memories
         the present.

    canon: the active session's raw canon string; every lookup is locked to it, so a same-named
    character in another canon is never touched. progress: optional callable(str) for the UI's
    status line. Slow by nature (several large-model calls with thinking), so the controller runs it
    in the background.

    Returns {"status": "aborted", "reason": ...} or
            {"status": "ok", "founded_gist": bool, "gist_rebuilt": bool, "dossier_changed": bool,
             "mem_ref": ..., "gist_text": ..., "dossier": <snapshot dict>}
    """
    import dream  # only the button needs the dream's gist folding from this module
    say = progress or (lambda msg: None)
    name = name.upper()
    conn = get_connection()
    # the handle the name is filed under: the name itself (upper case) if it is one, else the word's stem
    handle = name if conn.execute("SELECT 1 FROM Keywords WHERE Key = ?", (name,)).fetchone() else stem(name)
    if not conn.execute("SELECT KeyID FROM Keywords WHERE Key = ?", (handle,)).fetchone():
        conn.close()
        print(f"[DOSSIER] On-demand request for {name!r} — no memories found, aborting.")
        return {"status": "aborted", "reason": f"No memories found on {name!r} — go discuss them in chat first."}
    cid = memory_index.canon_id(conn, canon)
    if memory_index.canon_key(canon) and cid is None:
        conn.close()
        return {"status": "aborted", "reason": f"Nothing has been written in the canon {canon!r} yet."}
    founded_gist = find_gist_mem_ref(conn, handle, canon) is None

    say(f"Rebuilding {name.title()}'s gist from their memories...")
    gist_prompt = dream.load_prompt(dream.DREAM_GIST_PROMPT_FILE)
    rebuilt = dream.derive_gist(conn, {handle}, gist_prompt, label=f"Dossier button gist ({name})", cid=cid,
                                canon_label=memory_index.canon_name(conn, cid) if cid is not None else "",
                                from_scratch=True)
    mem_ref = find_gist_mem_ref(conn, handle, canon)
    if mem_ref is None:
        # Fewer memories than the dream's founding floor: a stub gist from what there is.
        memory_texts = fetch_memory_texts_for_keyword(conn, handle, canon)
        conn.close()
        mem_ref = create_stub_gist(handle, generate_gist_from_existing_memories(name, memory_texts), canon)
        rebuilt = True
        conn = get_connection()
    gist_text = conn.execute("SELECT Memory FROM Memory WHERE MemRef = ?", (mem_ref,)).fetchone()[0]
    recent = fetch_recent_memory_texts(conn, handle, cid)
    conn.close()

    say(f"Writing {name.title()}'s dossier...")
    ensure_dossier_shell(mem_ref, name)
    before = get_dossier_snapshot(mem_ref)
    build_dossier(mem_ref, name, gist_text, recent)
    after = get_dossier_snapshot(mem_ref)
    print(f"[DOSSIER] On-demand rebuild for {name!r} (canon={canon!r}) complete — founded_gist={founded_gist}, "
          f"gist_rebuilt={bool(rebuilt)}, dossier_changed={before != after}")
    return {
        "status": "ok",
        "founded_gist": founded_gist,
        "gist_rebuilt": bool(rebuilt),
        "dossier_changed": before != after,
        "mem_ref": mem_ref,
        "gist_text": gist_text,
        "dossier": after,
    }
