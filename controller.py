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

import os
import sys
import re
import time
import shutil
import subprocess
import sqlite3
import logging
import threading
import importlib.util
import webbrowser
from datetime import datetime
from pathlib import Path
from flask import Flask, request, jsonify, send_from_directory
import uuid

import consolelog
import llm_client
import memory_index
import director_helpers

# ── Paths ────────────────────────────────────────────────────────────────────
ROOT = Path(__file__).parent.resolve()
PROMPTS_DIR = ROOT / "prompts"
LOG_FILE = ROOT / "LOG.txt"
SETTINGS_FILE = ROOT / "settings.txt"
# NOTE: confirm this matches the actual DB_PATH used in dream.py/recall.py/seed.py.
DB_PATH = ROOT / "MEM.db"

# ── Defaults (overridden by settings.txt) ────────────────────────────────────
SETTINGS = {
    "flask_port":       "5000",
    "listening_mode_char_threshold": "2000",
    "director_tail_paragraphs": "3",
    "director_tail_chars": "10000",
    "idle_dream_check_seconds": "300",
    "dream_min_interval_seconds": "7200",
    "dream_min_new_memories": "50",
    "lock_ui_during_processing": "true",
    "log_files": "true",                             # LOG.txt (every exchange) and CONSOLE.txt (the terminal); false = write neither
    "director_classifier_mode": "shadow",            # shadow | live
    "director_classifier_min_confidence": "0.90",
}

SETTINGS_DEFAULTS = dict(SETTINGS)   # restored on !reload so removed keys don't linger

BRIDGES = {}

# Every prompt file the code loads. Checked at startup and on !reload so a
# missing or misnamed file is reported once up front instead of silently
# degrading the feature that needs it.
REQUIRED_PROMPTS = (
    "system.txt", "reply_gate.txt", "persona_gist_prompt.txt", "amnesia_help.txt",
    "temporal_anchor_prompt.txt", "image_gate_prompt.txt", "image_description_prompt.txt",
    "bulk_extraction_prompt.txt",
    "dream_derive_prompt.txt", "dream_gist_prompt.txt", "dream_compress_prompt.txt", "dream_gist_judge.txt",
    "director_prompt.txt", "director_action_prompt.txt", "director_dialogue_prompt.txt",
    "director_scene_prompt.txt", "director_dialogue_formatter.txt",
    "director_revision_prompt.txt", "director_pacing_prompt.txt",
    "director_classifier_q0.txt", "director_classifier_q1.txt",
    "director_classifier_q2.txt", "director_classifier_q3.txt",
    "action_resolver.txt", "dossier_change_classifier.txt", "dossier_field_patch.txt",
    "dossier_new_entity_scan.txt", "dossier_stub_gist.txt",
)

AMNESIA_MEMORY = (
    "It seems your memory system is not working. "
    "You have amnesia and should be uncertain of anything not present in this context window."
)

IMAGES_DIR = ROOT / "images"
# STORIES_DIR, STORY_SEPARATOR, and DIRECTOR_TAIL_CHAR_CAP now live in
# director_helpers.py — this file reaches DIRECTOR_TAIL_CHAR_CAP via
# director_helpers.<name> where it needs it.
# Hard backstop only (~4 chars/token) — the practical size controls are
# settings.txt's director_tail_paragraphs (tail text) and the DB's own
# density (recall output). This just prevents it from running away.
DIRECTOR_MEMORY_CHAR_CAP = 10000   # the chat budget; the other half of Director's context is the rolling story tail

# Recall sizing. Conversation makes ONE deterministic recall pass (see run_conversational_pipeline);
# the reply gate reads the first RECALL_GATE_CHARS of it. Director revision still uses the smaller budget.
RECALL_CHARS = 10000
RECALL_GATE_CHARS = 4000

# Director Mode — input classifier routing
# DIRECTOR_FALLBACK_TYPE moved to director_helpers.py (its only use site).
DIRECTOR_INPUT_TYPES = ("DIALOGUE", "ACTION", "DIRECTION", "SCENE", "PROSE")

# Output caps. Generous on purpose: Director calls use the "large" tier (thinking on), and a thinking
# model spends part of the cap before writing a word. The prompts, not these caps, set the length.
DIRECTOR_MAX_TOKENS = {
    "DIALOGUE":  2048,
    "ACTION":    2048,
    "SCENE":     4096,
    "DIRECTION": 4096,
}
DIRECTOR_REVISION_MAX_TOKENS = 4096
CHAT_MAX_TOKENS = 4096

# Commit/retry gating lives in director_helpers.classify_director_command()
# (DirectorCommands table in MEM.db — edit phrases there).

# Pacing gate — read-only grade of the current story window, cached inside
# director_helpers.py and refreshed async (after commit, and on cold entry
# into director mode). Never blocks generation; a still-running or failed
# grade reads as BALANCED (fail-quiet), since it's advisory, not a gate.
# DIRECTOR_PACING_LABELS moved to director_helpers.py, its only use site.

# Seed convention: the literal AI-side placeholder meaning "no response generated"
NULL_RESPONSE_TOKEN = "NULL"

# Bulk ingestion (long-form / listening mode)
BULK_EXTRACTION_CHAR_CAP = 6000
LISTENING_ACK_MESSAGE = "Got it — I've received all of that."

# Notes that let the model explain a problem it can still talk about. When the model itself cannot be
# reached, llm_client.describe_failure() writes the message instead.
DEGRADED_MEMORY_NOTE = (
    "Note about your memory this turn: {problems}. Recall still worked on keyword handles, so your memory is "
    "reduced, not gone. Mention this if the user asks about your memory or seems to expect something you don't "
    "recall; the fix is to start the embedding server named in embedding_backend in settings.txt (or clear that "
    "setting to run on handles alone), then type !reload."
)
NO_VISION_NOTE = (
    "The user attached an image this turn. You are not vision-capable right now, so you cannot see it; it has been "
    "saved to memory with this conversation and will be described once a vision model is active."
)
VISION_FAILED_NOTE = (
    "The user attached an image this turn, but sending it to your vision model failed ({error}), so you cannot "
    "see it. Tell them plainly, and suggest checking that the model server was started with its vision "
    "projector (--mmproj) and that the right <tier>_vision flag is set in settings.txt."
)
EMPTY_REPLY_MESSAGE = (
    "[Lena] My language model answered with nothing. If it is a thinking model, it may have spent its whole "
    "token budget thinking; otherwise its console should say what went wrong. Your message is saved — try again."
)


EMPTY_DRAFT_MESSAGE = EMPTY_REPLY_MESSAGE.replace("Your message is saved", "Nothing was written")


class BackendDown(Exception):
    """Raised by the Director handlers when no draft could be generated; the message is for the user."""


llm_outage_noted = False    # one memory per outage, not one per unanswered message


def note_llm_outage(error):
    """Remembers, once per outage, that the model could not be reached (so Lena can explain it later)."""
    global llm_outage_noted
    if llm_outage_noted or not memory_scripts_ok:
        return
    llm_outage_noted = True
    why = llm_client.failure_reason(error)[0]
    text = (f"On {datetime.now().strftime('%A, %B %d, %Y at %I:%M %p')} my language model could not be reached "
            f"({why}), so messages were remembered but not answered until it came back.")

    def _run():
        try:
            spec = importlib.util.spec_from_file_location("seed", ROOT / "seed.py")
            seed_mod = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(seed_mod)
            seed_mod.note(text)
        except Exception as e:
            print(f"[WARNING] Could not record the outage: {e}")
    threading.Thread(target=_run, daemon=True).start()
DEFAULT_BULK_EXTRACTION_PROMPT = (
    "You have received a bulk import from the user — a large block of pasted "
    "text that has already been saved to memory in full.\n\n"
    "Given the text below, determine whether it contains a question, request, "
    "decision, instruction, or actionable/conversational statement that "
    "warrants a response.\n\n"
    "If there is nothing requiring a response, output exactly:\nNO\n\n"
    "Otherwise, output only a concise, faithful representation of what the "
    "user wants a response to. Do not answer it. Do not summarize the whole "
    "import. Extract only the part that needs a reply."
)

# ── Runtime state ─────────────────────────────────────────────────────────────
# Short-term context: the last (user, ai) exchange of the direct-chat channel.
# Bridges are deliberately stateless (past conversations come back through
# recall instead). The lock keeps it safe even with UI locking disabled.
short_term        = ("", "")
short_term_lock   = threading.Lock()
dream_running     = False
dream_thread      = None
dream_stop_event  = None
memory_text       = AMNESIA_MEMORY
memory_scripts_ok = False
director_state    = "idle"
director_draft    = None
director_canon    = None
# Dossier records pulled during this block's generation call — carried
# through to commit time so the maintenance pass can review them without
# re-querying. Reset alongside draft/canon in reset_director_state().
director_pulled_dossiers = None
director_memory_warning = None   # set when a Director turn had to go without recall; shown in the status line
# director_pacing_state moved to director_helpers.py — read via
# director_helpers.get_pacing_state(), written only by grade_story_pacing().

bridge_active     = None
bridge_mode       = None
bridge_canon      = None
bridge_module     = None
bridge_stop_event = None
bridge_thread     = None

idle_timer        = None

app = Flask(__name__)


class _SuppressDreamStatusLog(logging.Filter):
    """Keeps the /dream/status poll off the console without touching any
    other request logging — it fires every 20s from the UI and adds nothing
    worth seeing at the terminal."""
    def filter(self, record):
        return "/dream/status" not in record.getMessage()


logging.getLogger("werkzeug").addFilter(_SuppressDreamStatusLog())


# ══════════════════════════════════════════════════════════════════════════════
# Settings
# ══════════════════════════════════════════════════════════════════════════════

def load_settings():
    SETTINGS.update(SETTINGS_DEFAULTS)
    BRIDGES.clear()
    if not SETTINGS_FILE.exists():
        print("[WARNING] No settings.txt found, using defaults.")
        return
    with open(SETTINGS_FILE) as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            if "=" in line:
                key, _, val = line.partition("=")
                key = key.strip()
                val = val.strip()
                if key in SETTINGS:
                    SETTINGS[key] = val
                elif key.startswith("bridge_"):
                    platform = key[len("bridge_"):].strip().lower()
                    if platform and val:
                        BRIDGES[platform] = val
    print(f"[INFO] Settings loaded from {SETTINGS_FILE}")
    if BRIDGES:
        print(f"[INFO] Bridges configured: {', '.join(BRIDGES.keys())}")


def check_prompt_files():
    """Returns the names of any REQUIRED_PROMPTS missing from prompts/."""
    return [name for name in REQUIRED_PROMPTS if not (PROMPTS_DIR / name).is_file()]


def config_report():
    """Startup/reload sanity check: prints every problem and note, returns
    them as chat-friendly lines."""
    problems, notes = llm_client.startup_report()
    for name in check_prompt_files():
        problems.append(f"Missing prompt file: prompts/{name}")
    for line in problems:
        print(f"[CONFIG][WARNING] {line}")
    for line in notes:
        print(f"[CONFIG] {line}")
    return problems, notes


def open_settings_file():
    """Opens settings.txt in the host machine's default editor, if it has one."""
    opener = None
    if sys.platform.startswith("win"):
        try:
            os.startfile(str(SETTINGS_FILE))   # noqa: attribute only exists on Windows
            opener = "startfile"
        except Exception as e:
            print(f"[SETTINGS] Could not open the settings file: {e}")
    else:
        cmd = shutil.which("open" if sys.platform == "darwin" else "xdg-open")
        if cmd:
            try:
                subprocess.Popen([cmd, str(SETTINGS_FILE)], stdout=subprocess.DEVNULL,
                                 stderr=subprocess.DEVNULL)
                opener = cmd
            except Exception as e:
                print(f"[SETTINGS] Could not open the settings file: {e}")
    where = f"Opened in the host's default editor ({opener})." if opener else "No desktop editor available on the host."
    return (f"[Settings] {SETTINGS_FILE}\n{where} After saving, type !reload to apply "
            f"changes, or !restart for changes that need it (flask_port, bridge scripts).")


def reload_settings_command():
    """!reload — re-reads settings.txt for the controller and llm_client."""
    load_settings()
    llm_client.reload_settings()
    problems, notes = config_report()
    lines = ["[Settings] Reloaded settings.txt. (flask_port and already-running bridges need !restart.)"]
    lines += [f"WARNING: {p}" for p in problems]
    lines += [f"Note: {n}" for n in notes]
    return "\n".join(lines)


def restart_process():
    """Stops background work, then re-executes this process in place."""
    print("[RESTART] Restarting Lena...")
    try:
        request_stop_dream_and_wait()
        if bridge_active:
            stop_bridge()
    except Exception as e:
        print(f"[RESTART][WARNING] Clean stop failed, restarting anyway: {e}")
    os.environ["LENA_RESTARTED"] = "1"   # tells main() not to open a second browser tab
    sys.stdout.flush()
    sys.stderr.flush()
    # The web server's listening socket is inheritable, so it would survive the exec and the new process could not
    # bind its own port ("Address already in use") and would exit. Close every descriptor above stdin/stdout/stderr.
    os.closerange(3, 65536)
    os.execv(sys.executable, [sys.executable] + sys.argv)


def coprocessing_enabled():
    """True when UI locking during dream/bridge is disabled via settings."""
    return SETTINGS.get("lock_ui_during_processing", "true").strip().lower() == "false"


def ensure_wal_mode():
    """
    Forces WAL journal mode on MEM.db before anything else (recall.py,
    seed.py, dream.py) gets a chance to open it. WAL, once set, is a
    property of the database file itself and persists across every future
    connection — this just guarantees it's set on day one. Must run before
    check_memory_scripts(), since that's the first thing that opens the DB.
    """
    if not DB_PATH.exists():
        return
    try:
        conn = sqlite3.connect(str(DB_PATH))
        conn.execute("PRAGMA journal_mode=WAL")
        conn.close()
        print("[INFO] MEM.db journal mode set to WAL.")
    except Exception as e:
        print(f"[WARNING] Could not set WAL mode on MEM.db: {e}")


# ══════════════════════════════════════════════════════════════════════════════
# Prompt loading
# ══════════════════════════════════════════════════════════════════════════════

def load_prompt_file(filename):
    path = PROMPTS_DIR / filename
    if not path.exists():
        print(f"[WARNING] Missing prompt file: {path}")
        return ""
    return path.read_text(encoding="utf-8").strip()


def load_system_prompt():
    text = load_prompt_file("system.txt")
    if not text:
        print("[WARNING] system.txt is empty or missing — proceeding without system prompt.")
    return text


# ══════════════════════════════════════════════════════════════════════════════
# Memory / helper scripts
# ══════════════════════════════════════════════════════════════════════════════

def check_memory_scripts():
    global memory_scripts_ok
    required = ["recall.py", "dream.py", "seed.py"]
    missing = [s for s in required if not (ROOT / s).exists()]
    if missing:
        print(f"[WARNING] Memory scripts not found: {', '.join(missing)}")
        print("[WARNING] Falling back to amnesia memory string.")
        memory_scripts_ok = False
        return False

    try:
        spec = importlib.util.spec_from_file_location("recall", ROOT / "recall.py")
        recall_mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(recall_mod)
        if hasattr(recall_mod, "check_connection"):
            ok, msg = recall_mod.check_connection()
            if not ok:
                print(f"[WARNING] SQL connection failed: {msg}")
                memory_scripts_ok = False
                return False
    except Exception as e:
        print(f"[WARNING] Could not load recall.py: {e}")
        memory_scripts_ok = False
        return False

    print("[INFO] Memory scripts OK.")
    memory_scripts_ok = True
    return True


def resolve_temporal_anchor(text, previous=""):
    """Resolve a temporal reference in text (the previous exchange is context). Returns unix epoch or None."""
    if not memory_scripts_ok:
        return None
    anchor_path = ROOT / "temporal_anchor.py"
    if not anchor_path.exists():
        print("[WARNING] temporal_anchor.py missing — using the current time as the anchor.")
        return None
    try:
        spec = importlib.util.spec_from_file_location("temporal_anchor", anchor_path)
        anchor_mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(anchor_mod)
        return anchor_mod.resolve_anchor(text, previous=previous)
    except Exception as e:
        print(f"[WARNING] temporal_anchor.py error: {e}")
        return None


def gate_image(candidate, top_two_texts, user_message, ai_response=None, model_saw_image=False):
    """Call image_gate.py to decide whether a retrieved image is worth showing."""
    gate_path = ROOT / "image_gate.py"
    if not gate_path.exists():
        print("[WARNING] image_gate.py missing — images will not be shown.")
        return False
    try:
        spec = importlib.util.spec_from_file_location("image_gate", gate_path)
        gate_mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(gate_mod)
        return gate_mod.should_show_image(
            candidate["percentile"] * 100,
            candidate["origin_time"],
            top_two_texts,
            user_message,
            ai_response=ai_response,
            description=candidate.get("description"),
            model_saw_image=model_saw_image,
        )
    except Exception as e:
        print(f"[WARNING] image_gate.py error: {e}")
        return False


# ── Deferred image gate ──────────────────────────────────────────────────────
# The reply is sent first; the image decision runs afterwards in a background
# thread and the UI polls /image/result/<token> for the outcome.
IMAGE_JOBS = {}
IMAGE_JOBS_LOCK = threading.Lock()
IMAGE_JOB_TTL_SECONDS = 300

# Without a vision route the model never saw the image, so only a fairly
# strong match is worth putting in front of the gate (the pre-vision floor).
BLIND_IMAGE_SHOW_FLOOR = 0.10


def start_image_job(job):
    """Starts the post-reply image gate for a job dict from
    run_conversational_pipeline and returns the token the UI polls with."""
    token = uuid.uuid4().hex
    now = time.time()
    with IMAGE_JOBS_LOCK:
        for t in [t for t, j in IMAGE_JOBS.items() if now - j["created"] > IMAGE_JOB_TTL_SECONDS]:
            del IMAGE_JOBS[t]
        IMAGE_JOBS[token] = {"status": "pending", "image": None, "created": now}

    def _run():
        show = gate_image(job["candidate"], job["top_two_texts"], job["user_message"],
                          ai_response=job["ai_response"], model_saw_image=job["saw"])
        with IMAGE_JOBS_LOCK:
            IMAGE_JOBS[token].update(
                status="done",
                image=f"/images/{job['candidate']['filename']}" if show else None,
            )

    threading.Thread(target=_run, daemon=True).start()
    return token


def build_image_note(allowed, candidates_by_path):
    """System-prompt note describing the attached images to the model."""
    lines = []
    any_recalled = False
    for i, entry in enumerate(allowed, 1):
        if entry["recalled"]:
            any_recalled = True
            cand = candidates_by_path.get(entry["path"], {})
            date = datetime.fromtimestamp(cand["origin_time"]).strftime("%Y-%m-%d") if cand.get("origin_time") else "unknown date"
            desc = f" Stored description: {cand['description']}" if cand.get("description") else ""
            lines.append(f"Image {i}: recalled from memory (saved {date}).{desc}")
        else:
            lines.append(f"Image {i}: just uploaded by the user this turn.")
    note = "Attached images, in order:\n" + "\n".join(lines)
    if any_recalled:
        note += ("\nRecalled images are memories, not something the user just sent, and the user "
                 "cannot see them unless a separate step shows one after your reply — never say "
                 "\"as you can see\" about them. Use them to reason and stay consistent; mention "
                 "them only when relevant.")
    note += "\nIf no image is actually visible to you, ignore this note."
    return note


def get_memories(text, max_chars=None, anchor_time=None, include_images=False, canon=None,
                  return_dossiers=False, max_images=1, speaker=None, problems_out=None, **recall_options):
    """
    Call recall.py and return the memory block string.

    return_dossiers: when True, also returns the list of dossier records
    (one per dossier-carrying entity actually pulled and kept within the
    char budget this call) — used by the post-commit dossier maintenance
    pass so it can review exactly what this turn's generation call saw,
    without re-querying the DB. See director_helpers.run_dossier_maintenance_pass.
    """
    def _empty(note=AMNESIA_MEMORY):
        if include_images and return_dossiers:
            return note, [], []
        if include_images:
            return note, []
        if return_dossiers:
            return note, []
        return note

    if not memory_scripts_ok:
        return _empty(amnesia_note(["the memory scripts failed the start-up check (see the console output from when Lena started)"]))
    try:
        spec = importlib.util.spec_from_file_location("recall", ROOT / "recall.py")
        recall_mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(recall_mod)
        kwargs = {}
        if max_chars is not None:
            kwargs["max_chars"] = max_chars
        if anchor_time is not None:
            kwargs["anchor_time"] = anchor_time
        if canon:
            kwargs["canon"] = canon
        kwargs["include_images"] = include_images
        kwargs["return_dossiers"] = return_dossiers
        kwargs["max_images"] = max_images
        if speaker:
            kwargs["speaker"] = speaker    # recall uses it to find the person's gist and dossier
        kwargs.update(recall_options)      # Director: vector_queries, context_text, exclude_texts, entity_text

        result = recall_mod.recall(text, "", "", **kwargs)
        if problems_out is not None:
            problems_out.extend(getattr(recall_mod, "LAST_PROBLEMS", []))

        if include_images and return_dossiers:
            mem_text, images, dossiers = result
            return (mem_text if mem_text else AMNESIA_MEMORY), images, dossiers
        elif include_images:
            mem_text, images = result
            return (mem_text if mem_text else AMNESIA_MEMORY), images
        elif return_dossiers:
            mem_text, dossiers = result
            return (mem_text if mem_text else AMNESIA_MEMORY), dossiers
        else:
            return result if result else AMNESIA_MEMORY
    except Exception as e:
        if type(e).__name__ == "RecallUnavailable" and e.args and isinstance(e.args[0], list):
            reasons = e.args[0]
        else:
            reasons = [f"recall.py raised {type(e).__name__}: {e}"]
        print(f"[WARNING] Recall unavailable — amnesia mode: {'; '.join(reasons)}")
        return _empty(amnesia_note(reasons))


def amnesia_note(reasons):
    """
    What the model sees instead of memories when recall is broken: the amnesia notice, the troubleshooting
    guide from prompts/amnesia_help.txt, and what actually went wrong, so it can tell the user why it cannot
    remember and what to check.
    """
    help_text = load_prompt_file("amnesia_help.txt")
    parts = [AMNESIA_MEMORY]
    if help_text:
        parts.append(help_text)
    parts.append("What the memory system reported:\n" + "\n".join(f"- {r}" for r in reasons))
    return "\n\n".join(parts)


def long_form_threshold():
    """Server-side single source of truth for the bulk-ingestion char threshold."""
    try:
        return int(SETTINGS["listening_mode_char_threshold"])
    except (KeyError, ValueError):
        return 2000


def current_light_state():
    """Persistent indicator-light state, independent of any single request."""
    if dream_running:
        return "dream"
    if bridge_active:
        return "bridge"
    return "idle"


def get_short_term():
    """Returns the last direct-chat (user, ai) exchange, or ("", "")."""
    with short_term_lock:
        return short_term


def set_short_term(user_msg, ai_msg):
    global short_term
    with short_term_lock:
        short_term = (user_msg, ai_msg)


def clear_short_term():
    set_short_term("", "")


def run_conversational_pipeline(user_message, speaker, canon=None, image_filename=None,
                                 include_short_term_context=True, extra_note=None, source="ui"):
    """
    THE single deterministic conversational pipeline:
        TEMPORAL ANCHOR -> RECALL -> RESPONSE-WORTHINESS GATE ->
        LLM (with images, if a vision route exists) ->
        POST RESPONSE -> SEED -> (after the reply) IMAGE GATE

    Used by: direct chat, bulk-ingestion's extracted query, and bridge full-mode.
    `source` is "ui" for anything typed in Lena's own chat box (direct chat and bulk
    ingestion) and "bridge" for comments arriving through a bridge. The reply gate
    only runs for bridges: someone talking to Lena directly always gets an answer,
    with no LLM call spent deciding that.
    `include_short_term_context` controls whether the prior direct-chat
    exchange (see get_short_term) is threaded in and then replaced by this
    one — true for direct chat (including its bulk-ingestion offshoot),
    false for bridge comments, which are stateless and never inherit the
    user's last chat turn.

    The gate is a pure "does this warrant a response" check — it must never
    be used as a confidence/knowledge gate. Recall makes no LLM call, so the
    gate simply reads the start of the recalled memories; a decline skips the
    reply call entirely. The exchange
    is still seeded either way, since every exchange gets recorded
    automatically — a gate decline only means no reply was generated.

    Returns (ai_response_or_None, image_job_or_None, gated_out: bool). image_job
    is a dict for start_image_job() — the post-reply show/don't-show decision
    for a recalled image — or None when no image was a candidate.
    """
    global llm_outage_noted
    labeled_message = f"{speaker} said: {user_message}"
    prev_user, prev_ai = get_short_term() if include_short_term_context else ("", "")

    # TEMPORAL ANCHOR — a deterministic cue check on the message (and, for unmistakable references, the
    # user's previous message); the LLM resolves a date only when a cue is present. Otherwise: now.
    prev_said = re.sub(r"^.{0,60}? said: ", "", prev_user or "", count=1)
    anchor_time = resolve_temporal_anchor(user_message, previous=prev_said)

    # RECALL — one deterministic pass: identity, vector hits, handles; recency measured from the anchor
    vision = llm_client.vision_mode()
    recall_problems = []
    final_memories, image_candidates = get_memories(
        user_message, max_chars=RECALL_CHARS, anchor_time=anchor_time,
        include_images=True, canon=canon,
        max_images=llm_client.vision_max_images() if vision else 1,
        speaker=speaker, problems_out=recall_problems,
    )

    # RESPONSE-WORTHINESS GATE — bridges only (a direct chat message is always answered); reads the
    # same memories, trimmed
    if source == "bridge" and not gate_reply(user_message, final_memories[:RECALL_GATE_CHARS], speaker):
        run_seed(user_message, NULL_RESPONSE_TOKEN, speaker, image_filename, canon)
        print(f"[GATE] Declined — seeded, no reply generated for {speaker}.")
        return None, None, True

    image_candidates = [c for c in image_candidates if (IMAGES_DIR / c["filename"]).is_file()]
    top_two_texts = [
        line for line in final_memories.split("\n")
        if line.strip() and not line.startswith("[")
    ][:2]

    # Images the model will reason over: this turn's upload first, then
    # recalled ones. filter_images applies the cap and the privacy switch.
    vision_images = []
    candidates_by_path = {}
    if vision:
        if image_filename:
            vision_images.append({"path": IMAGES_DIR / image_filename, "recalled": False})
        for c in image_candidates:
            path = IMAGES_DIR / c["filename"]
            candidates_by_path[path] = c
            vision_images.append({"path": path, "recalled": True})
        vision_images = llm_client.filter_images(vision_images)

    # Build system prompt
    system_prompt = load_system_prompt()
    now = datetime.now().strftime("%A, %B %d, %Y - %I:%M %p")
    system_content = system_prompt if system_prompt else "You are a helpful AI assistant."
    system_content += f"\n\nCurrent date and time: {now}"
    if extra_note:
        system_content += f"\n\n{extra_note}"
    if final_memories and final_memories != AMNESIA_MEMORY:
        system_content += f"\n\n{final_memories}"
    if recall_problems:
        system_content += "\n\n" + DEGRADED_MEMORY_NOTE.format(problems="; ".join(recall_problems))
    if vision_images:
        system_content += f"\n\n{build_image_note(vision_images, candidates_by_path)}"
    if image_filename and not vision:
        system_content += f"\n\n{NO_VISION_NOTE}"

    # LLM reasoning
    messages = [{"role": "system", "content": system_content}]
    prev_user, prev_ai = get_short_term() if include_short_term_context else ("", "")
    if prev_user and prev_ai:
        messages.append({"role": "user",      "content": prev_user})
        messages.append({"role": "assistant", "content": prev_ai})
    messages.append({"role": "user", "content": labeled_message})

    images_sent = 0
    llm_error = None
    try:
        ai_response, meta = llm_client.get_completion(
            messages=messages,
            max_tokens=CHAT_MAX_TOKENS,
            temperature=0.7,
            timeout=180,
            images=vision_images or None,
            return_meta=True,
            vision_failure_note=VISION_FAILED_NOTE if image_filename else None,
        )
        images_sent = meta["images_sent"]
    except Exception as e:
        llm_error, ai_response = e, ""

    # NO REPLY — the model could not be reached or said nothing. The message is still remembered (with no
    # reply), the outage is remembered once, and the user gets a plain explanation from the code itself.
    # A bridge posts nothing.
    if llm_error is not None or not ai_response.strip():
        message = llm_client.describe_failure(llm_error) if llm_error is not None else EMPTY_REPLY_MESSAGE
        print(f"[CHAT][WARNING] No reply generated: {llm_error if llm_error is not None else 'the model returned nothing'}")
        run_seed(user_message, NULL_RESPONSE_TOKEN, speaker, image_filename, canon)
        if llm_error is not None:
            note_llm_outage(llm_error)
        append_log(labeled_message, message)
        if source == "bridge":
            return None, None, True
        return message, None, False
    llm_outage_noted = False

    # POST RESPONSE
    if include_short_term_context:
        set_short_term(labeled_message, ai_response)
    append_log(labeled_message, ai_response)

    # SEED
    run_seed(user_message, ai_response, speaker, image_filename, canon)

    # IMAGE CANDIDATE — the best recalled image goes to the post-reply gate.
    # If the model actually saw it, any recalled match qualifies (the gate
    # decides); if it worked blind, only a strong match is worth asking about.
    image_job = None
    if image_candidates:
        best = image_candidates[0]
        saw = images_sent > 0 and (IMAGES_DIR / best["filename"]) in {e["path"] for e in vision_images}
        if saw or best["percentile"] >= BLIND_IMAGE_SHOW_FLOOR:
            image_job = {
                "candidate": best, "top_two_texts": top_two_texts,
                "user_message": user_message, "ai_response": ai_response, "saw": saw,
            }

    return ai_response, image_job, False


def extract_bulk_query(user_message):
    """
    Small, deterministic LLM call: does this bulk dump contain a question,
    request, decision, or actionable statement Lena should respond to?
    Returns None if not (or on failure — fails closed to silence, not error),
    else the extracted query string.
    """
    prompt = load_prompt_file("bulk_extraction_prompt.txt") or DEFAULT_BULK_EXTRACTION_PROMPT
    excerpt = user_message[:BULK_EXTRACTION_CHAR_CAP]
    try:
        result = llm_client.get_completion(
            system_prompt=prompt,
            user_text=excerpt,
            max_tokens=300,
            temperature=0.1,
            timeout=60,
            preferred_tier="small",
        ).strip()
    except Exception as e:
        print(f"[LISTEN] Bulk extraction failed: {e}")
        raise
    if not result or result.strip().upper() == "NO":
        return None
    return result


def run_dream():
    """
    Async, fire-and-forget dream cycle.

    Owns a fresh threading.Event for this run and hands it to dream.run()
    as stop_event. Each call reloads dream.py as a brand-new module via
    importlib (see below), so the event can't live as module-level state
    inside dream.py itself — it has to be created and held here, and
    passed in explicitly, or a controller-side stop request would have no
    way to reach whichever dream_mod instance is actually running.
    dream_thread is likewise kept here so request_stop_dream_and_wait()
    below has something to join() on.
    """
    global dream_running, dream_thread, dream_stop_event
    if not memory_scripts_ok:
        return
    dream_stop_event = threading.Event()
    def _run(stop_event):
        global dream_running
        dream_running = True
        try:
            spec = importlib.util.spec_from_file_location("dream", ROOT / "dream.py")
            dream_mod = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(dream_mod)
            if hasattr(dream_mod, "run"):
                print("[DREAM] Running manual dream cycle...")
                dream_mod.run(stop_event=stop_event)
                print("[DREAM] Dream cycle complete.")
        except Exception as e:
            print(f"[WARNING] dream.py error: {e}")
        finally:
            dream_running = False
    dream_thread = threading.Thread(target=_run, args=(dream_stop_event,), daemon=True)
    dream_thread.start()


def request_stop_dream_and_wait():
    """
    Cooperative stop, called from the chat route instead of the old
    soft-lock bail-out message. Signals the running dream cycle to finish
    whatever LLM call + DB write it's currently mid-item on and exit at
    the next pass boundary (see dream.py's "Cooperative interruption"
    docstring section), then blocks on dream_thread.join() until it
    actually has. No polling on either side — join() is the
    synchronization point, and its return is the confirmation.

    Cost is bounded by one LLM call + one DB write (typically a few
    seconds, worst case something like a minute), not by how much of the
    cycle was left to run. A no-op if no dream cycle is currently active.
    """
    global dream_thread, dream_stop_event
    if not dream_running or dream_stop_event is None:
        return
    print("[DREAM] Stop requested — waiting for dream cycle to reach a safe exit point.")
    dream_stop_event.set()
    if dream_thread is not None:
        dream_thread.join()
    print("[DREAM] Dream cycle stopped — proceeding with pending command.")


def run_dream_sync():
    """Blocking dream cycle — used on shutdown."""
    global dream_running
    if not memory_scripts_ok:
        return
    dream_running = True
    try:
        spec = importlib.util.spec_from_file_location("dream", ROOT / "dream.py")
        dream_mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(dream_mod)
        if hasattr(dream_mod, "run"):
            print("[SHUTDOWN] Running dream cycle before shutdown...")
            dream_mod.run()
            print("[SHUTDOWN] Dream cycle complete.")
    except Exception as e:
        print(f"[WARNING] dream.py error during shutdown: {e}")
    finally:
        dream_running = False


def check_idle_dream_conditions():
    """
    Loads dream.py and asks it whether an automatic dream cycle is warranted
    right now. All DB-side judgment (time-since-last-pass, memory-count-
    since-last-pass) lives in dream.py, matching how recall.py/seed.py own
    their own queries. Fails closed (no dream) on any error.
    """
    if not memory_scripts_ok:
        return False
    try:
        spec = importlib.util.spec_from_file_location("dream", ROOT / "dream.py")
        dream_mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(dream_mod)
        if not hasattr(dream_mod, "should_run_dream"):
            return False
        min_interval = int(SETTINGS["dream_min_interval_seconds"])
        min_new_memories = int(SETTINGS["dream_min_new_memories"])
        return bool(dream_mod.should_run_dream(min_interval, min_new_memories))
    except Exception as e:
        print(f"[IDLE-DREAM] should_run_dream() check failed, skipping: {e}")
        return False


def reset_idle_timer():
    """
    (Re)starts the idle-to-dream timer. No polling loop — a single
    threading.Timer that gets cancelled and restarted on genuine UI
    activity (chat() calls this on every real turn). If it's ever allowed
    to run out uninterrupted, _idle_timer_fired() below checks whether a
    dream cycle is warranted, then reschedules itself either way — this
    keeps the check going during long idle stretches (e.g. background
    bridge activity slowly satisfying the memory-count gate) without ever
    needing a tight polling loop.
    """
    global idle_timer
    if idle_timer is not None:
        idle_timer.cancel()
    try:
        idle_seconds = int(SETTINGS["idle_dream_check_seconds"])
    except (KeyError, ValueError):
        idle_seconds = 300
    if idle_seconds <= 0:                    # 0 or less: no automatic dreaming ("!dream" still runs one)
        idle_timer = None
        return
    idle_timer = threading.Timer(idle_seconds, _idle_timer_fired)
    idle_timer.daemon = True
    idle_timer.start()


def _idle_timer_fired():
    """
    Fires only when the idle window has elapsed with zero resets — i.e.
    the user hasn't typed anything (directly) in that whole window. A
    running bridge does NOT block this; bridges and dream cycles are
    allowed to run concurrently. Only an already-running dream cycle
    blocks re-firing.
    """
    if not dream_running and check_idle_dream_conditions():
        print("[IDLE-DREAM] Conditions met — triggering automatic dream cycle.")
        run_dream()
    reset_idle_timer()


def _do_seed(user_message, ai_response, speaker="Anonymous", image_filename=None, canon=None, document=False):
    if not memory_scripts_ok:
        return
    try:
        spec = importlib.util.spec_from_file_location("seed", ROOT / "seed.py")
        seed_mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(seed_mod)
        if hasattr(seed_mod, "seed"):
            seed_mod.seed(user_message, ai_response, speaker, image_filename, canon, document=document)
    except Exception as e:
        print(f"[WARNING] seed.py error: {e}")


def run_seed(user_message, ai_response, speaker="Anonymous", image_filename=None, canon=None, document=False):
    """Fire-and-forget seed."""
    if not memory_scripts_ok:
        return
    threading.Thread(
        target=_do_seed,
        args=(user_message, ai_response, speaker, image_filename, canon, document),
        daemon=True
    ).start()


def run_seed_sync(user_message, ai_response, speaker="Anonymous", image_filename=None, canon=None):
    """Blocking seed — used by listening mode."""
    _do_seed(user_message, ai_response, speaker, image_filename, canon)


def run_seed_then_dossier_maintenance(user_message, ai_response, speaker, image_filename, canon,
                                       dossiers_pulled, committed_text):
    """
    Director Mode commit only. Fire-and-forget, but as ONE background
    thread that seeds synchronously and only then runs the dossier
    maintenance pass — not run_seed() plus a second independent
    fire-and-forget call. run_seed() on its own returns as soon as its
    thread is *launched*, not when seeding actually finishes, so chaining
    a second async call after it would race seed.py's writes for this
    same turn. Calling _do_seed() (the blocking half) directly here
    inside our own thread is what actually guarantees the ordering
    director_helpers.run_dossier_maintenance_pass depends on.
    """
    if not memory_scripts_ok:
        return

    def _run():
        _do_seed(user_message, ai_response, speaker, image_filename, canon)
        director_helpers.run_dossier_maintenance_pass(dossiers_pulled, committed_text, canon)

    threading.Thread(target=_run, daemon=True).start()


# ══════════════════════════════════════════════════════════════════════════════
# External bridges (Reddit, Discord, etc.)
# ══════════════════════════════════════════════════════════════════════════════

def _load_bridge_module(platform):
    script = BRIDGES.get(platform)
    if not script:
        return None
    path = ROOT / script
    if not path.exists():
        print(f"[BRIDGE] Configured script not found: {path}")
        return None
    try:
        spec = importlib.util.spec_from_file_location(f"bridge_{platform}", path)
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        if not hasattr(mod, "run"):
            print(f"[BRIDGE] {script} has no run() function — cannot start.")
            return None
        return mod
    except Exception as e:
        print(f"[BRIDGE] Failed to load {script}: {e}")
        return None


def gate_reply(comment_text, memories, speaker):
    """Call reply_gate.py to decide whether a comment is worth replying to."""
    gate_path = ROOT / "reply_gate.py"
    if not gate_path.exists():
        print("[WARNING] reply_gate.py missing — defaulting to silence.")
        return False
    try:
        spec = importlib.util.spec_from_file_location("reply_gate", gate_path)
        gate_mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(gate_mod)
        return bool(gate_mod.should_reply(comment_text, memories, speaker))
    except Exception as e:
        print(f"[BRIDGE] reply_gate.py error, defaulting to silence: {e}")
        return False


def process_bridge_comment(comment_text, speaker):
    """
    Process a comment from an active bridge. This is the same conversational
    pipeline as direct chat — bridge input just enters through a different
    source. Two bridge-specific sub-modes on top of that:
      - "listen": deliberately never generates a reply, only records. This is
        a bridge configuration choice, separate from the response-worthiness
        gate (which still runs for "full" mode below).
      - "full": runs the standard pipeline. Short-term context is NOT
        threaded in — a bridge conversation is a distinct per-speaker
        thread, not a continuation of the user's own direct chat turn, and
        past exchanges come back through recall.
    """
    canon = bridge_canon

    if bridge_mode == "listen":
        # a dropped file is a document, not someone talking: its name is the speaker (Speakers table), and
        # its text is stored as it is, with no "<name> said:" label
        run_seed(comment_text, NULL_RESPONSE_TOKEN, speaker, None, canon, document=(bridge_active == "file"))
        print(f"[BRIDGE:{bridge_active}] Listen mode — seeded, no reply considered.")
        return None

    ai_response, _image_job, gated_out = run_conversational_pipeline(
        comment_text, speaker, canon=canon, image_filename=None,
        include_short_term_context=False, source="bridge",
    )
    if gated_out:
        print(f"[BRIDGE:{bridge_active}] Gate declined — seeded, no reply.")
        return None

    print(f"[BRIDGE:{bridge_active}] Replying to {speaker}.")
    return ai_response


def start_bridge(platform, mode, canon):
    """Loads and launches a bridge script in a background thread."""
    global bridge_active, bridge_mode, bridge_canon, bridge_module
    global bridge_stop_event, bridge_thread

    if bridge_active:
        return (f"[Bridge] {bridge_active} is already active ({bridge_mode} mode). "
                f"Type !quit first.")

    mod = _load_bridge_module(platform)
    if mod is None:
        if platform not in BRIDGES:
            return (f"[Bridge] No bridge configured for '{platform}'. "
                    f"Add 'bridge_{platform} = <script>.py' to settings.txt.")
        return f"[Bridge] Failed to load bridge for '{platform}' — check console."

    bridge_active     = platform
    bridge_mode       = mode
    bridge_canon      = canon
    bridge_module     = mod
    bridge_stop_event = threading.Event()

    bridge_thread = threading.Thread(
        target=mod.run,
        args=(mode, canon, process_bridge_comment, bridge_stop_event),
        daemon=True,
    )
    bridge_thread.start()

    canon_note = f"canon '{canon}'" if canon else "no canon (blank/general)"
    print(f"[BRIDGE] Started {platform} in {mode} mode, {canon_note}.")
    return (f"[Bridge] {platform} started in {mode} mode, {canon_note}. "
            f"Normal chat is locked until !quit.")


def stop_bridge():
    """Signals the active bridge to stop and clears bridge state."""
    global bridge_active, bridge_mode, bridge_canon, bridge_module
    global bridge_stop_event, bridge_thread

    if not bridge_active:
        return "[Bridge] No bridge is currently active."

    platform = bridge_active
    if bridge_stop_event:
        bridge_stop_event.set()

    bridge_active     = None
    bridge_mode       = None
    bridge_canon      = None
    bridge_module     = None
    bridge_stop_event = None
    bridge_thread     = None

    print(f"[BRIDGE] Stopped {platform}. Back to normal chat.")
    return f"[Bridge] {platform} stopped. Back to normal chat."


# ══════════════════════════════════════════════════════════════════════════════
# Logging
# ══════════════════════════════════════════════════════════════════════════════

def logs_enabled():
    """False only when the log_files setting says so; read live, so !reload applies it at once."""
    return SETTINGS.get("log_files", "true").strip().lower() != "false"


def append_log(user_msg, ai_msg):
    """Appends one exchange to LOG.txt, unless the log_files setting is false."""
    if not logs_enabled():
        return
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    with open(LOG_FILE, "a", encoding="utf-8") as f:
        f.write(f"----------\n")
        f.write(f"[{timestamp}]\n")
        f.write(f"{user_msg}\n")
        f.write(f"AI: {ai_msg}\n\n")


# ══════════════════════════════════════════════════════════════════════════════
# Listening mode — large text dump handling
# ══════════════════════════════════════════════════════════════════════════════

LISTENING_WARNING = (
    "WARNING, USER INPUT UNUSUALLY LARGE TEXT. IT IS SAVED BUT THAT WAS A LOT TO REMEMBER ALL AT ONCE. "
    "If your response doesn't seem to land, the user may want to restate or narrow their question."
)


def handle_bulk_ingestion(user_message, speaker, image_filename, canon=None):
    """
    Long-form / bulk-ingestion flow:
        SEED ALL -> BULK QUESTION EXTRACTION -> NO: ack, DONE
                                              -> query: re-enter the normal
                                                 conversational pipeline

    The whole payload is seeded durably regardless of size — no recall or
    context-window fitting is attempted against the raw dump itself. Only a
    small extraction call ever sees an excerpt of it. If something in there
    warrants a reply, the extracted query becomes a normal conversational
    input and runs through the exact same pipeline as direct chat (anchor,
    recall, gate, LLM, seed) — it is not a special
    "bulk response" path.

    Returns (ai_response, image_job_or_None) — see run_conversational_pipeline.
    """
    # Step 1 — seed the entire raw block immediately (AI side NULL)
    run_seed_sync(user_message, NULL_RESPONSE_TOKEN, speaker, image_filename, canon)
    print(f"[LISTEN] Seeded raw block ({len(user_message)} chars)")

    # Step 2 — bulk question extraction
    try:
        extracted_query = extract_bulk_query(user_message)
    except Exception as e:
        return (f"{LISTENING_ACK_MESSAGE} But I couldn't check it for a question: "
                f"{llm_client.describe_failure(e).removeprefix('[Lena] ')}"), None

    # Step 3 — nothing actionable found
    if extracted_query is None:
        print("[LISTEN] Nothing actionable found in bulk import.")
        return LISTENING_ACK_MESSAGE, None

    # Step 4 — re-enter the normal conversational pipeline with the extracted query
    print(f"[LISTEN] Extracted query: {extracted_query[:80]}")
    ai_response, image_job, gated_out = run_conversational_pipeline(
        extracted_query, speaker, canon=canon, image_filename=None,
        include_short_term_context=True,
        extra_note=LISTENING_WARNING,
    )
    if gated_out:
        return LISTENING_ACK_MESSAGE, None
    return ai_response, image_job


# ══════════════════════════════════════════════════════════════════════════════
# Director / Ghostwriter mode
# ══════════════════════════════════════════════════════════════════════════════

def reset_director_state(reason):
    """
    Central place for dumping any in-progress director draft. Used by the
    mode-switch bail-out, the bridge-command bail-out, and the explicit
    front-end ping (/director/reset) fired when the Ghost Writer toggle is
    switched off — that last one is what actually closes the leak where
    toggling off and back on without an intervening normal-mode message
    left the old draft/state sitting in the globals.
    """
    global director_state, director_draft, director_canon, director_pulled_dossiers
    if director_state != "idle":
        print(f"[DIRECTOR] State reset ({reason}) — draft dumped, was in '{director_state}'.")
    director_state = "idle"
    director_draft = None
    director_canon = None
    director_pulled_dossiers = None


def director_recall(input_text, canon, draft=None, speaking_as=None):
    """
    Director Mode's recall, shared by a new block and a revision (a revision is recalled like a new
    section: the draft plays the part of the newest text, and the note may bring in someone new).

    Uses the chat budget (the other half of the context is the rolling story tail, director_tail_chars), and
    spreads the vector hits over the story instead of asking once:
      new block: (input + end of the latest committed block) -> 2/3 of recall_vector_hits, then the three
                 committed blocks before it, working up the document -> 1/3 each
      revision:  (note + end of the draft) -> 2/3, the note alone -> 1/3, then the two latest committed
                 blocks -> 1/3 each
    The end of the newest text also contributes its rarest handles (context_text). Everyone named in the
    input or the newest text gets their dossier (entity_text). Anything already in the prompt (the raw tail
    paragraphs, the draft) is excluded so nothing is shown twice. No temporal anchor: story time is not
    wall-clock time.

    speaking_as: on the paths written from inside the scene (action, dialogue, scene), the character being played is
    passed as the speaker, so their gist and dossier are always recalled and their gist's handles join the identity
    lane, whether or not the input names them.

    Images: when a vision route exists, recalled images (character pictures, maps, the ship) come back too.
    They are for the model to reason over; Director Mode never shows them to the user.

    Returns (memory_block, dossier_records, warning, image_candidates). On a recall failure the amnesia note
    is NOT returned (it would end up in the prose); memory_block is "" and warning says what happened.
    """
    k = int(memory_index.settings()["recall_vector_hits"])
    tail_k, block_k = max(1, round(k * 2 / 3)), max(1, k // 3)
    full = director_helpers.read_story_full(canon)
    blocks = [b.strip() for b in full.split(director_helpers.STORY_SEPARATOR) if b.strip()] if full.strip() else []
    raw_tail = director_helpers.get_story_tail_chars(canon, chars=int(SETTINGS["director_tail_chars"]))
    newest = draft if draft is not None else (blocks[-1] if blocks else "")
    newest_end = newest[-1500:]
    queries = [(f"{input_text}\n\n{newest_end}".strip(), tail_k)]
    if draft is not None:
        queries.append((input_text, block_k))
        earlier = blocks[-2:]
    else:
        earlier = blocks[-4:-1]
    for blk in reversed(earlier):
        queries.append((blk[-1500:], block_k))
    exclude = [raw_tail] + ([draft] if draft else [])
    vision = llm_client.vision_mode()
    memories, images, dossiers = get_memories(
        input_text, max_chars=DIRECTOR_MEMORY_CHAR_CAP, canon=canon, return_dossiers=True,
        include_images=True, max_images=llm_client.vision_max_images() if vision else 1,
        vector_queries=queries, context_text=newest_end, exclude_texts=exclude,
        entity_text=f"{input_text}\n{newest}", speaker=speaking_as or None,
    )
    images = [c for c in (images or []) if vision and (IMAGES_DIR / c["filename"]).is_file()]
    if not memories or memories.startswith(AMNESIA_MEMORY):
        # get_memories() returns the bare AMNESIA_MEMORY marker for "nothing recalled" (a brand-new canon) and
        # the longer amnesia note only when recall actually failed; only the second deserves a warning.
        warning = None
        if memories and memories.startswith(AMNESIA_MEMORY) and memories != AMNESIA_MEMORY:
            warning = "Memory unavailable — this draft was written without recalled memories (see the console)."
            print(f"[DIRECTOR][WARNING] {warning}")
        return "", dossiers or [], warning, []
    return memories, dossiers or [], None, images


def director_vision(image_candidates):
    """(images to attach, system note) for recalled images in Director Mode — for reasoning only, never shown."""
    if not image_candidates:
        return [], ""
    by_path = {IMAGES_DIR / c["filename"]: c for c in image_candidates}
    allowed = llm_client.filter_images([{"path": p, "recalled": True} for p in by_path])
    if not allowed:
        return [], ""
    print(f"[DIRECTOR] Reasoning over {len(allowed)} recalled image(s): {[str(e['path'].name) for e in allowed]}")
    return allowed, build_image_note(allowed, by_path)


def handle_director_new_block(user_message, speaker, canon=None):
    """
    State one — starting a new story block. Classifies the input into a
    routing branch (DIALOGUE / ACTION / DIRECTION / SCENE / PROSE), then
    dispatches to branch-specific prompting. Memory assembly — one combined
    recall pass (user input + tail block) plus the raw literal tail — is
    shared by every generating branch; PROSE alone skips both retrieval and
    generation entirely. ACTION additionally runs the resolver on the raw
    tail block directly, ahead of the shared recall pass.

    speaking_as is currently just `speaker` — kept as a separate local name
    so branch logic reads clearly and doesn't assume the two will always
    be identical if that changes later.
    """
    global director_pulled_dossiers, director_memory_warning
    clear_short_term()
    director_pulled_dossiers = []
    director_memory_warning = None
    speaking_as = speaker
    labeled_message = f"{speaker} said: {user_message}"

    print(f"[DIRECTOR] New block — input from {speaker!r}: {user_message!r}")

    input_type = director_helpers.classify_director_input(
        user_message, speaking_as,
        mode=SETTINGS["director_classifier_mode"].strip().lower(),
        min_confidence=float(SETTINGS["director_classifier_min_confidence"]),
    )

    if input_type == "PROSE":
        # User-authored material — no generation call, straight to revision.
        return user_message

    # Tail block, capped as a hard backstop — the ACTION resolver's context
    # (recall builds its own queries, see director_recall). Settings'
    # director_tail_paragraphs is the practical size control on the story
    # side; this slice is just insurance against an oversized commit.
    tail_block = director_helpers.get_story_tail_block(canon)
    tail_for_context = tail_block[-director_helpers.DIRECTOR_TAIL_CHAR_CAP:] if tail_block else ""

    if input_type == "ACTION":
        # Resolver needs "what's true right now" — raw scene state — not
        # associative/compressed gist memories, which is exactly what
        # gists would blur away. Also avoids prefilling the full recall
        # blob twice (once for the resolver, once for generation).
        outcome_label, outcome_justification = director_helpers.resolve_action(user_message, speaking_as, tail_for_context)

    # Recall: input + newest story text, vector hits spread over the story, dossiers of everyone named,
    # nothing that is already in the raw tail below. See director_recall().
    in_scene = input_type in ("ACTION", "DIALOGUE", "SCENE")      # the character being played matters there, not for direction
    recalled_memories, pulled_dossiers, warning, image_candidates = director_recall(
        user_message, canon, speaking_as=speaking_as if in_scene else None)
    director_pulled_dossiers = pulled_dossiers
    director_memory_warning = warning
    vision_images, image_note = director_vision(image_candidates)

    # Raw tail text (literal prose, not recall)
    raw_tail = director_helpers.get_story_tail_chars(canon, chars=int(SETTINGS["director_tail_chars"]))

    memory_blob = ""
    if recalled_memories:
        memory_blob += f"{recalled_memories}\n\n"
    if raw_tail:
        memory_blob += f"[CURRENT STORY, MOST RECENT TEXT]\n{raw_tail}\n[/CURRENT STORY]"

    user_content = labeled_message

    if input_type == "ACTION":
        system_prompt = load_prompt_file("director_action_prompt.txt") or load_system_prompt()
        user_content = (
            f"{labeled_message}\n\n"
            f"Established outcome: {outcome_label} — {outcome_justification}"
        )
    elif input_type == "DIALOGUE":
        system_prompt = load_prompt_file("director_dialogue_prompt.txt") or load_system_prompt()
        # Format the dialogue first, independently of generation
        dialogue_formatted = director_helpers.format_dialogue(user_message, raw_tail or "")
    elif input_type == "SCENE":
        system_prompt = load_prompt_file("director_scene_prompt.txt") or load_system_prompt()
    else:  # DIRECTION — existing default behavior
        system_prompt = load_prompt_file("director_prompt.txt") or load_system_prompt()

    now = datetime.now().strftime("%A, %B %d, %Y - %I:%M %p")
    system_content = system_prompt if system_prompt else "You are a helpful AI assistant."
    system_content += f"\n\nCurrent date and time: {now}"
    if memory_blob:
        system_content += f"\n\n{memory_blob}"
    if image_note:
        system_content += f"\n\n{image_note}"

    # Pacing note — silent when BALANCED so it adds no noise to the common
    # case. get_pacing_state() reads a cached async grade (see
    # director_helpers.grade_story_pacing); never blocks, never triggers a call.
    pacing_state = director_helpers.get_pacing_state()
    print(f"[DIRECTOR] Pacing state at generation time: {pacing_state}")
    if pacing_state == "STALLED":
        system_content += (
            "\n\nPacing note: the recent story has been holding in place for a "
            "while. This beat should land something — a new detail, reaction, "
            "or development — rather than another beat of stasis."
        )
    elif pacing_state == "FAST":
        system_content += (
            "\n\nPacing note: the recent story has been moving quickly through "
            "developments. This beat can afford to breathe rather than "
            "introduce another escalation."
        )

    messages = [{"role": "system", "content": system_content}]
    messages.append({"role": "user", "content": user_content})

    try:
        generation_result = llm_client.get_completion(
            messages=messages,
            max_tokens=DIRECTOR_MAX_TOKENS[input_type],
            temperature=0.7,
            timeout=180,
            preferred_tier="large",
            images=vision_images or None,
        )
    except Exception as e:
        raise BackendDown(llm_client.describe_failure(e, saved=False))
    if not generation_result.strip():
        raise BackendDown(EMPTY_DRAFT_MESSAGE)
    # For DIALOGUE, prepend the formatted dialogue to the generation's scene/reaction
    if input_type == "DIALOGUE":
        return f"{dialogue_formatted}\n\n{generation_result}"
    return generation_result


def _director_payload(state):
    """The 'director' part of a /chat reply; carries a memory warning for the status line when recall failed."""
    payload = {"state": state}
    if director_memory_warning:
        payload["warning"] = director_memory_warning
    return payload


def handle_director_revision(user_message, current_draft, speaker, canon=None):
    """
    State two+ — editing a draft already held in RAM. Recalled like a new section (see director_recall):
    the note and the draft both shape recall, so an edit that brings in someone new brings their memories
    and dossier with it. Dossiers pulled here join the turn's list for the commit-time maintenance pass.
    """
    global director_pulled_dossiers, director_memory_warning
    print(f"[DIRECTOR] Revision note from {speaker!r}: {user_message!r}")
    memories, dossiers, director_memory_warning, image_candidates = director_recall(user_message, canon, draft=current_draft)
    vision_images, image_note = director_vision(image_candidates)
    known = {d["mem_ref"] for d in (director_pulled_dossiers or [])}
    director_pulled_dossiers = (director_pulled_dossiers or []) + [d for d in dossiers if d["mem_ref"] not in known]

    revision_prompt = load_prompt_file("director_revision_prompt.txt")
    if not revision_prompt:
        revision_prompt = (
            "You are revising an existing piece of creative writing. "
            "The text below is the CURRENT DRAFT, not something you are "
            "continuing — it is complete as far as content goes. The user "
            "will give you a note describing what to change. Apply only "
            "what the note asks for. Preserve every other sentence, "
            "detail, and piece of phrasing exactly as written. Do not "
            "regenerate the passage from scratch, do not add commentary "
            "or preamble, and do not wrap your reply in quotes or labels — "
            "output only the full revised draft."
        )

    now = datetime.now().strftime("%A, %B %d, %Y - %I:%M %p")
    system_content = f"{revision_prompt}\n\nCurrent date and time: {now}"
    if memories:
        system_content += f"\n\n{memories}"
    if image_note:
        system_content += f"\n\n{image_note}"

    labeled_message = (
        f"CURRENT DRAFT:\n{current_draft}\n\n"
        f"---\n"
        f"{speaker}'s revision note: {user_message}"
    )

    messages = [
        {"role": "system", "content": system_content},
        {"role": "user",   "content": labeled_message},
    ]

    try:
        revised = llm_client.get_completion(
            messages=messages,
            max_tokens=DIRECTOR_REVISION_MAX_TOKENS,
            temperature=0.7,
            timeout=180,
            preferred_tier="large",
            images=vision_images or None,
        )
    except Exception as e:
        raise BackendDown(llm_client.describe_failure(e, saved=False))
    if not revised.strip():
        raise BackendDown(EMPTY_DRAFT_MESSAGE)
    return revised


# ══════════════════════════════════════════════════════════════════════════════
# Flask UI
# ══════════════════════════════════════════════════════════════════════════════

HTML = """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>Lena</title>
<style>
  * { box-sizing: border-box; margin: 0; padding: 0; }
  body {
    background: #1a1a1a;
    color: #d4d4d4;
    font-family: 'Consolas', 'Courier New', monospace;
    font-size: 14px;
    display: flex;
    flex-direction: column;
    height: 100vh;
    height: 100dvh; /* iOS Safari: 100vh ignores the keyboard/toolbar, dvh doesn't */
    overflow: hidden;
  }
  #top-bar {
    background: #111;
    border-bottom: 1px solid #2a2a2a;
    padding: 4px 12px;
    display: flex;
    flex-wrap: wrap;
    align-items: center;
    row-gap: 6px;
    column-gap: 10px;
    font-size: 11px;
    color: #8f8f8f;
  }
  #top-bar label { color: #9a9a9a; }
  #name-input {
    background: #1a1a1a;
    border: 1px solid #333;
    color: #7ec8e3;
    padding: 2px 8px;
    font-family: inherit;
    font-size: 11px;
    border-radius: 3px;
    width: 140px;
    outline: none;
  }
  #name-input:focus { border-color: #555; }
  #canon-input {
    background: #1a1a1a;
    border: 1px solid #333;
    color: #d99ee3;
    padding: 2px 8px;
    font-family: inherit;
    font-size: 11px;
    border-radius: 3px;
    width: 120px;
    outline: none;
  }
  #canon-input:focus { border-color: #555; }
  /* locked while Director Mode is on: a story stays in the canon it started in */
  #canon-input[readonly] { opacity: 0.6; border-style: dashed; cursor: not-allowed; }
  #status-text {
    flex: 1;
    text-align: right;
    color: #888;
  }
  #listen-dot {
    width: 8px;
    height: 8px;
    border-radius: 50%;
    background: #333;
    transition: background 0.3s, box-shadow 0.3s;
  }
  #listen-dot.active {
    background: #e8b84b;
    box-shadow: 0 0 6px #e8b84b;
  }
  #listen-dot.flash {
    animation: listen-flash 0.6s ease-in-out infinite;
  }
  @keyframes listen-flash {
    0%, 100% { opacity: 1; }
    50% { opacity: 0.25; }
  }
  #listen-dot.dream-flash {
    background: #c0392b;
    animation: dream-flash 0.6s ease-in-out infinite;
  }
  @keyframes dream-flash {
    0%, 100% { opacity: 1; box-shadow: 0 0 6px #c0392b; }
    50% { opacity: 0.25; box-shadow: none; }
  }
  #listen-dot.dream-solid {
    background: #c0392b;
    box-shadow: 0 0 6px #c0392b;
  }
  #ghost-toggle-wrap {
    display: flex;
    align-items: center;
    gap: 6px;
    font-size: 11px;
    color: #9a9a9a;
  }
  #ghost-switch {
    position: relative;
    display: inline-block;
    width: 30px;
    height: 16px;
    flex-shrink: 0;
  }
  #ghost-switch input {
    opacity: 0;
    width: 0;
    height: 0;
  }
  #ghost-switch .slider {
    position: absolute;
    cursor: pointer;
    top: 0; left: 0; right: 0; bottom: 0;
    background: #2a2a2a;
    border: 1px solid #444;
    border-radius: 16px;
    transition: background 0.2s, border-color 0.2s;
  }
  #ghost-switch .slider::before {
    content: "";
    position: absolute;
    width: 12px;
    height: 12px;
    left: 2px;
    bottom: 2px;
    background: #888;
    border-radius: 50%;
    transition: transform 0.2s, background 0.2s;
  }
  #ghost-switch input:checked + .slider {
    background: #2a1a3a;
    border-color: #7e5ec8;
  }
  #ghost-switch input:checked + .slider::before {
    transform: translateX(14px);
    background: #b79ee8;
  }
  #history {
    flex: 1;
    overflow-y: auto;
    padding: 16px;
    display: flex;
    flex-direction: column;
    gap: 12px;
  }
  .msg-user { color: #7ec8e3; }
  .msg-ai { color: #d4d4d4; }
  .msg-label { color: #555; font-size: 11px; margin-bottom: 2px; }
  .msg-text { white-space: pre-wrap; line-height: 1.5; }
  #director-view {
    display: none;
    flex: 1;
    overflow-y: auto;
    padding: 16px;
    white-space: pre-wrap;
    line-height: 1.6;
  }
  #director-baseline { color: #888; }
  #director-draft { color: #d4d4d4; margin-top: 12px; }
  #input-area {
    border-top: 1px solid #2a2a2a;
    padding: 12px;
    display: flex;
    gap: 8px;
    background: #111;
  }
  #user-input {
    flex: 1;
    background: #222;
    border: 1px solid #333;
    color: #d4d4d4;
    padding: 8px 12px;
    font-family: inherit;
    font-size: 14px;
    resize: none;
    outline: none;
    border-radius: 3px;
    min-height: 38px;
    max-height: 120px;
  }
  #user-input:focus { border-color: #555; }
  #send-btn {
    background: #2a2a2a;
    border: 1px solid #444;
    color: #aaa;
    padding: 8px 18px;
    cursor: pointer;
    font-family: inherit;
    border-radius: 3px;
  }
  #send-btn:hover { background: #333; color: #d4d4d4; }
  #send-btn:disabled { opacity: 0.4; cursor: default; }
  #shutdown-btn {
    background: #1a0a0a;
    border: 1px solid #622;
    color: #c66;
    padding: 2px 10px;
    font-family: inherit;
    font-size: 11px;
    cursor: pointer;
    border-radius: 3px;
  }
  #shutdown-btn:hover { background: #2a1a1a; color: #e88; }
  #image-btn {
    background: #222;
    border: 1px solid #333;
    color: #aaa;
    padding: 8px 12px;
    cursor: pointer;
    font-family: inherit;
    font-size: 16px;
    border-radius: 3px;
  }
  #image-btn:hover { background: #2a2a2a; color: #d4d4d4; }
  #image-btn.staged { border-color: #7ec8e3; color: #7ec8e3; }
  /* Director Mode repurposes this button for the dossier flow (§6) rather
     than image attachment — a distinct color keeps it from reading as
     "an image is staged" when it's actually "this fires the dossier
     lookup instead." */
  #image-btn.dossier-mode { border-color: #c9a15a; color: #c9a15a; }
  #image-btn.dossier-mode.staged { border-color: #c9a15a; color: #c9a15a; }
  #image-label {
    align-self: center;
    font-size: 11px;
    color: #7ec8e3;
    max-width: 120px;
    overflow: hidden;
    text-overflow: ellipsis;
    white-space: nowrap;
  }
  .tb-group {
    display: flex;
    align-items: center;
    gap: 10px;
    flex-shrink: 0;
    white-space: nowrap;
  }
  #tb-fields { flex: 0 0 auto; }
  #tb-status { flex: 1 1 auto; min-width: 0; }
  #tb-actions { flex: 0 0 auto; }

  @media (max-width: 420px) {
    #name-input { width: 100px; }
    #canon-input { width: 90px; }
  }
  @media (max-width: 600px) {
    #name-input, #canon-input, #user-input { font-size: 16px; }
  }
</style>
</head>
<body>
<div id="top-bar">
  <div class="tb-group" id="tb-fields">
    <label>Speaking as:</label>
    <input type="text" id="name-input" placeholder="Anonymous" maxlength="40">
    <label title="Locks recall/storage to this world only — blank means unrestricted">Canon:</label>
    <input type="text" id="canon-input" placeholder="(none)" maxlength="60">
  </div>
  <div class="tb-group" id="tb-status">
    <div id="ghost-toggle-wrap" title="Ghost Writer mode">
      <label id="ghost-switch">
        <input type="checkbox" id="ghost-toggle">
        <span class="slider"></span>
      </label>
      <span>Ghost Writer</span>
    </div>
    <span id="listen-dot" title="Bridge activity / dream state"></span>
    <span id="status-text">Ready</span>
  </div>
  <div class="tb-group" id="tb-actions">
    <button id="shutdown-btn" onclick="shutdown()">Shutdown</button>
  </div>
</div>
<div id="history"></div>
<div id="director-view"><div id="director-baseline"></div><div id="director-draft"></div></div>
<div id="input-area">
  <input type="file" id="image-input" accept="image/*" style="display:none">
  <button id="image-btn" onclick="handleAttachClick()">📎</button>
  <span id="image-label"></span>
  <textarea id="user-input" placeholder="Type your message..." rows="1"></textarea>
  <button id="send-btn" onclick="sendMessage()">Send</button>
</div>
<script>
  const history    = document.getElementById('history');
  const input      = document.getElementById('user-input');
  const btn        = document.getElementById('send-btn');
  const statusText = document.getElementById('status-text');
  const nameInput  = document.getElementById('name-input');
  const canonInput = document.getElementById('canon-input');
  const imageInput = document.getElementById('image-input');
  const imageBtn   = document.getElementById('image-btn');
  const imageLabel = document.getElementById('image-label');
  const ghostToggle = document.getElementById('ghost-toggle');
  const shutdownBtn = document.getElementById('shutdown-btn');
  const directorView     = document.getElementById('director-view');
  const directorBaseline = document.getElementById('director-baseline');
  const directorDraft    = document.getElementById('director-draft');
  const listenDot  = document.getElementById('listen-dot');
  let currentDraftText   = '';
  // Tracks whether the *last known* dream state was running, so the
  // background poll below can tell "just started" / "just finished" apart
  // from "still running" / "still idle" without any button click driving it.
  let dreamWasRunning = false;

  // iOS Safari: 100dvh (CSS, above) covers most of this, but on some
  // versions the layout doesn't recompute cleanly when the on-screen
  // keyboard opens/closes, leaving #history unable to scroll back up.
  // Explicitly pinning body height to the visual viewport is the more
  // reliable fix where it's supported.
  if (window.visualViewport) {
    function syncViewportHeight() {
      document.body.style.height = window.visualViewport.height + 'px';
    }
    window.visualViewport.addEventListener('resize', syncViewportHeight);
    syncViewportHeight();
  }

  imageInput.addEventListener('change', function() {
    const f = imageInput.files[0];
    if (f) {
      imageLabel.textContent = f.name;
      imageBtn.classList.add('staged');
    } else {
      imageLabel.textContent = '';
      imageBtn.classList.remove('staged');
    }
  });

  // Director Mode repurposes the attachment button for the on-demand
  // dossier flow (design doc §6) rather than image attachment — see
  // enterDirectorMode/exitDirectorMode below for the icon/class swap.
  function handleAttachClick() {
    if (ghostToggle.checked) {
      requestDossier();
    } else {
      imageInput.click();
    }
  }

  async function requestDossier(nameOverride) {
    const name = nameOverride !== undefined ? nameOverride : input.value.trim();
    statusText.textContent = 'Starting dossier rebuild...';
    try {
      const resp = await fetch('/director/dossier', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ name: name, canon: canonInput.value.trim() }),
      });
      let data = await resp.json();

      if (data.status === 'need_name') {
        // Deterministic on the backend (§6 step 1) — the browser prompt
        // IS "pop a simple text box" for this embedded UI. An empty/
        // cancelled prompt just drops back to Ready rather than looping.
        const typed = window.prompt(data.prompt);
        if (typed && typed.trim()) {
          return requestDossier(typed.trim());
        } else {
          statusText.textContent = 'Ready';
        }
        return;
      }

      // The rebuild runs in the background (several large-model calls); poll until it reports back.
      while (data.status === 'running') {
        statusText.textContent = data.stage || 'Rebuilding dossier...';
        await new Promise(r => setTimeout(r, 3000));
        data = await (await fetch('/director/dossier/status?job=' + encodeURIComponent(data.job))).json();
      }

      if (data.status === 'aborted' || data.status === 'error') {
        statusText.textContent = data.reason;
      } else {
        const who = (data.dossier && data.dossier.full_name) || name;
        if (data.dossier_changed) {
          statusText.textContent = `Rebuilt ${who}'s gist and dossier.`;
        } else {
          statusText.textContent = `Rebuilt ${who}'s gist; the dossier came back unchanged.`;
        }
      }
    } catch (e) {
      statusText.textContent = 'Error';
    }
    // Status message is transient like every other statusText use in this
    // page (Composing.../Thinking.../Ready) — clear it back to Ready after
    // a beat rather than leaving a stale dossier result sitting there.
    setTimeout(() => { statusText.textContent = 'Ready'; }, 6000);
  }

  async function enterDirectorMode() {
    // Every story lives in a canon, and the canon can't change mid-story: it is locked while Director Mode is on.
    if (!canonInput.value.trim()) {
      ghostToggle.checked = false;
      statusText.textContent = "Type the story's name in the Canon box first, then switch on Director Mode.";
      setTimeout(() => { statusText.textContent = 'Ready'; }, 5000);
      return;
    }
    canonInput.readOnly = true;
    canonInput.title = 'Locked while Director Mode is on. Switch Director Mode off to change it.';
    history.style.display = 'none';
    directorView.style.display = 'block';
    directorDraft.textContent = '';
    currentDraftText = '';
    // Swap the attachment button into dossier mode — see handleAttachClick.
    imageBtn.textContent = '📇';
    imageBtn.classList.add('dossier-mode');
    imageBtn.title = 'Form/update a dossier for a named character';
    try {
      const canon = canonInput.value.trim();
      const resp = await fetch('/director/story?canon=' + encodeURIComponent(canon));
      const data = await resp.json();
      // Full, untruncated story for the human to read — separate from
      // the tail Lena reasons over (data.story), which stays bounded by
      // director_tail_paragraphs regardless of what's shown here.
      directorBaseline.textContent = data.full_story || '(No story written yet.)';
    } catch (e) {
      directorBaseline.textContent = '[Error loading story]';
    }
  }

  function exitDirectorMode() {
    canonInput.readOnly = false;
    canonInput.title = '';
    history.style.display = 'flex';
    directorView.style.display = 'none';
    // Swap the attachment button back to plain image attachment.
    imageBtn.textContent = '📎';
    imageBtn.classList.remove('dossier-mode');
    imageBtn.title = '';
  }

  ghostToggle.addEventListener('change', function() {
    if (this.checked) {
      enterDirectorMode();
    } else {
      exitDirectorMode();
      // Tell the backend right now, not on the next message — otherwise a
      // toggle-off/toggle-on with nothing typed in between leaves the old
      // draft/state sitting server-side and the next attempt gets treated
      // as a revision of the discarded draft instead of a fresh start.
      fetch('/director/reset', { method: 'POST' }).catch(() => {});
    }
  });

  function applyLightState(light) {
    // Persistent state (bridge/idle) — dream has its own dedicated flash/solid
    // sequence managed by the background poll below, so leave it alone here.
    if (light === 'dream') return;
    listenDot.classList.remove('active');
    if (light === 'bridge') {
      listenDot.classList.add('active');
    }
  }

  async function sendMessage() {
    const msg = input.value.trim();
    if (!msg) return;

    const speaker = nameInput.value.trim() || 'Anonymous';
    const canon = canonInput.value.trim();
    const ghostWriter = ghostToggle.checked;

    // STEP 1 — display the completed input immediately, before the controller
    // has done anything with it. Only applies to the plain chat view; Director
    // Mode has its own draft-based display and is a separate system.
    if (!ghostWriter) {
      const msgDiv = document.createElement('div');
      msgDiv.innerHTML = `
        <div class="msg-label">${speaker}</div>
        <div class="msg-text msg-user">${msg}</div>
      `;
      history.appendChild(msgDiv);
      history.scrollTop = history.scrollHeight;
    }

    input.value = '';
    btn.disabled = true;
    statusText.textContent = ghostWriter ? 'Composing...' : 'Thinking...';
    listenDot.classList.add('flash');

    const formData = new FormData();
    formData.append('message', msg);
    formData.append('name', speaker);
    formData.append('canon', canon);
    formData.append('ghost_writer', ghostWriter);

    const imgFile = imageInput.files[0];
    if (imgFile) {
      formData.append('image', imgFile);
    }

    try {
      const response = await fetch('/chat', {
        method: 'POST',
        body: formData
      });
      const data = await response.json();

      if (ghostWriter) {
        if (data.director && data.director.state === 'draft') {
          directorDraft.textContent = data.response;
          currentDraftText = data.response;
        } else if (data.director && data.director.state === 'committed') {
          const hasBaseline = directorBaseline.textContent &&
                               directorBaseline.textContent !== '(No story written yet.)';
          directorBaseline.textContent =
            (hasBaseline ? directorBaseline.textContent + '\\n\\n---\\n\\n' : '') + currentDraftText;
          directorDraft.textContent = '';
          currentDraftText = '';
        } else if (data.director && data.director.state === 'reset') {
          // Retry-phrase escape — draft scrapped, nothing committed.
          directorDraft.textContent = data.response;
          currentDraftText = '';
        }
        // state 'error': no draft could be generated; the draft on screen (if any) is kept and the
        // explanation goes to the status line below.
        directorView.scrollTop = directorView.scrollHeight;
      } else if (data.response) {
        // Empty response means the response-worthiness gate declined — the
        // exchange was still recorded server-side, there's just nothing to show.
        const aiDiv = document.createElement('div');
        const aiLabel = document.createElement('div');
        aiLabel.className = 'msg-label';
        aiLabel.textContent = 'Lena';
        aiDiv.appendChild(aiLabel);

        const aiText = document.createElement('div');
        aiText.className = 'msg-text msg-ai';
        aiText.textContent = data.response;
        aiDiv.appendChild(aiText);

        history.appendChild(aiDiv);
        history.scrollTop = history.scrollHeight;

        // Reply is already on screen; the image gate runs afterwards and
        // any image it approves is appended a moment later.
        if (data.image_pending) {
          showImageWhenReady(data.image_pending, aiDiv);
        }
      }

      if (data.light) applyLightState(data.light);
      statusText.textContent = (data.director && data.director.warning) ? data.director.warning : 'Ready';
    } catch (e) {
      statusText.textContent = 'Error';
    }

    imageInput.value = '';
    imageLabel.textContent = '';
    imageBtn.classList.remove('staged');
    listenDot.classList.remove('flash');
    btn.disabled = false;
    input.focus();
  }

  async function showImageWhenReady(token, aiDiv) {
    for (let i = 0; i < 60; i++) {
      await new Promise(r => setTimeout(r, 500));
      try {
        const resp = await fetch('/image/result/' + token);
        const data = await resp.json();
        if (data.status === 'done') {
          if (data.image) {
            const imgEl = document.createElement('img');
            imgEl.src = data.image;
            imgEl.style.maxWidth = '100%';
            imgEl.style.borderRadius = '4px';
            imgEl.style.marginTop = '8px';
            aiDiv.appendChild(imgEl);
            history.scrollTop = history.scrollHeight;
          }
          return;
        }
      } catch (e) {
        return;
      }
    }
  }

  input.addEventListener('keydown', function(e) {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault();
      sendMessage();
    }
  });

  input.focus();

  // Background poll — no manual "Dream" button anymore (dream cycles fire on
  // their own after idle, or via the hidden !dream escape), so this is the
  // only thing keeping the indicator light honest about a dream cycle nobody
  // in the UI explicitly asked for. Deliberately slow (20s) since it's purely
  // cosmetic — nothing time-sensitive depends on this being fast.
  const DREAM_POLL_MS = 20000;
  async function pollDreamStatus() {
    try {
      const resp = await fetch('/dream/status');
      const data = await resp.json();
      if (data.running) {
        if (!dreamWasRunning) {
          listenDot.classList.remove('active', 'flash', 'dream-solid');
          listenDot.classList.add('dream-flash');
          statusText.textContent = 'Dreaming...';
        }
        dreamWasRunning = true;
      } else if (dreamWasRunning) {
        listenDot.classList.remove('dream-flash');
        listenDot.classList.add('dream-solid');
        statusText.textContent = 'Ready';
        setTimeout(() => listenDot.classList.remove('dream-solid'), 1500);
        dreamWasRunning = false;
      }
    } catch (e) {
      // Silent — this is a background convenience poll, not core function.
    }
  }
  setInterval(pollDreamStatus, DREAM_POLL_MS);

  async function shutdown() {
    if (!confirm('Shut down Lena now?')) return;
    btn.disabled = true;
    shutdownBtn.disabled = true;
    listenDot.classList.remove('active', 'flash', 'dream-flash');
    listenDot.classList.add('dream-solid');
    statusText.textContent = 'Shutting down...';
    await fetch('/shutdown', {method: 'POST'});
  }
</script>
</body>
</html>"""


# ══════════════════════════════════════════════════════════════════════════════
# Flask routes
# ══════════════════════════════════════════════════════════════════════════════

@app.route("/")
def index():
    return HTML


@app.route("/chat", methods=["POST"])
def chat():
    """
    Main chat route — single deterministic workflow with conditional early
    exits (escape command -> long-form/bulk check -> normal pipeline).
    Director Mode is a separate system handled outside this workflow.
    """
    global director_state, director_draft, director_canon
    global bridge_active

    user_message = request.form.get("message", "").strip()
    speaker      = request.form.get("name", "Anonymous").strip()
    canon        = request.form.get("canon", "").strip()
    ghost_writer_mode = request.form.get("ghost_writer", "false").strip().lower() == "true"

    if not user_message:
        return jsonify({"response": "No message provided.", "light": current_light_state()})

    # Any genuine UI send resets the idle-to-dream timer. Pipeline-internal
    # traffic (bridge comments) never reaches this route, so it never resets it —
    # a bridge session running unattended can still get dreamed over.
    reset_idle_timer()

    # ── STEP 2: escape / controller command check — no LLM call ────────────
    if user_message.lower() == "!quit":
        return jsonify({"response": stop_bridge(), "light": current_light_state()})

    # Hardcoded manual dream escape — not surfaced anywhere in the UI, mirrors
    # the bridge command style. Lets you force a dream cycle on demand without
    # waiting on the idle/interval/count gate.
    if user_message.lower() == "!dream":
        if dream_running:
            msg = "[Dream] Already running."
        else:
            run_dream()
            msg = "[Dream] Manual dream cycle triggered."
        return jsonify({"response": msg, "light": current_light_state()})

    # Settings commands — handled here, before the generic !<bridge> matcher.
    if user_message.lower() == "!settings":
        return jsonify({"response": open_settings_file(), "light": current_light_state()})
    if user_message.lower() == "!reload":
        return jsonify({"response": reload_settings_command(), "light": current_light_state()})
    if user_message.lower() == "!restart":
        threading.Timer(0.5, restart_process).start()
        return jsonify({"response": "[Restart] Restarting Lena — refresh this page in a few seconds.",
                        "light": current_light_state()})

    bridge_cmd_match = re.match(r'^!(\w+)(?:\s+(listen))?$', user_message, re.IGNORECASE)
    if bridge_cmd_match:
        platform = bridge_cmd_match.group(1).lower()
        mode = "listen" if bridge_cmd_match.group(2) else "full"
        return jsonify({"response": start_bridge(platform, mode, canon), "light": current_light_state()})

    # Bridge is active — block normal chat, return status (soft UI lock).
    # Bridges and dream cycles are allowed to run concurrently with each
    # other regardless of this setting; this flag only controls whether the
    # *user* is still locked out of typing while either is in progress.
    if bridge_active and not coprocessing_enabled():
        return jsonify({
            "response": f"[Bridge] {bridge_active} is active ({bridge_mode} mode). "
                        f"Type !quit to return to normal chat.",
            "light": current_light_state()
        })

    # Dream cycle in progress — when the UI lock is enabled (the default),
    # stop it and wait for confirmed shutdown rather than bailing out with
    # a "try again" message. This is what removes the race that used to
    # produce timeouts when a command landed mid-dream-cycle: the command
    # is held here until dream cycle has actually finished its current
    # item and exited, not until the caller happens to retry after it's
    # done. When coprocessing is explicitly enabled via settings, dream
    # cycle is left running and this request proceeds concurrently with it.
    if dream_running and not coprocessing_enabled():
        request_stop_dream_and_wait()

    # Handle uploaded image, if any
    uploaded_image = request.files.get("image")
    image_filename = None
    if uploaded_image and uploaded_image.filename:
        ext = Path(uploaded_image.filename).suffix.lower()
        if ext not in llm_client.SUPPORTED_IMAGE_EXTS:
            print(f"[UPLOAD] Rejected {uploaded_image.filename!r}: unsupported type {ext!r}.")
            return jsonify({
                "response": "[Upload] Only image files are accepted: "
                            + ", ".join(sorted(llm_client.SUPPORTED_IMAGE_EXTS)) + ".",
                "light": current_light_state(),
            })
        image_filename = f"{int(time.time())}_{uuid.uuid4().hex[:8]}{ext}"
        IMAGES_DIR.mkdir(exist_ok=True)
        uploaded_image.save(str(IMAGES_DIR / image_filename))

    labeled_message = f"{speaker} said: {user_message}"

    # Director mode bail-out
    if not ghost_writer_mode and director_state != "idle":
        reset_director_state("switched to normal chat")

    # Director mode bail-out on bridge command
    is_bridge_command = (
        user_message.lower() == "!quit"
        or re.match(r'^!(\w+)(?:\s+(listen))?$', user_message, re.IGNORECASE)
    )
    if is_bridge_command and director_state != "idle":
        reset_director_state("bridge command")

    # Director mode branch
    if ghost_writer_mode:
        if director_state == "idle":
            # Every story lives in a canon-named file; there is no default story.
            if not director_helpers.normalize_canon(canon):
                msg = ("[Director] Director Mode needs a Canon. Type the story's name "
                       "in the Canon box (letters, numbers, spaces, - and _) and try again.")
                return jsonify({"response": msg, "director": {"state": "reset"}})
            director_canon = canon
            try:
                ai_response = handle_director_new_block(user_message, speaker, director_canon)
            except BackendDown as e:
                print(f"[DIRECTOR][WARNING] No draft generated: {e}")
                reset_director_state("no draft generated")
                append_log(labeled_message, str(e))
                return jsonify({"response": str(e), "director": {"state": "error", "warning": str(e)}})
            director_draft = ai_response
            director_state = "editing"
            append_log(labeled_message, ai_response)
            return jsonify({"response": ai_response, "director": _director_payload("draft")})
        else:
            # Deterministic command gate (DirectorCommands table in MEM.db):
            # RETRY scraps the draft with no memory write, COMMIT saves it,
            # anything else is a revision instruction. No LLM call.
            command = director_helpers.classify_director_command(user_message)
            if command == "RETRY":
                print(f"[DIRECTOR] Retry command matched in {user_message!r} — discarding draft.")
                reset_director_state("retry phrase")
                confirm_msg = "[Director] Draft discarded — go ahead and type your next attempt."
                append_log(labeled_message, confirm_msg)
                return jsonify({"response": confirm_msg, "director": {"state": "reset"}})

            if command == "COMMIT":
                commit_text = director_draft
                director_helpers.commit_story_block(director_canon, commit_text)
                # no speaker: the committed story is just the story, attributed to nobody
                run_seed_then_dossier_maintenance(
                    commit_text, NULL_RESPONSE_TOKEN, None, None, director_canon,
                    director_pulled_dossiers, commit_text,
                )
                story_name = director_helpers.normalize_canon(director_canon)
                confirm_msg = f"[Director] Committed to stories/{story_name}.txt ({len(commit_text)} chars)."
                append_log(labeled_message, confirm_msg)
                # Kick pacing re-grade now, in the dead time while the user
                # reads the confirmation — canon captured before the reset
                # below clears director_canon.
                director_helpers.grade_story_pacing_async(
                    director_canon, tail_n=int(SETTINGS["director_tail_paragraphs"]),
                )
                reset_director_state("committed")
                clear_short_term()
                return jsonify({"response": confirm_msg, "director": {"state": "committed"}})
            else:
                try:
                    ai_response = handle_director_revision(user_message, director_draft, speaker, director_canon)
                except BackendDown as e:
                    print(f"[DIRECTOR][WARNING] Revision not generated, draft kept: {e}")
                    append_log(labeled_message, str(e))
                    return jsonify({"response": str(e), "director": {"state": "error", "warning": str(e)}})
                director_draft = ai_response
                append_log(labeled_message, ai_response)
                return jsonify({"response": ai_response, "director": _director_payload("draft")})

    # ── STEP 3: long-form / bulk-ingestion check (server-side threshold) ───
    if len(user_message) > long_form_threshold():
        print(f"[LISTEN] Large input detected ({len(user_message)} chars)")
        ai_response, image_job = handle_bulk_ingestion(user_message, speaker, image_filename, canon)
        response_payload = {"response": ai_response, "light": current_light_state()}
        if image_job:
            response_payload["image_pending"] = start_image_job(image_job)
        return jsonify(response_payload)

    # ── NORMAL MODE: the single conversational pipeline ─────────────────────
    print(f"[CHAT] {speaker}: {user_message[:60]}")
    ai_response, image_job, gated_out = run_conversational_pipeline(
        user_message, speaker, canon=canon, image_filename=image_filename,
        include_short_term_context=True,
    )

    if gated_out:
        # Recorded (seeded inside the pipeline), but nothing warranted a reply.
        return jsonify({"response": "", "light": current_light_state()})

    response_payload = {"response": ai_response, "light": current_light_state()}
    if image_job:
        response_payload["image_pending"] = start_image_job(image_job)

    return jsonify(response_payload)


@app.route("/image/result/<token>")
def image_result(token):
    """Polled by the UI after a reply; reports the post-reply image gate."""
    with IMAGE_JOBS_LOCK:
        job = IMAGE_JOBS.get(token)
        if job is None:
            return jsonify({"status": "done", "image": None})
        return jsonify({"status": job["status"], "image": job["image"]})


@app.route("/dream/manual", methods=["POST"])
def dream_manual():
    """Trigger a manual dream cycle."""
    if dream_running:
        return jsonify({"status": "already running"})
    run_dream()
    return jsonify({"status": "started"})


@app.route("/dream/status", methods=["GET"])
def dream_status():
    """Poll dream cycle status."""
    return jsonify({"running": dream_running})


@app.route("/director/story", methods=["GET"])
def director_story():
    """
    Fetch the story tail for the current canon, plus the FULL story text
    for the human-facing display. Also kicks an async pacing re-grade —
    this is the "cold entry" trigger: fired the moment the Ghost Writer
    toggle goes on and the front end asks for the baseline, so the grade
    is warm by the time the first message goes out rather than waiting
    for a commit that hasn't happened yet this session.

    Two separate fields on purpose, not one switched by size: "story" is
    the same tail (director_tail_paragraphs-bounded, DIRECTOR_TAIL_CHAR_CAP-
    capped) that's fed to Lena as context elsewhere in this file — kept
    as-is so anything relying on that shape doesn't change. "full_story"
    is the entire canon file, unbounded, for the Director Mode baseline
    view — James reading the whole thing shouldn't be limited by
    whatever window size Lena's currently tuned to reason over; those are
    two different concerns (what she needs in context vs. what he wants
    to see) that happened to share one field before this.
    """
    canon = request.args.get("canon", "").strip()
    tail_n = int(SETTINGS["director_tail_paragraphs"])
    tail = director_helpers.get_story_tail_paragraphs(canon, n=tail_n)
    full_story = director_helpers.read_story_full(canon)
    director_helpers.grade_story_pacing_async(canon, tail_n=tail_n)
    return jsonify({"story": tail, "full_story": full_story})


# Dossier rebuilds started by the Director's button, by job id (see director_dossier).
_dossier_jobs = {}
_dossier_jobs_lock = threading.Lock()


@app.route("/director/dossier", methods=["POST"])
def director_dossier():
    """
    On-demand entity/dossier creation (design doc §6) — the Director Mode
    attachment-button flow. The button's icon-swap while in Director Mode,
    and reading the adjacent text box's current contents, are frontend
    concerns; this route is the seam the frontend calls into once the
    button is pressed.

    Body: {"name": "<text currently in the adjacent text box>",
           "canon": "<current canon-input value, may be blank>"}.

    canon is required here, not optional plumbing: without it,
    create_or_update_dossier_on_demand() and find_gist_mem_ref() have no
    way to tell this canon's gist apart from a same-named entity's gist
    in a different canon (or a canon-less one), and any gist founded here
    would go untagged and become invisible to this canon's own recall.

    Deterministic at this level, no LLM judgment on whether to act: if
    the text fails is_valid_entity_name() (empty, or "anomalous input" —
    too long, contains a newline), responds asking the frontend to prompt
    for a name instead of touching the DB at all. Otherwise starts
    director_helpers.create_or_update_dossier_on_demand() — a full rebuild
    of the gist and the dossier — in a background thread and answers at
    once with {"status": "running", "job": id}; the page polls
    /director/dossier/status. A rebuild takes minutes (several large-model
    calls with thinking), longer than a browser should be left waiting on
    one request. Pressing the button again for a character whose rebuild
    is still running joins that job instead of starting a second one.
    """
    body = request.json or {}
    name = body.get("name", "")
    canon = body.get("canon", "").strip()
    if not director_helpers.is_valid_entity_name(name):
        return jsonify({
            "status": "need_name",
            "prompt": "What character would you like to form a dossier/gist for?",
        })
    key = (name.strip().upper(), canon.lower())
    with _dossier_jobs_lock:
        for jid, job in _dossier_jobs.items():
            if job["key"] == key and job["status"] == "running":
                return jsonify({"status": "running", "job": jid, "stage": job["stage"]})
        jid = uuid.uuid4().hex[:12]
        job = _dossier_jobs[jid] = {"key": key, "status": "running", "stage": "Starting dossier rebuild...", "result": None}

    def progress(msg):
        job["stage"] = msg

    def work():
        try:
            result = director_helpers.create_or_update_dossier_on_demand(name.strip(), canon, progress=progress)
        except Exception as e:
            print(f"[DOSSIER] Rebuild for {name.strip()!r} failed: {e}")
            result = {"status": "error", "reason": f"Dossier rebuild failed: {e}"}
        job["result"] = result
        job["status"] = "done"

    threading.Thread(target=work, daemon=True).start()
    return jsonify({"status": "running", "job": jid, "stage": job["stage"]})


@app.route("/director/dossier/status")
def director_dossier_status():
    """Progress of a dossier rebuild started by /director/dossier: the stage while running, then its result."""
    job = _dossier_jobs.get(request.args.get("job", ""))
    if job is None:
        return jsonify({"status": "error", "reason": "That dossier rebuild is no longer known (was Lena restarted?)."})
    if job["status"] == "running":
        return jsonify({"status": "running", "job": request.args.get("job"), "stage": job["stage"]})
    return jsonify(job["result"])


@app.route("/director/reset", methods=["POST"])
def director_reset():
    """
    Fired by the front end the moment the Ghost Writer toggle is switched
    off. This is the actual fix for the mode-switch leak: the /chat bail-out
    only clears state when the *next* message arrives, so toggling off and
    back on with no message in between left the old draft/state sitting in
    the globals untouched. This gives the toggle itself a way to clear it
    immediately, independent of whether a message ever gets sent.
    """
    reset_director_state("front-end toggle-off")
    return jsonify({"ok": True})


@app.route("/shutdown", methods=["POST"])
def shutdown_route():
    """Shutdown with optional final dream cycle."""
    do_dream = request.form.get("dream", "false").strip().lower() == "true"
    if do_dream:
        run_dream_sync()
    print("[SHUTDOWN] Exiting.")
    def _kill():
        time.sleep(0.5)
        os.kill(os.getpid(), 9)
    threading.Thread(target=_kill, daemon=True).start()
    return jsonify({"status": "shutting down"})


@app.route("/images/<filename>")
def serve_image(filename):
    """Serve stored images."""
    return send_from_directory(IMAGES_DIR, filename)


# ══════════════════════════════════════════════════════════════════════════════
# Startup
# ══════════════════════════════════════════════════════════════════════════════

def main():
    # Must happen before anything else touches MEM.db.
    ensure_wal_mode()

    load_settings()

    # Everything printed from here on is also kept in CONSOLE.txt (unless log_files = false), for runs without a console.
    consolelog.install(ROOT / "CONSOLE.txt", logs_enabled)

    config_report()

    # Create MEM.db if it is missing, add any missing tables/columns, run one-time upgrades. Once per start.
    try:
        import db_setup
        db_setup.run()
    except Exception as e:
        print(f"[WARNING] Database setup failed: {e}")

    print("[INFO] Checking memory scripts...")
    check_memory_scripts()
    if not memory_scripts_ok:
        print("[INFO] Memory system unavailable — amnesia fallback active.")

    if not llm_client.is_backend_alive():
        print("[WARN] LLM backend not responding.")
        print("[WARN] Make sure your inference server is running before typing.")

    # Starts the idle-to-dream timer.
    reset_idle_timer()

    flask_port = int(SETTINGS["flask_port"])
    url = f"http://localhost:{flask_port}"
    print(f"[INFO] Starting chat UI at {url}")
    if not os.environ.pop("LENA_RESTARTED", None):
        threading.Timer(1.5, lambda: webbrowser.open(url)).start()

    app.run(host="0.0.0.0", port=flask_port, debug=False)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n[INFO] Shutting down.")
