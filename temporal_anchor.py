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
temporal_anchor.py - Resolves a temporal reference in the user's message
Called by controller.py before recall() to determine if the user is
referencing a specific past time (e.g. "yesterday", "last week") and if
so, resolves it to an actual date. That date becomes the anchor_time
passed into recall()'s scoring function, replacing "now" as the
reference point for memory decay calculations.

Fails gracefully — any parse failure or ambiguous result falls back
to "SAME", meaning recall uses the current time as usual.
"""

import re
import time
import llm_client
from pathlib import Path
from datetime import datetime

# ── Paths ─────────────────────────────────────────────────────────────────────
ROOT = Path(__file__).parent.resolve()
TEMPORAL_ANCHOR_PROMPT_FILE = ROOT / "prompts" / "temporal_anchor_prompt.txt"

DATE_PATTERN = re.compile(r"^\d{4}-\d{2}-\d{2}$")

# Deterministic prefilter: if the message contains none of these cues there
# is nothing to resolve, so the LLM call is skipped and "now" is returned.
# Deliberately broad — a false positive only costs the call we'd have made
# anyway; a false negative silently loses a temporal anchor.
TEMPORAL_CUES = re.compile(
    r"\b("
    r"yesterday|tomorrow|tonight|today|ago|earlier|later|before|after|since|until|"
    r"previous(?:ly)?|prior|recent(?:ly)?|lately|last|next|past|former(?:ly)?|"
    r"once|then|when|while|used to|back then|the other day|"
    r"day|days|week|weeks|weekend|month|months|year|years|decade|decades|"
    r"morning|afternoon|evening|night|midnight|noon|dawn|dusk|"
    r"monday|tuesday|wednesday|thursday|friday|saturday|sunday|"
    r"january|february|march|april|may|june|july|august|september|october|"
    r"november|december|jan|feb|mar|apr|jun|jul|aug|sep|sept|oct|nov|dec|"
    r"spring|summer|autumn|fall|winter|christmas|thanksgiving|halloween|easter|"
    r"birthday|holiday|childhood|"
    r"\d{4}|\d{1,2}[/-]\d{1,2}|\d{1,2}(?:st|nd|rd|th)"
    r")\b",
    re.IGNORECASE,
)


# The previous message only counts when it carries an unmistakable time reference. The broad list above
# would fire on almost any earlier turn ("when", "day", "then"...), turning the skip into an LLM call on
# every message.
STRONG_TEMPORAL_CUES = re.compile(
    r"\b("
    r"yesterday|tomorrow|tonight|ago|the other day|back then|used to|"
    r"last (?:night|week|weekend|month|year|spring|summer|fall|autumn|winter|time)|"
    r"next (?:week|weekend|month|year|spring|summer|fall|autumn|winter)|"
    r"(?:this|that) (?:morning|afternoon|evening|week|weekend|month|year)|"
    r"monday|tuesday|wednesday|thursday|friday|saturday|sunday|"
    r"january|february|march|april|june|july|august|september|october|november|december|"
    r"christmas|thanksgiving|halloween|easter|birthday|"
    r"(?:19|20)\d{2}|\d{1,2}[/-]\d{1,2}"
    r")\b",
    re.IGNORECASE,
)


def load_prompt(path):
    if not path.exists():
        print(f"[TEMPORAL][WARNING] Prompt file not found: {path}")
        return ""
    return path.read_text(encoding="utf-8").strip()


def resolve_anchor(user_message: str, previous: str = "") -> float:
    """
    Entry point called by controller before recall().
    Returns a unix epoch timestamp to use as the scoring anchor (recall's recency decay is measured from it).
    Returns current time if no temporal shift is detected or on any failure.

    previous: the user's previous message. A follow-up such as "what did we eat there?" after
    "remember our trip last spring?" carries its time cue in the previous turn. The current message is
    checked against the broad cue list; the previous one only against unmistakable references
    (STRONG_TEMPORAL_CUES). When the LLM is asked, it sees the previous message as context.
    """
    now = time.time()

    if not TEMPORAL_CUES.search(user_message or "") and not STRONG_TEMPORAL_CUES.search(previous or ""):
        print("[TEMPORAL] No temporal cue in message or previous message, using now (no LLM call).")
        return now

    prompt = load_prompt(TEMPORAL_ANCHOR_PROMPT_FILE)
    if not prompt:
        print("[TEMPORAL][WARNING] temporal_anchor_prompt.txt missing, using now.")
        return now

    now_dt = datetime.fromtimestamp(now)
    date_context = (
        f"Year:{now_dt.year}\n"
        f"Month:{now_dt.month}\n"
        f"Day:{now_dt.day}\n"
        f"Time:{now_dt.strftime('%H%M')}\n\n"
        + (f"Previous message (context only):\n{previous[:1500]}\n\n" if previous else "")
        + f"User message: {user_message}"
    )

    try:
        raw = llm_client.get_completion(
            system_prompt=prompt,
            user_text=date_context,
            max_tokens=80,
            temperature=0.1,
            timeout=30,
            preferred_tier="small",
        )
    except Exception as e:
        print(f"[TEMPORAL][WARNING] LLM call failed: {e}")
        return now

    if not raw:
        return now

    cleaned = raw.strip().upper()

    if cleaned == "SAME":
        print("[TEMPORAL] No temporal shift detected, using now.")
        return now

    # Try to parse as YYYY-MM-DD
    match = DATE_PATTERN.match(raw.strip())
    if not match:
        print(f"[TEMPORAL] Unparseable response '{raw}', falling back to now.")
        return now

    try:
        anchor_dt = datetime.strptime(raw.strip(), "%Y-%m-%d")
        anchor_epoch = anchor_dt.timestamp()
        print(f"[TEMPORAL] Anchor resolved: {raw.strip()}")
        return anchor_epoch
    except Exception as e:
        print(f"[TEMPORAL] Failed to parse resolved date '{raw}': {e}, using now.")
        return now


# ══════════════════════════════════════════════════════════════════════════════
# Console test entry point
# ══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    import sys
    if len(sys.argv) >= 2:
        msg = sys.argv[1]
        anchor = resolve_anchor(msg)
        print(f"\nResolved anchor: {datetime.fromtimestamp(anchor)}")
    else:
        print("Usage: python3 temporal_anchor.py \"user message\"")
