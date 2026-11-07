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
reddit_bridge.py — external ingress/egress adapter for a single Reddit
thread. Implements the bridge contract controller.py expects:

    def run(mode, canon, process_comment, stop_event):
        ...

controller.py owns: canon gating, the reply-worth-it gate, recall,
generation, and seeding. This file owns everything Reddit-specific:
authenticating, polling the thread, deciding *when* to post (the jitter
delay — this behavior is intentionally kept here rather than in controller,
since it's Reddit's bot-detection concerns driving it, not anything
general to how Lena reasons), and not double-processing a comment it's
already seen.

Settings live in reddit_settings.txt (sibling file, NOT controller's
settings.txt) — see that file for what to fill in.

Requires PRAW:  pip install praw
"""

import json
import random
import time
from pathlib import Path

import praw

ROOT = Path(__file__).parent.resolve()
BRIDGE_SETTINGS_FILE = ROOT / "reddit_settings.txt"
SEEN_FILE = ROOT / "reddit_seen.json"

DEFAULTS = {
    "client_id":     "",
    "client_secret": "",
    "username":      "",
    "password":      "",
    "user_agent":    "Lena/1.0",
    "submission_id": "",    # thread's Reddit ID — the part after /comments/, not the full URL
    "poll_interval": "30",  # seconds between comment-tree refreshes
    "jitter_min":    "180", # seconds — 3 min
    "jitter_max":    "600", # seconds — 10 min
}


def _load_settings():
    settings = dict(DEFAULTS)
    if not BRIDGE_SETTINGS_FILE.exists():
        print(f"[REDDIT] No {BRIDGE_SETTINGS_FILE.name} found — auth will fail.")
        return settings
    for line in BRIDGE_SETTINGS_FILE.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, val = line.partition("=")
        key, val = key.strip(), val.strip()
        if key in settings:
            settings[key] = val
    return settings


def _load_seen():
    # Dedupe state on disk, not in MEM.db — this is bridge bookkeeping
    # (which comment IDs have already been handed to controller), not
    # memory content, and doesn't belong in the memory database.
    if not SEEN_FILE.exists():
        return set()
    try:
        return set(json.loads(SEEN_FILE.read_text(encoding="utf-8")))
    except Exception:
        return set()


def _save_seen(seen):
    try:
        SEEN_FILE.write_text(json.dumps(list(seen)), encoding="utf-8")
    except Exception as e:
        print(f"[REDDIT] Could not save seen-comments state: {e}")


def _responsive_sleep(seconds, stop_event, tick=5.0):
    """time.sleep() in small increments so stop_event is checked regularly
    instead of blocking a full jitter/poll interval on shutdown."""
    waited = 0.0
    while waited < seconds and not stop_event.is_set():
        time.sleep(min(tick, seconds - waited))
        waited += tick


def run(mode, canon, process_comment, stop_event):
    """
    Entry point controller.py calls in its own background thread.

    Polls a single submission's comment tree. Every new comment gets
    handed to process_comment(text, speaker) exactly once — controller
    decides whether that's worth a reply. Only in full mode, and only if
    process_comment actually returned reply text, does this function wait
    out a jittered delay and then post. Listen mode never reaches the
    posting branch at all: process_comment always returns None in that
    mode, so there's nothing to post regardless.
    """
    settings = _load_settings()

    if not settings["client_id"] or not settings["submission_id"]:
        print("[REDDIT] Missing client_id or submission_id in reddit_settings.txt — cannot start.")
        return

    try:
        reddit = praw.Reddit(
            client_id=settings["client_id"],
            client_secret=settings["client_secret"],
            username=settings["username"],
            password=settings["password"],
            user_agent=settings["user_agent"],
        )
        submission = reddit.submission(id=settings["submission_id"])
    except Exception as e:
        print(f"[REDDIT] Auth/connection failed: {e}")
        return

    seen = _load_seen()
    poll_interval = float(settings["poll_interval"])
    jitter_min = float(settings["jitter_min"])
    jitter_max = float(settings["jitter_max"])

    print(f"[REDDIT] Bridge running — mode={mode}, canon={canon or '(blank)'}, "
          f"thread={settings['submission_id']}")

    while not stop_event.is_set():
        try:
            submission.comment_sort = "new"
            submission.comments.replace_more(limit=0)

            for comment in submission.comments.list():
                if stop_event.is_set():
                    break
                if comment.id in seen:
                    continue
                seen.add(comment.id)

                speaker = str(comment.author) if comment.author else "[deleted]"
                reply_text = process_comment(comment.body, speaker)

                if mode == "full" and reply_text:
                    delay = random.uniform(jitter_min, jitter_max)
                    print(f"[REDDIT] Replying to {speaker} in {delay:.0f}s...")
                    _responsive_sleep(delay, stop_event)
                    if not stop_event.is_set():
                        try:
                            comment.reply(reply_text)
                            print(f"[REDDIT] Posted reply to {speaker}.")
                        except Exception as e:
                            print(f"[REDDIT] Failed to post reply: {e}")

            _save_seen(seen)
        except Exception as e:
            print(f"[REDDIT] Poll error: {e}")

        _responsive_sleep(poll_interval, stop_event)

    print("[REDDIT] Bridge stopped.")
