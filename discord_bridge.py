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
discord_bridge.py — external ingress/egress adapter for a single Discord
channel. Implements the same bridge contract as reddit_bridge.py and
email_bridge.py:

    def run(mode, canon, process_comment, stop_event):
        ...

controller.py owns canon gating, the reply gate, recall, generation, and
seeding — identical to every other bridge. This file owns everything
Discord-specific: authenticating as a bot user, polling one channel for
new messages, not double-processing a message it's already seen, and
posting replies threaded to the original message.

Uses Discord's plain REST API directly via `requests` — no discord.py
dependency. Requires a registered Discord Application with a bot user
(discord.com/developers/applications); this must be a real bot account,
never a normal user token — automating a regular Discord account
("self-botting") is a Terms of Service violation and a real ban risk,
unlike the read-only scraping this sidesteps on Reddit.

Settings live in discord_settings.txt (sibling file, NOT controller's
settings.txt).
"""

import json
import random
import time
from pathlib import Path

import requests

ROOT = Path(__file__).parent.resolve()
BRIDGE_SETTINGS_FILE = ROOT / "discord_settings.txt"
CURSOR_FILE = ROOT / "discord_cursor.json"

API_BASE = "https://discord.com/api/v10"

DEFAULTS = {
    "bot_token":     "",
    "channel_id":    "",
    "poll_interval": "15",
    "jitter_min":    "0",   # Discord bots are expected to be bots — no
    "jitter_max":    "0",   # bot-detection concern like Reddit, so this
                            # defaults to off. Set nonzero if you still
                            # want replies to feel less instant.
    "ignore_bots":   "true",  # skip messages from other bot accounts (and
                              # this bot's own messages) — cheapest possible
                              # loop-protection: bot-to-bot ping-pong never
                              # starts in the first place.
}


# ── Settings ──────────────────────────────────────────────────────────────────

def _load_settings():
    settings = dict(DEFAULTS)
    if not BRIDGE_SETTINGS_FILE.exists():
        print(f"[DISCORD] No {BRIDGE_SETTINGS_FILE.name} found — auth will fail.")
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


# ── Cursor (last message ID processed) ────────────────────────────────────────
# Discord message IDs are snowflakes — monotonically increasing, so a
# single "last seen ID" plus the API's own "after" pagination parameter is
# enough to avoid reprocessing. No separate seen-set needed, unlike Reddit
# (whose comment-tree API has no equivalent "since" cursor).

def _load_cursor():
    if not CURSOR_FILE.exists():
        return None
    try:
        return json.loads(CURSOR_FILE.read_text(encoding="utf-8")).get("last_id")
    except Exception:
        return None


def _save_cursor(last_id):
    try:
        CURSOR_FILE.write_text(json.dumps({"last_id": last_id}), encoding="utf-8")
    except Exception as e:
        print(f"[DISCORD] Could not save cursor: {e}")


def _responsive_sleep(seconds, stop_event, tick=5.0):
    waited = 0.0
    while waited < seconds and not stop_event.is_set():
        time.sleep(min(tick, seconds - waited))
        waited += tick


# ── Discord REST calls ────────────────────────────────────────────────────────

def _headers(token):
    return {
        "Authorization": f"Bot {token}",
        "User-Agent": "DiscordBot (https://github.com/UmbraNocti/Project-Lena, 1.0)",
        "Content-Type": "application/json",
    }


def _get_bot_user_id(token):
    r = requests.get(f"{API_BASE}/users/@me", headers=_headers(token), timeout=15)
    r.raise_for_status()
    return r.json()["id"]


def _fetch_new_messages(token, channel_id, after_id):
    params = {"limit": 50}
    if after_id:
        params["after"] = after_id
    r = requests.get(
        f"{API_BASE}/channels/{channel_id}/messages",
        headers=_headers(token), params=params, timeout=15,
    )
    if r.status_code == 429:
        retry_after = r.json().get("retry_after", 5)
        print(f"[DISCORD] Rate limited, waiting {retry_after}s.")
        time.sleep(retry_after)
        return []
    r.raise_for_status()
    # Discord returns newest-first regardless of the "after" boundary used —
    # reverse so messages are handled oldest-to-newest, same order they
    # actually happened in.
    return list(reversed(r.json()))


def _send_reply(token, channel_id, reply_text, reply_to_message_id):
    payload = {
        "content": reply_text,
        "message_reference": {"message_id": reply_to_message_id},
    }
    r = requests.post(
        f"{API_BASE}/channels/{channel_id}/messages",
        headers=_headers(token), json=payload, timeout=15,
    )
    if r.status_code == 429:
        retry_after = r.json().get("retry_after", 5)
        print(f"[DISCORD] Rate limited on send, waiting {retry_after}s.")
        time.sleep(retry_after)
        r = requests.post(
            f"{API_BASE}/channels/{channel_id}/messages",
            headers=_headers(token), json=payload, timeout=15,
        )
    r.raise_for_status()


# ── Main loop ─────────────────────────────────────────────────────────────────

def run(mode, canon, process_comment, stop_event):
    """
    Entry point controller.py calls in its own background thread.

    Polls one channel for new messages. Every new message (that isn't from
    a bot, if ignore_bots is on) gets handed to process_comment(content,
    speaker) exactly once — controller decides whether that's worth a
    reply, same as every other bridge. Only in full mode, and only if
    process_comment actually returned reply text, does this function wait
    out any configured jitter and post — threaded as a reply to the
    original message via message_reference.
    """
    settings = _load_settings()

    if not settings["bot_token"] or not settings["channel_id"]:
        print("[DISCORD] Missing bot_token or channel_id in discord_settings.txt — cannot start.")
        return

    token = settings["bot_token"]
    channel_id = settings["channel_id"]
    ignore_bots = settings["ignore_bots"].strip().lower() != "false"

    try:
        bot_user_id = _get_bot_user_id(token)
    except Exception as e:
        print(f"[DISCORD] Auth failed — check bot_token: {e}")
        return

    last_id = _load_cursor()
    poll_interval = float(settings["poll_interval"])
    jitter_min = float(settings["jitter_min"])
    jitter_max = float(settings["jitter_max"])

    print(f"[DISCORD] Bridge running — mode={mode}, canon={canon or '(blank)'}, "
          f"channel={channel_id}")

    while not stop_event.is_set():
        try:
            messages = _fetch_new_messages(token, channel_id, last_id)

            for msg in messages:
                if stop_event.is_set():
                    break

                last_id = msg["id"]
                author = msg.get("author", {})

                if author.get("id") == bot_user_id:
                    continue
                if ignore_bots and author.get("bot"):
                    continue

                content = msg.get("content", "").strip()
                if not content:
                    continue

                speaker = author.get("global_name") or author.get("username", "Unknown")
                reply_text = process_comment(content, speaker)

                if mode == "full" and reply_text:
                    if jitter_max > 0:
                        delay = random.uniform(jitter_min, jitter_max)
                        _responsive_sleep(delay, stop_event)
                    if not stop_event.is_set():
                        try:
                            _send_reply(token, channel_id, reply_text, msg["id"])
                            print(f"[DISCORD] Replied to {speaker}.")
                        except Exception as e:
                            print(f"[DISCORD] Failed to send reply: {e}")

            if last_id:
                _save_cursor(last_id)
        except Exception as e:
            print(f"[DISCORD] Poll error: {e}")

        _responsive_sleep(poll_interval, stop_event)

    print("[DISCORD] Bridge stopped.")
