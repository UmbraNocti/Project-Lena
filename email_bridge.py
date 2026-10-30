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
email_bridge.py — external ingress/egress adapter for a single mailbox.
Implements the same bridge contract as reddit_bridge.py:

    def run(mode, canon, process_comment, stop_event):
        ...

controller.py owns canon gating, the reply gate, recall, generation, and
seeding — identical to every other bridge. This file owns everything
email-specific: connecting via IMAP/SMTP, deciding what counts as a new
message, threading a reply correctly, appending the disclosure signature,
and the per-sender loop-protection lockout.

Settings live in email_settings.txt (sibling file, NOT controller's
settings.txt). Minimum required there is an email address and password —
this script auto-detects IMAP/SMTP host and port for Gmail, Yahoo,
Outlook/Hotmail/Live, and iCloud from the address's domain. Anything else
falls back to a best-guess imap.<domain>/smtp.<domain> and prints a
warning; fill in the optional override fields for a custom/self-hosted
server.

No third-party dependencies — imaplib, smtplib, and email are all stdlib.
"""

import json
import random
import re
import smtplib
import time
from datetime import datetime, timedelta
from email.header import decode_header
from email.message import EmailMessage
from email.utils import parseaddr, make_msgid
from pathlib import Path
import email as email_lib
import imaplib

ROOT = Path(__file__).parent.resolve()
BRIDGE_SETTINGS_FILE = ROOT / "email_settings.txt"
SEEN_FILE = ROOT / "email_seen.json"
LOCKOUT_FILE = ROOT / "email_lockouts.json"

DEFAULTS = {
    "email":         "",
    "password":      "",
    "imap_host":     "",
    "imap_port":     "",
    "smtp_host":     "",
    "smtp_port":     "",
    "use_ssl":       "",
    "poll_interval": "60",
    "jitter_min":    "0",
    "jitter_max":    "0",
    "signature":     "Kind regards,\\nLena herself \u2014 not a human contributor.",
    "max_replies_per_sender_per_hour": "4",
    "strip_quoted_replies": "true",
}

# Known providers — covers the common cases so most people only ever need
# to fill in "email" and "password". Domain match is by suffix (so
# "mail.yahoo.com"-style aliases still resolve via the bare domain key).
PROVIDER_PRESETS = {
    "gmail.com":       {"imap_host": "imap.gmail.com",        "imap_port": 993, "smtp_host": "smtp.gmail.com",        "smtp_port": 587, "use_ssl": False},
    "googlemail.com":  {"imap_host": "imap.gmail.com",        "imap_port": 993, "smtp_host": "smtp.gmail.com",        "smtp_port": 587, "use_ssl": False},
    "yahoo.com":       {"imap_host": "imap.mail.yahoo.com",   "imap_port": 993, "smtp_host": "smtp.mail.yahoo.com",   "smtp_port": 587, "use_ssl": False},
    "outlook.com":     {"imap_host": "outlook.office365.com", "imap_port": 993, "smtp_host": "smtp.office365.com",    "smtp_port": 587, "use_ssl": False},
    "hotmail.com":     {"imap_host": "outlook.office365.com", "imap_port": 993, "smtp_host": "smtp.office365.com",    "smtp_port": 587, "use_ssl": False},
    "live.com":        {"imap_host": "outlook.office365.com", "imap_port": 993, "smtp_host": "smtp.office365.com",    "smtp_port": 587, "use_ssl": False},
    "icloud.com":      {"imap_host": "imap.mail.me.com",      "imap_port": 993, "smtp_host": "smtp.mail.me.com",      "smtp_port": 587, "use_ssl": False},
    "me.com":          {"imap_host": "imap.mail.me.com",      "imap_port": 993, "smtp_host": "smtp.mail.me.com",      "smtp_port": 587, "use_ssl": False},
}


# ── Settings ──────────────────────────────────────────────────────────────────

def _load_settings():
    settings = dict(DEFAULTS)
    if not BRIDGE_SETTINGS_FILE.exists():
        print(f"[EMAIL] No {BRIDGE_SETTINGS_FILE.name} found — auth will fail.")
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


def _resolve_server_settings(address, settings):
    """Merge provider preset (by domain) with any non-blank manual
    overrides from email_settings.txt. Manual overrides always win."""
    domain = address.split("@")[-1].lower() if "@" in address else ""
    preset = PROVIDER_PRESETS.get(domain)

    if preset is None:
        print(f"[EMAIL] Unrecognized domain '{domain}' — guessing standard "
              f"imap./smtp. hostnames. Fill in imap_host/smtp_host in "
              f"{BRIDGE_SETTINGS_FILE.name} if this doesn't connect.")
        preset = {"imap_host": f"imap.{domain}", "imap_port": 993,
                  "smtp_host": f"smtp.{domain}", "smtp_port": 587, "use_ssl": False}

    resolved = dict(preset)
    if settings["imap_host"]:
        resolved["imap_host"] = settings["imap_host"]
    if settings["imap_port"]:
        resolved["imap_port"] = int(settings["imap_port"])
    if settings["smtp_host"]:
        resolved["smtp_host"] = settings["smtp_host"]
    if settings["smtp_port"]:
        resolved["smtp_port"] = int(settings["smtp_port"])
    if settings["use_ssl"]:
        resolved["use_ssl"] = settings["use_ssl"].strip().lower() == "true"
    return resolved


# ── Dedupe state (which Message-IDs have already been handed to controller) ──

def _load_seen():
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
        print(f"[EMAIL] Could not save seen-messages state: {e}")


# ── Loop protection: per-sender hourly reply lockout ─────────────────────────
# Not enforced by the mail provider — Gmail-style sending caps are daily
# volume limits, not same-sender-per-hour, so a fast reply loop (an
# autoresponder, a bounce, another AI on the other end) would sail right
# through those. This is the actual circuit breaker. Human review = open
# email_lockouts.json and clear (or just delete) the sender's entry.

def _load_lockout_log():
    if not LOCKOUT_FILE.exists():
        return {}
    try:
        return json.loads(LOCKOUT_FILE.read_text(encoding="utf-8"))
    except Exception:
        return {}


def _save_lockout_log(log):
    try:
        LOCKOUT_FILE.write_text(json.dumps(log), encoding="utf-8")
    except Exception as e:
        print(f"[EMAIL] Could not save lockout log: {e}")


def _sender_locked_out(sender_addr, log, max_per_hour):
    cutoff = datetime.utcnow() - timedelta(hours=1)
    timestamps = [datetime.fromisoformat(t) for t in log.get(sender_addr, [])]
    recent = [t for t in timestamps if t > cutoff]
    log[sender_addr] = [t.isoformat() for t in recent]
    return len(recent) >= max_per_hour


def _record_reply(sender_addr, log):
    log.setdefault(sender_addr, []).append(datetime.utcnow().isoformat())


# ── Body extraction ───────────────────────────────────────────────────────────

_TAG_RE = re.compile(r'<[^>]+>')

def _strip_html(html_text):
    # Reliable signal first: Gmail, Outlook, Apple Mail, and Yahoo all wrap
    # quoted history in a <blockquote> when composing HTML mail. Truncating
    # there — rather than trying to balance nested tags with a regex —
    # removes most quoted chains before any plain-text heuristic even runs.
    match = re.search(r'(?is)<blockquote', html_text)
    if match:
        html_text = html_text[:match.start()]

    text = re.sub(r'(?is)<(script|style).*?>.*?(</\1>)', '', html_text)
    # Turn block-level boundaries into line breaks before stripping tags,
    # so a line-based quote marker (e.g. "On ... wrote:") inside HTML-
    # authored mail still lands on its own line for _strip_quoted_reply.
    text = re.sub(r'(?i)<(br\s*/?|/p|/div|/li)>', '\n', text)
    text = _TAG_RE.sub(' ', text)
    text = re.sub(r'[ \t]+', ' ', text)
    text = re.sub(r'\n[ \t]*\n+', '\n', text)
    return text.strip()


# Lines that mark the start of a quoted reply chain, in the plain-text
# conventions Gmail, Outlook, Apple Mail, and Thunderbird all produce.
_QUOTE_BLOCK_PATTERNS = [
    re.compile(r'^\s*>'),                                          # standard '>' quoted line
    re.compile(r'^\s*On\s.{0,120}\swrote:\s*$', re.IGNORECASE),     # Gmail / Apple Mail / Thunderbird
    re.compile(r'^\s*-{2,}\s*Original Message\s*-{2,}\s*$', re.IGNORECASE),  # Outlook
    re.compile(r'^\s*From:\s*.+@.+$', re.IGNORECASE),               # Outlook plain-text header block
]


def _strip_quoted_reply(text):
    """
    Trims a reply down to just the new content above the quoted history.
    Heuristic, not a guarantee — if it ever cuts off real content (someone's
    own message legitimately starts a line with ">", or contains the
    literal phrase "wrote:"), disable via strip_quoted_replies = false in
    email_settings.txt. If the very first line matches (an empty reply
    with nothing but quoted content), the full text is kept rather than
    returning nothing.
    """
    lines = text.splitlines()
    for i, line in enumerate(lines):
        for pattern in _QUOTE_BLOCK_PATTERNS:
            if pattern.match(line):
                trimmed = "\n".join(lines[:i]).strip()
                return trimmed if trimmed else text.strip()
    return text.strip()


def _decode_header_value(raw):
    if not raw:
        return ""
    parts = decode_header(raw)
    decoded = ""
    for text, enc in parts:
        if isinstance(text, bytes):
            decoded += text.decode(enc or "utf-8", errors="replace")
        else:
            decoded += text
    return decoded


def _extract_body(msg, strip_quotes=True):
    plain, html = None, None
    if msg.is_multipart():
        for part in msg.walk():
            ctype = part.get_content_type()
            disp = str(part.get("Content-Disposition") or "")
            if "attachment" in disp:
                continue
            if ctype == "text/plain" and plain is None:
                plain = part.get_payload(decode=True)
                plain_charset = part.get_content_charset() or "utf-8"
                plain = plain.decode(plain_charset, errors="replace")
            elif ctype == "text/html" and html is None:
                html = part.get_payload(decode=True)
                html_charset = part.get_content_charset() or "utf-8"
                html = html.decode(html_charset, errors="replace")
    else:
        payload = msg.get_payload(decode=True)
        charset = msg.get_content_charset() or "utf-8"
        text = payload.decode(charset, errors="replace") if payload else ""
        if msg.get_content_type() == "text/html":
            html = text
        else:
            plain = text

    if plain:
        body = plain.strip()
    elif html:
        body = _strip_html(html)
    else:
        return "[Empty message]"

    return _strip_quoted_reply(body) if strip_quotes else body


def _responsive_sleep(seconds, stop_event, tick=5.0):
    waited = 0.0
    while waited < seconds and not stop_event.is_set():
        time.sleep(min(tick, seconds - waited))
        waited += tick


# ── Sending ───────────────────────────────────────────────────────────────────

def _send_reply(server, settings, original_msg, sender_addr, reply_text):
    signature = settings["signature"].replace("\\n", "\n")
    body = f"{reply_text}\n\n--\n{signature}"

    reply = EmailMessage()
    reply["Message-ID"] = make_msgid()
    reply["From"] = settings["email"]
    reply["To"] = sender_addr

    subject = _decode_header_value(original_msg.get("Subject", ""))
    reply["Subject"] = subject if subject.lower().startswith("re:") else f"Re: {subject}"

    orig_id = original_msg.get("Message-ID", "")
    if orig_id:
        reply["In-Reply-To"] = orig_id
        references = original_msg.get("References", "")
        reply["References"] = f"{references} {orig_id}".strip()

    reply.set_content(body)

    smtp_host, smtp_port = server["smtp_host"], server["smtp_port"]
    if server["use_ssl"]:
        conn = smtplib.SMTP_SSL(smtp_host, smtp_port, timeout=30)
    else:
        conn = smtplib.SMTP(smtp_host, smtp_port, timeout=30)
        conn.starttls()
    try:
        conn.login(settings["email"], settings["password"])
        conn.send_message(reply)
    finally:
        conn.quit()


# ── Main loop ─────────────────────────────────────────────────────────────────

def run(mode, canon, process_comment, stop_event):
    """
    Entry point controller.py calls in its own background thread.

    Polls INBOX for unseen messages. Every new message gets handed to
    process_comment(body, sender) exactly once — controller decides
    whether that's worth a reply, same as every other bridge. Only in
    full mode, and only if process_comment actually returned reply text,
    does this function check the per-sender lockout, wait out any
    configured jitter, and send.

    A human replying manually through the account's normal webmail/app
    never touches this script at all, so the disclosure signature only
    ever appears on messages she actually sends through here.
    """
    settings = _load_settings()

    if not settings["email"] or not settings["password"]:
        print("[EMAIL] Missing email or password in email_settings.txt — cannot start.")
        return

    server = _resolve_server_settings(settings["email"], settings)
    seen = _load_seen()
    lockout_log = _load_lockout_log()

    poll_interval = float(settings["poll_interval"])
    jitter_min = float(settings["jitter_min"])
    jitter_max = float(settings["jitter_max"])
    max_per_hour = int(settings["max_replies_per_sender_per_hour"])
    strip_quotes = settings["strip_quoted_replies"].strip().lower() != "false"

    print(f"[EMAIL] Bridge running \u2014 mode={mode}, canon={canon or '(blank)'}, "
          f"account={settings['email']}")

    while not stop_event.is_set():
        try:
            imap = imaplib.IMAP4_SSL(server["imap_host"], server["imap_port"])
            imap.login(settings["email"], settings["password"])
            imap.select("INBOX")

            status, data = imap.search(None, "UNSEEN")
            if status == "OK":
                for num in data[0].split():
                    if stop_event.is_set():
                        break
                    status, msg_data = imap.fetch(num, "(RFC822)")
                    if status != "OK" or not msg_data or not msg_data[0]:
                        continue

                    raw = msg_data[0][1]
                    msg = email_lib.message_from_bytes(raw)
                    msg_id = msg.get("Message-ID", f"uid-{num.decode()}")

                    if msg_id in seen:
                        imap.store(num, "+FLAGS", "\\Seen")
                        continue
                    seen.add(msg_id)

                    display_name, addr = parseaddr(msg.get("From", ""))
                    speaker = display_name or addr
                    body = _extract_body(msg, strip_quotes=strip_quotes)

                    reply_text = process_comment(body, speaker)
                    imap.store(num, "+FLAGS", "\\Seen")

                    if mode == "full" and reply_text and addr:
                        if _sender_locked_out(addr, lockout_log, max_per_hour):
                            print(f"[EMAIL] {addr} is locked out (>{max_per_hour} replies/hr) "
                                  f"— pending human review in {LOCKOUT_FILE.name}. Not replying.")
                        else:
                            if jitter_max > 0:
                                delay = random.uniform(jitter_min, jitter_max)
                                _responsive_sleep(delay, stop_event)
                            if not stop_event.is_set():
                                try:
                                    _send_reply(server, settings, msg, addr, reply_text)
                                    _record_reply(addr, lockout_log)
                                    print(f"[EMAIL] Replied to {speaker} <{addr}>.")
                                except Exception as e:
                                    print(f"[EMAIL] Failed to send reply: {e}")

            imap.logout()
            _save_seen(seen)
            _save_lockout_log(lockout_log)
        except Exception as e:
            print(f"[EMAIL] Poll error: {e}")

        _responsive_sleep(poll_interval, stop_event)

    print("[EMAIL] Bridge stopped.")
