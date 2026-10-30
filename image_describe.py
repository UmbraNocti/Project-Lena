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


"""
image_describe.py - Supplemental text descriptions for image memories.

An image is normally findable only by the keywords of the conversation
around it. When a vision route exists (llm_client.vision_mode()), this
module asks the model to look at the image once, stores a short
description in Images.Description, and links extra keywords drawn from
that description to the same image row. Nothing about the existing
keyword-only image memory changes: images without a description keep
working exactly as before, and a blind setup simply never calls this.

Callers:
  seed.py  - right after an uploaded image is stored (recalled=False)
  dream.py - Pass 6 backfill for images still lacking a description
             (recalled=True: old images fall under specialist_send_recalled)

A description is only ever stored if the image was really attached to the
call (meta["images_sent"] > 0) — a blind model must never invent one.
"""

import sqlite3
from pathlib import Path

import llm_client
from keyword_utils import handle_keywords, ensure_keyword, known_names

ROOT = Path(__file__).parent.resolve()
DB_PATH = ROOT / "MEM.db"
IMAGES_DIR = ROOT / "images"
DESCRIBE_PROMPT_FILE = ROOT / "prompts" / "image_description_prompt.txt"


def _load(path):
    return path.read_text(encoding="utf-8").strip() if path.exists() else ""


def _get_connection():
    conn = sqlite3.connect(DB_PATH)
    conn.execute("PRAGMA journal_mode=WAL")
    return conn


def describe_and_link(image_id, filename, context_text="", recalled=False):
    """
    Generates and stores a description for one image, then links new
    keywords from it. Returns True only if a description was stored.
    Never raises — every failure just leaves the image as it was.
    """
    if llm_client.vision_mode() is None:
        return False
    path = IMAGES_DIR / filename
    prompt = _load(DESCRIBE_PROMPT_FILE)
    if not path.is_file() or not prompt:
        return False
    # Never spend a call unless the image will really be attached (e.g. the
    # specialist route with specialist_send_recalled = false strips old
    # images, and a blind "description" would just be invented).
    if not llm_client.filter_images([{"path": path, "recalled": recalled}]):
        print(f"[IMAGE_DESCRIBE] {filename} can't be sent on the current vision route, skipping.")
        return False

    user_text = "Describe this image."
    if context_text:
        user_text += f"\n\nConversation it was shared in, for context:\n{context_text}"

    try:
        description, meta = llm_client.get_completion(
            system_prompt=prompt,
            user_text=user_text,
            max_tokens=250,
            temperature=0.2,
            timeout=120,
            images=[{"path": path, "recalled": recalled}],
            return_meta=True,
        )
    except Exception as e:
        print(f"[IMAGE_DESCRIBE] Description call failed for {filename}: {e}")
        return False
    description = (description or "").strip()
    if not description or meta.get("images_sent", 0) < 1:
        print(f"[IMAGE_DESCRIBE] Image not actually seen for {filename}, nothing stored.")
        return False

    conn = None
    try:
        conn = _get_connection()
        keywords = handle_keywords(description, known_names(conn))     # the description's own handles, no LLM call
        conn.execute("UPDATE Images SET Description = ? WHERE ImageID = ?", (description, image_id))
        existing = {r[0] for r in conn.execute("SELECT KeyID FROM Nodes WHERE ImageID = ?", (image_id,))}
        added = []
        for kw in keywords:
            kid = ensure_keyword(conn, kw)
            if kid not in existing:
                conn.execute("INSERT INTO Nodes (ImageID, KeyID) VALUES (?, ?)", (image_id, kid))
                existing.add(kid)
                added.append(kw)
        conn.commit()
        print(f"[IMAGE_DESCRIBE] ImageID={image_id} described; new keywords: {added}")
        return True
    except Exception as e:
        print(f"[IMAGE_DESCRIBE] DB write failed for ImageID={image_id}: {e}")
        return False
    finally:
        if conn is not None:
            conn.close()


def backfill(limit=10, stop_event=None):
    """Describes existing images that have no description yet. Makes at
    most `limit` description attempts per call (successful or not), taken
    in random order so one image that keeps failing can't starve the rest.
    Returns the number described. Skips images whose file is gone or that
    the current vision route can't send."""
    if llm_client.vision_mode() is None:
        return 0
    conn = _get_connection()
    try:
        rows = conn.execute(
            "SELECT ImageID, Filename FROM Images WHERE Description IS NULL OR Description = '' "
            "ORDER BY RANDOM()"
        ).fetchall()
    finally:
        conn.close()

    done = attempts = 0
    for image_id, filename in rows:
        if attempts >= limit or (stop_event is not None and stop_event.is_set()):
            break
        path = IMAGES_DIR / filename
        if not path.is_file() or not llm_client.filter_images([{"path": path, "recalled": True}]):
            continue
        attempts += 1
        if describe_and_link(image_id, filename, recalled=True):
            done += 1
    return done
