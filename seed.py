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
seed.py - Memory seeding script
Called by controller.py in a background thread after each exchange.
Chunks the user text and the AI text, gives every chunk its deterministic handles (its own content
words, stop words dropped; no LLM call), stores it, and embeds it when an embedding backend is set.
Every memory is stored as bare text and records who said it (Speakers: the person, a document, or Lena herself)
and which canon it belongs to (Canons, NULL = open memory). Lena's own lines also record whom she was answering
(PartnerID). The "X said:" labels are never stored; memory_index.render adds them when a memory is shown.
Director Mode commits are seeded with no speaker: the story is just the story.
"""

import sqlite3
import time
import re
from pathlib import Path

import memory_index
from keyword_utils import handle_keywords, ensure_keyword, known_names

# ── Paths ─────────────────────────────────────────────────────────────────────
ROOT = Path(__file__).parent.resolve()
DB_PATH = ROOT / "MEM.db"


# ══════════════════════════════════════════════════════════════════════════════
# Database operations
# ══════════════════════════════════════════════════════════════════════════════

def get_connection():
    conn = sqlite3.connect(DB_PATH)
    conn.execute("PRAGMA journal_mode=WAL")
    return conn


def insert_memory(conn, memory_text, origin_time, speaker_id=None, cid=None, partner_id=None):
    cursor = conn.execute(
        "INSERT INTO Memory (OriginTime, Memory, RecallCount, SpeakerID, PartnerID, CanonID) VALUES (?, ?, ?, ?, ?, ?)",
        (origin_time, memory_text, 1, speaker_id, partner_id, cid)
    )
    conn.commit()
    return cursor.lastrowid


def link_memory_to_keywords(conn, mem_ref, keyword_ids):
    for key_id in keyword_ids:
        conn.execute(
            "INSERT INTO Nodes (MemID, KeyID) VALUES (?, ?)",
            (mem_ref, key_id)
        )
    conn.commit()


def store_memory(memory_text, keywords, speaker_id=None, embed=True, cid=None, partner_id=None):
    """
    memory_text: the bare text, no speaker label.
    speaker_id: who said or wrote it (None when nobody did, e.g. Director Mode's story).
    partner_id: for Lena's own lines, the person she was answering.
    cid: the canon it belongs to (None = open memory). Inside a canon a chunk is kept even with no handles
    of its own (the vector lane can still find it). embed: embed the new memory now (skipped silently when
    no embedder is configured).
    """
    if not keywords and cid is None:
        return

    origin_time = int(time.time())
    conn = None

    try:
        conn = get_connection()

        keyword_ids = [ensure_keyword(conn, kw) for kw in keywords]

        mem_ref = insert_memory(conn, memory_text, origin_time, speaker_id, cid, partner_id)
        link_memory_to_keywords(conn, mem_ref, keyword_ids)
        print(f"[SEED] Stored memory (MemRef={mem_ref}) with keywords: {keywords}")
        if embed:
            memory_index.embed_memory(conn, mem_ref)

    except Exception as e:
        print(f"[SEED][ERROR] Database write failed: {e}")
    finally:
        if conn is not None:
            conn.close()


def store_image_memory(filename, keywords, cid=None):
    """
    Inserts a row into Images and links it to keywords via Nodes.ImageID.
    MemID is left NULL on these Nodes rows — a Nodes row is either a
    Memory-Keyword link or an Image-Keyword link, never both.
    """
    if not keywords:
        return None

    origin_time = int(time.time())
    conn = None
    try:
        conn = get_connection()
        cursor = conn.execute(
            "INSERT INTO Images (Filename, DateCreated, CanonID) VALUES (?, ?, ?)",
            (filename, origin_time, cid)
        )
        conn.commit()
        image_id = cursor.lastrowid

        keyword_ids = [ensure_keyword(conn, kw) for kw in keywords]
        for kid in keyword_ids:
            conn.execute(
                "INSERT INTO Nodes (ImageID, KeyID) VALUES (?, ?)",
                (image_id, kid)
            )
        conn.commit()
        print(f"[SEED] Stored image (ImageID={image_id}, file={filename}) "
              f"with keywords: {keywords}")
        return image_id
    except Exception as e:
        print(f"[SEED][ERROR] Image DB write failed: {e}")
        return None
    finally:
        if conn is not None:
            conn.close()

# ══════════════════════════════════════════════════════════════════════════════
# Utility Functions
# ══════════════════════════════════════════════════════════════════════════════
def _split_on_hard_cap(text, hard_cap):
    """
    Force-split text into pieces no longer than hard_cap characters.
    Cuts right after the last sentence-ending punctuation found within
    the cap window; falls back to the nearest whitespace if there's no
    punctuation in range; falls back to a hard slice only if there's
    neither — i.e. one continuous unbroken string.
    """
    pieces = []
    remaining = text
    while len(remaining) > hard_cap:
        window = remaining[:hard_cap]
        cut = None
        for m in re.finditer(r"[.!?]+", window):
            cut = m.end()
        if cut is None:
            cut = window.rfind(" ")
            if cut <= 0:
                cut = hard_cap  # nothing to break on at all — hard slice
        piece = remaining[:cut].strip()
        if piece:
            pieces.append(piece)
        remaining = remaining[cut:].strip()
    if remaining:
        pieces.append(remaining)
    return pieces


def chunk_text(text, min_sentences=4, hard_cap_min=400, hard_cap_max=3000):
    """
    Split text into paragraph chunks.
    Paragraphs shorter than min_sentences are merged with the next one
    until the minimum is met. Single short lines never become orphan memories.

    Failsafe: no chunk is allowed to exceed a hard character cap of ~2x
    the average paragraph length seen in this text, clamped to a sane
    range. Protects against a wall-of-text paste with no blank lines and
    little/no punctuation, which would otherwise blow straight past
    min_sentences and become one giant unbroken chunk.
    """
    # Split on blank lines
    raw_paragraphs = re.split(r"\n\s*\n", text.strip())
    raw_paragraphs = [p.strip() for p in raw_paragraphs if p.strip()]
    if not raw_paragraphs:
        return [text]

    avg_para_len = sum(len(p) for p in raw_paragraphs) / len(raw_paragraphs)
    hard_cap = int(max(hard_cap_min, min(avg_para_len * 2, hard_cap_max)))

    chunks = []
    buffer = ""
    for para in raw_paragraphs:
        if buffer:
            buffer += "\n\n" + para
        else:
            buffer = para
        buffer_sentences = len(re.findall(r"[.!?]+", buffer))
        # Flush on sentence count OR on hitting the hard cap early —
        # keeps buffer from ballooning while still building normally
        if buffer_sentences >= min_sentences or len(buffer) >= hard_cap:
            chunks.append(buffer)
            buffer = ""
    # Flush anything remaining
    if buffer:
        if chunks:
            chunks[-1] += "\n\n" + buffer
        else:
            chunks.append(buffer)
    if not chunks:
        chunks = [text]

    # Failsafe pass — nothing leaves this function oversized, no matter
    # how it got that way.
    final_chunks = []
    for c in chunks:
        if len(c) > hard_cap:
            final_chunks.extend(_split_on_hard_cap(c, hard_cap))
        else:
            final_chunks.append(c)

    return final_chunks


# ══════════════════════════════════════════════════════════════════════════════
# Main seed function
# ══════════════════════════════════════════════════════════════════════════════




def seed(user_message: str, ai_response: str, speaker: str = "Anonymous",
         image_filename: str = None, canon: str = None, document: bool = False):
    """
    speaker: who wrote user_message. None means nobody "said" it (Director Mode's committed story): it is
        stored as plain text with no speaker.
    canon: raw canon name from the Canon box, or None/blank for open memory. Every memory stored by this call
        (user, AI and image) belongs to that canon (Canons table, created on first use), which is what
        recall's canon lock reads.

    Every named speaker gets a Speakers row (feeds and documents too: the row is attribution). Whether they
    ever get a person gist is decided later by the conversation-partner bar. Lena's reply is stored under her
    own SpeakerID with PartnerID = the speaker.
    document: user_message is a document (a file dropped on the file bridge); its row is a 'document', shown
        as "<Name>: ..." instead of "<Name> said: ...". It never gets a person gist: nobody answers a document.
    """
    conn = memory_index.get_connection()
    try:
        cid = memory_index.canon_id(conn, canon, create=True)
        speaker_id = None
        if speaker is not None:
            speaker_id = memory_index.get_or_create_speaker(conn, speaker, kind="document" if document else "person")
        lena_id = memory_index.self_id(conn)
    finally:
        conn.close()
    label = (f"{speaker}: " if document else f"{speaker} said: ") if speaker is not None else ""   # handle context only

    names_conn = get_connection()          # names already in the table decide how a sentence-initial or lower-case name is filed
    known = known_names(names_conn)

    def extract(chunk):
        return handle_keywords(chunk, known)

    # ── Pass 1: User message ──────────────────────────────────────────────────
    if user_message.strip().upper() == "NULL":
        pass #Do Nothing, Nihil Est
    else:
        user_chunks = chunk_text(user_message)
        print(f"[SEED] User message split into {len(user_chunks)} chunk(s)")
        for i, chunk in enumerate(user_chunks):
            user_keywords = extract(chunk)
            if user_keywords or cid is not None:
                store_memory(chunk, user_keywords, speaker_id=speaker_id, cid=cid)
            else:
                print(f"[SEED] User chunk {i+1}: not worth storing.")

    # ── Pass 2: AI response ───────────────────────────────────────────────────
    if ai_response.strip().upper() == "NULL":
        pass #Hic est nihil
    else:
        ai_chunks = chunk_text(ai_response)
        print(f"[SEED] AI response split into {len(ai_chunks)} chunk(s)")
        for i, chunk in enumerate(ai_chunks):
            ai_keywords = extract(chunk)
            if ai_keywords or cid is not None:
                # her own line, answering this person
                store_memory(chunk, ai_keywords, speaker_id=lena_id, cid=cid, partner_id=speaker_id)
            else:
                print(f"[SEED] AI chunk {i+1}: not worth storing.")

    # ── Pass 3: Image (if one was uploaded this exchange) ────────────────────
    if image_filename:
        combined_text = f"{label}{user_message}\nI said: {ai_response}"
        image_keywords = extract(combined_text)
        if image_keywords:
            image_id = store_image_memory(image_filename, image_keywords, cid=cid)
            # Supplemental description + extra keywords, only when a vision
            # route exists. Runs here in the seed thread, so it never blocks
            # a reply; the image is already usable by keyword either way.
            if image_id is not None:
                try:
                    import image_describe
                    image_describe.describe_and_link(image_id, image_filename, combined_text)
                except Exception as e:
                    print(f"[SEED][WARNING] Image description skipped: {e}")
        else:
            print("[SEED] Image: no keywords extracted, not linked.")
    names_conn.close()


def note(text):
    """Stores one standalone memory in Lena's own words (no speaker, no reply), e.g. that her model was unreachable."""
    conn = get_connection()
    keywords = handle_keywords(text, known_names(conn))
    conn.close()
    if keywords:
        store_memory(text, keywords)


if __name__ == "__main__":
    import sys
    if len(sys.argv) == 3:
        seed(sys.argv[1], sys.argv[2])
    else:
        print("Usage: python3 seed.py \"user message\" \"ai response\"")
