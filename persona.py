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
persona.py — dream-cycle pass: one person gist and one dossier per person, per place.

Recall always puts the speaker's person gist (and dossier) in front of the model, so they have to stay
current. A person can have one in open chat and one in each canon they talk to Lena in (Personas table):
inside a canon, only what they said there, so a Discord canon never learns about someone's family from
their open-chat gist. For every (person, canon) where they are a conversation partner (Lena has answered
them at least persona_min_replies times and for at least persona_min_reply_ratio of their turns there),
this stage:

  * adopts their existing gist if there is none yet: the gist in that canon whose only handle is their name,
    i.e. what Lena already knew about them secondhand,
  * reads their NEW statements there (verbatim, spoken by them) since the last run,
  * REWRITES that gist from the old text plus the new statements (one gist per person per canon, replaced in
    place),
  * maintains its dossier (state: goals, limitations, assets, knowledge, location, relationships, knowledge
    gaps), creating it if needed,
  * re-embeds the rewritten gist.

Once adopted (Personas.GistRef), the dream's symbol passes leave that gist alone.
"""

import re
import time
from pathlib import Path

import llm_client
import memory_index
from keyword_utils import ensure_keyword

ROOT = Path(__file__).parent.resolve()
PROMPTS_DIR = ROOT / "prompts"

DEFAULT_GIST_PROMPT = (
    "You maintain a gist describing a person, used by an assistant that talks with them. You are given the current gist "
    "(it may be empty) and new things the person said recently. Write the updated gist. Keep every durable fact: name or "
    "nickname and how they like to be addressed, work and employer, home, partner, family members with their names, "
    "places and jobs, pets, allergies and health, friends, notable history, plans and trips, preferences (units, food, "
    "music, how they like to be spoken to). Replace facts that have changed and say what changed when it matters. "
    "Use only supported information. Plain prose, 6 to 10 sentences. No headings and no explanation."
)
DOSSIER_FIELDS = ["GOALS", "LIMITATIONS", "ASSETS", "KNOWLEDGE", "LOCATION", "RELATIONSHIPS", "KNOWLEDGE_GAPS"]
FIELD_CAP = 700


def _load(name, default=""):
    path = PROMPTS_DIR / name
    try:
        text = path.read_text(encoding="utf-8").strip()
        return text or default
    except OSError:
        return default


def _gist_prompt():
    return _load("persona_gist_prompt.txt", DEFAULT_GIST_PROMPT)


def _dossier_prompt():
    base = _load("dossier_field_patch.txt")
    return base.replace("committed story text", "recent conversation") if base else ""


def _strip_label(text):
    return re.sub(r"^((Gist|GIST):\s*)+", "", text or "").strip()


def _ask(system_prompt, user_text, max_tokens):
    """One LLM call. An empty answer (for instance a thinking model that spent the whole budget thinking)
    is retried once with a much larger budget, and reported instead of failing silently."""
    for tokens in (max_tokens, max_tokens * 3):
        try:
            memory_index.dream_pause()
            out = llm_client.get_completion(system_prompt=system_prompt, user_text=user_text,
                                            max_tokens=tokens, temperature=0.3, timeout=300)
        except Exception as e:
            print(f"[PERSONA][WARNING] LLM call failed: {e}")
            return None
        if out and out.strip():
            return out
        print(f"[PERSONA][WARNING] LLM returned nothing at {tokens} tokens"
              + (" — retrying with a larger budget." if tokens == max_tokens else "."))
    return None


def _parse_dossier_lines(text):
    """FIELD: value lines -> {FIELD: value}; a value of NONE means clear the field."""
    out = {}
    for line in (text or "").splitlines():
        m = re.match(r"^(GOALS|LIMITATIONS|ASSETS|KNOWLEDGE|LOCATION|RELATIONSHIPS|KNOWLEDGE_GAPS):\s*(.*)$", line.strip())
        if m:
            out[m.group(1)] = m.group(2).strip()
    return out


def _read_dossier(conn, mem_ref):
    row = conn.execute("SELECT Goals, Limitations, Assets, Knowledge, Location, Relationships, KnowledgeGaps "
                       "FROM Dossiers WHERE MemRef = ?", (mem_ref,)).fetchone()
    if not row:
        return {}
    return {f: v for f, v in zip(DOSSIER_FIELDS, row) if v}


def _write_dossier(conn, mem_ref, name, fields):
    vals = [fields.get(f) for f in DOSSIER_FIELDS]
    conn.execute("""INSERT INTO Dossiers (MemRef, Goals, Limitations, Assets, Knowledge, Location, Relationships, KnowledgeGaps, UpdatedAt, FullName)
                    VALUES (?,?,?,?,?,?,?,?,?,?)
                    ON CONFLICT(MemRef) DO UPDATE SET Goals=excluded.Goals, Limitations=excluded.Limitations, Assets=excluded.Assets,
                        Knowledge=excluded.Knowledge, Location=excluded.Location, Relationships=excluded.Relationships,
                        KnowledgeGaps=excluded.KnowledgeGaps, UpdatedAt=excluded.UpdatedAt,
                        FullName=COALESCE(Dossiers.FullName, excluded.FullName)""",
                 (mem_ref, *vals, time.time(), name))
    conn.commit()


def _write_gist(conn, persona_id, cid, name_key, gist_ref, text):
    """Replaces the person's gist in place, or creates it in canon cid (single handle = the name). Returns the MemRef."""
    now = int(time.time())
    body = "Gist: " + text
    if gist_ref is not None and conn.execute("SELECT 1 FROM Memory WHERE MemRef = ?", (gist_ref,)).fetchone():
        conn.execute("UPDATE Memory SET Memory = ?, OriginTime = ?, Embedding = NULL WHERE MemRef = ?", (body, now, gist_ref))
        ref = gist_ref
    else:
        ref = conn.execute("INSERT INTO Memory (OriginTime, Memory, RecallCount, CanonID) VALUES (?, ?, ?, ?)",
                           (now, body, -1, cid)).lastrowid
    memory_index.link_handles(conn, ref, [name_key])      # a person gist's only handle is the person's name
    conn.execute("UPDATE Personas SET GistRef = ? WHERE PersonaID = ?", (ref, persona_id))
    conn.commit()
    memory_index.embed_memory(conn, ref)
    return ref


def refresh_speaker(conn, spk_id, cid, persona_id, name, name_key, gist_ref, last_ref, stop_event=None):
    """Brings one person's gist and dossier in canon cid (None = open chat) up to date. Returns the number of blocks processed."""
    cfg = memory_index.settings()
    block_size = int(cfg["persona_block_size"])
    new = conn.execute("""
        SELECT m.MemRef, m.OriginTime, m.Memory FROM Memory m
        WHERE m.SpeakerID = ? AND m.CanonID IS ? AND m.RecallCount >= 0 AND m.MemRef > ?
        ORDER BY m.MemRef""", (spk_id, cid, last_ref)).fetchall()
    if len(new) < int(cfg["persona_min_new_memories"]):
        return 0
    gist_prompt, dossier_prompt = _gist_prompt(), _dossier_prompt()
    done = 0
    for i in range(0, len(new), block_size):
        if done >= int(cfg["persona_max_blocks_per_cycle"]) or (stop_event is not None and stop_event.is_set()):
            break
        block = new[i:i + block_size]
        lines = "\n".join(f"- [{time.strftime('%Y-%m-%d', time.localtime(t))}] {m[:300]}"
                          for _, t, m in block)
        current = _strip_label(conn.execute("SELECT Memory FROM Memory WHERE MemRef = ?", (gist_ref,)).fetchone()[0]) if gist_ref else ""
        where = f" (as they are in {memory_index.canon_name(conn, cid)!r})" if cid is not None else ""
        gist = _ask(gist_prompt, f"Person: {name}{where}\n\nCurrent gist:\n{current or '(none)'}\n\nNew statements:\n{lines}", 1100)
        if not gist:
            break
        gist = _strip_label(gist)
        gist_ref = _write_gist(conn, persona_id, cid, name_key, gist_ref, gist)
        if dossier_prompt:
            fields = _read_dossier(conn, gist_ref)
            cur = "\n".join(f"{k}: {v}" for k, v in fields.items()) or "(empty)"
            out = _ask(dossier_prompt, f"Entity: {name} (the person the assistant is talking with)\nCurrent dossier:\n{cur}\n\n"
                                       f"Gist: {gist}\n\nRecent statements:\n{lines}", 700)
            for field, value in _parse_dossier_lines(out).items():
                if value.upper() == "NONE":
                    fields.pop(field, None)
                elif value:
                    fields[field] = value[:FIELD_CAP]
            _write_dossier(conn, gist_ref, name, fields)
        last_ref = block[-1][0]
        conn.execute("UPDATE Personas SET LastPersonaRef = ? WHERE PersonaID = ?", (last_ref, persona_id))
        conn.commit()
        done += 1
        print(f"[PERSONA] {name}: block {done} done ({len(block)} statements, gist {len(gist)} chars)")
    return done


def run(stop_event=None):
    """Dream-cycle entry point. Does nothing when nobody qualifies or there is nothing new to fold in."""
    conn = memory_index.get_connection()
    total = 0
    try:
        scopes = conn.execute("""SELECT DISTINCT s.SpeakerID, s.Name, s.NameKey, m.CanonID FROM Speakers s
                                 JOIN Memory m ON m.SpeakerID = s.SpeakerID WHERE s.Kind = 'person'
                                 ORDER BY s.SpeakerID, m.CanonID""").fetchall()
        for spk_id, name, name_key, cid in scopes:
            if stop_event is not None and stop_event.is_set():
                break
            turns, replied = memory_index.conversation_stats(conn, spk_id, cid)
            if not memory_index.is_conversation_partner(turns, replied):
                continue                              # feeds, passing commenters: ordinary gists, no person gist/dossier
            persona_id, gist_ref, last_ref = memory_index.persona_row(conn, spk_id, cid, create=True)
            if gist_ref is None:                      # adopt what Lena already knew about them here, secondhand
                found = memory_index.find_identity(conn, name, cid)
                if found is not None:
                    conn.execute("UPDATE Personas SET GistRef = ? WHERE PersonaID = ?", (found, persona_id))
                    conn.commit()
                    gist_ref = found
            total += refresh_speaker(conn, spk_id, cid, persona_id, name, name_key, gist_ref, last_ref or 0, stop_event)
    except Exception as e:
        print(f"[PERSONA][WARNING] Person-gist pass stopped: {e}")
    finally:
        conn.close()
    return total
