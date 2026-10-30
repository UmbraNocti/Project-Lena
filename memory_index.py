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
memory_index.py — the long-lived half of recall.

recall.py is re-loaded by controller.py on every call, so anything that should survive between questions lives
here, in a normally imported module:

  * settings()      recall_* and persona_* tuning values from settings.txt (re-read when the file changes)
  * canons          canon_key, canon_id, canon_name (the Canons table; a canon is a row, not a keyword)
  * speakers        speaker_key, self_id, get_or_create_speaker, speaker_table, render (a memory with its
                    speaker label, which is never stored in the text), conversation_stats, is_conversation_partner
  * personas        persona_row, adopted_gists, find_identity (the person gist for one person in one canon)
  * gist_with_handles  the gist whose handles are exactly a given set (what it is about), in a given canon;
                    text_handles / link_handles write a derived memory's tags
  * TableStore      the associations: handle rarity, shared handles and handle counts straight from Keywords and Nodes,
                    one query each, nothing cached between recalls
  * EMBEDDINGS      the stored memory vectors as one matrix for cosine search, refreshed incrementally
  * embed_memory / embed_missing / sync_embedding_meta   writing embeddings

It never creates or alters tables: db_setup.py does that once at start-up.
"""

import array
import math
import re
import sqlite3
import threading
import time
from pathlib import Path

import llm_client
from keyword_utils import handle_keywords, ensure_keyword, known_names

try:  # numpy is optional; without it the cosine search falls back to plain Python
    import numpy as _np
except Exception:  # pragma: no cover
    _np = None

ROOT = Path(__file__).parent.resolve()
DB_PATH = ROOT / "MEM.db"
SETTINGS_FILE = ROOT / "settings.txt"

# ══════════════════════════════════════════════════════════════════════════════
# Settings
# ══════════════════════════════════════════════════════════════════════════════

_DEFAULTS = {
    "gist_min_mentions": 10,        # a handle may found a gist once this many memories (all time, in its canon) carry it
    "gist_prior": 0.043,            # share of the words the gist gate has to judge that really deserve a gist
    "gist_fold_chars": 24000,
    "dream_pause_seconds": 0.0,
    "gist_judge_margin": 1.0,       # Bayes scores this close to the line (and names that are ordinary words) are asked of a model in the dream; 0 = never     # rest between the dream's model calls (a hot or shared GPU); 0 = none
    "alias_overlap": 0.85,          # two names are aliases when each one's memories are at least this share of the other's       # most source text sent in one gist call (about 6-7k tokens); more is folded in over several calls
    "recall_vector_hits": 2,
    "recall_mined_weight": 0.2,
    "recall_mined_per_hit": 10,
    "recall_mined_identity": 14,
    "recall_identity_weight": 0.4,
    "recall_context_handles": 20,
    "recall_standing_entities": 6,
    "recall_source_bonus": 0.6,
    "recall_min_rarity": 0.5,
    "recall_length_exponent": 0.25,
    "recall_recency_days": 120.0,
    "recall_recency_floor": 0.6,
    "persona_min_new_memories": 15,
    "persona_min_replies": 10,
    "persona_min_reply_ratio": 0.75,
    "persona_block_size": 25,
    "persona_max_blocks_per_cycle": 6,
}
_SETTINGS_CACHE = {"mtime": None, "values": dict(_DEFAULTS)}


def settings():
    try:
        mtime = SETTINGS_FILE.stat().st_mtime
    except OSError:
        return _SETTINGS_CACHE["values"]
    if mtime == _SETTINGS_CACHE["mtime"]:
        return _SETTINGS_CACHE["values"]
    values = dict(_DEFAULTS)
    try:
        for line in SETTINGS_FILE.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, val = line.split("=", 1)
            key, val = key.strip().strip('"').strip("'"), val.strip().strip('"').strip("'")
            if key in values:
                default = _DEFAULTS[key]
                try:
                    values[key] = type(default)(val)
                except ValueError:
                    pass
    except OSError:
        pass
    _SETTINGS_CACHE.update(mtime=mtime, values=values)
    return values


# ══════════════════════════════════════════════════════════════════════════════
# Connection, canons, speakers, personas
# ══════════════════════════════════════════════════════════════════════════════

def get_connection():
    conn = sqlite3.connect(DB_PATH, timeout=30)
    conn.execute("PRAGMA journal_mode=WAL")
    return conn


def speaker_key(name):
    """Whitespace stripped and upper-cased: "James Smith" -> "JAMESSMITH". Also used for canon names."""
    return re.sub(r"\s+", "", (name or "").strip()).upper()


canon_key = speaker_key


def canon_id(conn, canon, create=False):
    """
    The CanonID for a canon name as typed in the Canon box (case and spaces don't matter), or None for a blank
    name. create=True makes the row when it doesn't exist yet (seeding); otherwise an unknown canon is None.
    """
    key = canon_key(canon)
    if not key:
        return None
    row = conn.execute("SELECT CanonID FROM Canons WHERE NameKey = ?", (key,)).fetchone()
    if row:
        return row[0]
    if not create:
        return None
    cur = conn.execute("INSERT INTO Canons (Name, NameKey, CreatedAt) VALUES (?, ?, strftime('%s','now'))", (canon.strip(), key))
    conn.commit()
    return cur.lastrowid


def canon_name(conn, cid):
    """The canon's display name, or "" for open memory."""
    if cid is None:
        return ""
    row = conn.execute("SELECT Name FROM Canons WHERE CanonID = ?", (cid,)).fetchone()
    return row[0] if row else f"canon {cid}"


SELF_KEY = "#SELF"     # Lena's own row in Speakers (db_setup creates it)


def self_id(conn):
    """Lena's own SpeakerID."""
    row = conn.execute("SELECT SpeakerID FROM Speakers WHERE Kind = 'self' ORDER BY SpeakerID LIMIT 1").fetchone()
    return row[0] if row else None


def get_or_create_speaker(conn, name, create=True, kind="person"):
    """
    Returns the SpeakerID for a name ("James", a bridge user, a document's file name), creating the row
    (with the given kind: 'person' or 'document') unless create=False. None for blank names. Every named
    speaker gets a row, feeds included: a row is attribution, not a relationship (person gists have their
    own bar, see is_conversation_partner).
    """
    key = speaker_key(name)
    if not key or key in ("ANONYMOUS", "I"):
        return None
    row = conn.execute("SELECT SpeakerID FROM Speakers WHERE NameKey = ?", (key,)).fetchone()
    if row:
        return row[0]
    if not create:
        return None
    cur = conn.execute("INSERT INTO Speakers (Name, NameKey, Kind) VALUES (?, ?, ?)", (name.strip(), key, kind))
    conn.commit()
    return cur.lastrowid


def speaker_table(conn):
    """{SpeakerID: (Name, Kind)} for render()."""
    return {r[0]: (r[1], r[2]) for r in conn.execute("SELECT SpeakerID, Name, Kind FROM Speakers")}


def render(text, speaker_id, speakers):
    """
    A memory as the model (and the dream, and the embedder) sees it. The text is stored bare; the speaker is
    added from the Speakers table: Lena's own lines as "I said: ...", a person's as "<Name> said: ...", a
    document's as "<Name>: ...". Memories with no speaker (gists, summaries, story text) are shown as they are.
    """
    if speaker_id is None or speaker_id not in speakers:
        return text
    name, kind = speakers[speaker_id]
    if kind == "self":
        return f"I said: {text}"
    if kind == "document":
        return f"{name}: {text}"
    return f"{name} said: {text}"


def rendered(conn, mem_ref):
    """One stored memory, rendered (see render)."""
    row = conn.execute("SELECT Memory, SpeakerID FROM Memory WHERE MemRef = ?", (mem_ref,)).fetchone()
    return render(row[0], row[1], speaker_table(conn)) if row else ""


def conversation_stats(conn, speaker_id, cid=None):
    """
    (turns, replied) for one speaker in one canon (cid None = open chat): a turn is a run of their own
    memories, replied when one of Lena's lines answering them (PartnerID) follows it. Used to decide who is a
    real conversation partner there.
    """
    me = self_id(conn)
    turns = replied = 0
    in_turn = False
    for (spk,) in conn.execute("""SELECT SpeakerID FROM Memory
                                  WHERE CanonID IS ? AND RecallCount >= 0
                                    AND (SpeakerID = ? OR (SpeakerID = ? AND PartnerID = ?))
                                  ORDER BY MemRef""", (cid, speaker_id, me, speaker_id)):
        if spk == me:
            if in_turn:
                replied += 1
            in_turn = False
        elif not in_turn:
            turns += 1
            in_turn = True
    return turns, replied


def is_conversation_partner(turns, replied):
    """
    The bar for a person gist and dossier: Lena has answered them at least persona_min_replies times, and
    answered at least persona_min_reply_ratio of everything they sent. Someone chatting with her clears it
    quickly; a news feed or a passing commenter does not.
    """
    cfg = settings()
    return replied >= int(cfg["persona_min_replies"]) and turns > 0 and replied / turns >= float(cfg["persona_min_reply_ratio"])


def persona_row(conn, speaker_id, cid, create=False):
    """(PersonaID, GistRef, LastPersonaRef) for one person in one canon (cid None = open chat), or None."""
    row = conn.execute("SELECT PersonaID, GistRef, LastPersonaRef FROM Personas WHERE SpeakerID = ? AND CanonID IS ?",
                       (speaker_id, cid)).fetchone()
    if row or not create:
        return row
    conn.execute("INSERT INTO Personas (SpeakerID, CanonID, LastPersonaRef) VALUES (?, ?, 0)", (speaker_id, cid))
    conn.commit()
    return persona_row(conn, speaker_id, cid)


def adopted_gists(conn):
    """MemRefs of every gist a persona owns. The dream's symbol passes and compression leave these alone."""
    return {r[0] for r in conn.execute("SELECT GistRef FROM Personas WHERE GistRef IS NOT NULL")}


def gist_with_handles(conn, keys, cid=None):
    """
    The gist (newest first) whose handles are EXACTLY `keys`, in canon `cid` (None = open memory), or None. A gist
    is tagged only with what it is about (usually one handle, a name), so this is how a symbol's gist is found:
    {GERALT} in open memory and {GERALT} in the Witcher canon are two different gists.
    """
    keys = {k for k in keys if k}
    if not keys:
        return None
    q = ",".join("?" * len(keys))
    for (ref,) in conn.execute(f"""
            SELECT m.MemRef FROM Memory m JOIN Nodes n ON n.MemID = m.MemRef JOIN Keywords k ON k.KeyID = n.KeyID
            WHERE m.RecallCount < 0 AND m.CanonID IS ?
            GROUP BY m.MemRef
            HAVING COUNT(*) = ? AND SUM(CASE WHEN k.Key IN ({q}) THEN 1 ELSE 0 END) = ?
            ORDER BY m.OriginTime DESC""", (cid, len(keys), *keys, len(keys))):
        return ref
    return None


_LABEL = re.compile(r"^((Gist|GIST|Derived):\s*)+")


def text_handles(conn, text):
    """The handles of a memory's own text (a leading 'Gist:' / 'Derived:' label is not part of it)."""
    return handle_keywords(_LABEL.sub("", text or ""), known_names(conn))


def dream_pause():
    """The rest the dream takes before each model call (dream_pause_seconds; 0 = none)."""
    pause = float(settings()["dream_pause_seconds"])
    if pause > 0:
        time.sleep(pause)


def alias_groups(conn, cid=None):
    """
    The aliases of canon cid (None = open memory), from the Aliases table: (main_of, members) where main_of maps
    each alias to its main name (KEETON -> MARLOWE) and members maps each main name to its whole group, main first
    ([MARLOWE, JAMES, KEETON]). Empty for a database without the table.
    """
    main_of, members = {}, {}
    try:
        rows = conn.execute("SELECT k.Key, mk.Key FROM Aliases a JOIN Keywords k ON k.KeyID = a.KeyID "
                            "JOIN Keywords mk ON mk.KeyID = a.MainKeyID WHERE a.CanonID IS ? ORDER BY k.Key", (cid,)).fetchall()
    except sqlite3.OperationalError:
        return main_of, members
    for alias, main in rows:
        main_of[alias] = main
        members.setdefault(main, [main]).append(alias)
    return main_of, members


def link_handles(conn, mem_ref, keys):
    """Replaces a memory's tags with exactly these handles. Does not commit."""
    ids = list(dict.fromkeys(ensure_keyword(conn, k) for k in keys if k))
    conn.execute("DELETE FROM Nodes WHERE MemID = ?", (mem_ref,))
    conn.executemany("INSERT INTO Nodes (MemID, KeyID) VALUES (?, ?)", [(mem_ref, i) for i in ids])


def find_identity(conn, speaker, cid=None):
    """
    The speaker's person gist in this canon (cid None = open chat), or None. Order: the persona's adopted gist,
    then the gist (in this canon) whose ONLY handle is the speaker's name, also tried with just the first name.
    That second case is what Lena knew about them secondhand before they ever talked to her.
    """
    if not speaker:
        return None
    key = speaker_key(speaker)
    spk = conn.execute("SELECT SpeakerID FROM Speakers WHERE NameKey = ?", (key,)).fetchone()
    if spk:
        row = persona_row(conn, spk[0], cid)
        if row and row[1] is not None and conn.execute("SELECT 1 FROM Memory WHERE MemRef = ?", (row[1],)).fetchone():
            return row[1]
    candidates = [key]
    first = speaker_key(speaker.strip().split()[0]) if speaker.strip() else ""
    if first and first != key and len(first) >= 3 and ":" not in speaker:
        candidates.append(first)      # "James Smith" -> a gist whose only handle is JAMES
    for cand in candidates:
        found = gist_with_handles(conn, {cand}, cid)
        if found:
            return found
    return None


# ══════════════════════════════════════════════════════════════════════════════
# Associations: the walk over Keywords and Nodes
# ══════════════════════════════════════════════════════════════════════════════

class _LazyMem:
    """Memory rows by MemRef, fetched from the database in blocks as they are asked for."""
    def __init__(self, conn):
        self.conn = conn
        self.rows = {}
        self.missing = set()

    def prefetch(self, refs):
        need = [r for r in refs if r not in self.rows and r not in self.missing]
        for i in range(0, len(need), 500):
            part = need[i:i + 500]
            q = ",".join("?" * len(part))
            for ref, text, t, rc, cid, spk in self.conn.execute(
                    f"SELECT MemRef, Memory, OriginTime, RecallCount, CanonID, SpeakerID FROM Memory WHERE MemRef IN ({q})", part):
                self.rows[ref] = {"text": text, "t": t, "gist": rc < 0, "canon": cid, "speaker": spk}
            self.missing.update(r for r in part if r not in self.rows)

    def get(self, ref):
        if ref not in self.rows and ref not in self.missing:
            self.prefetch([ref])
        return self.rows.get(ref)

    def __getitem__(self, ref):
        return self.rows[ref] if ref in self.rows else self.get(ref)

    def __contains__(self, ref):
        return self.get(ref) is not None


class TableStore:
    """
    Associations straight from the database: a memory is linked to its handles by Nodes, and everything recall
    needs (how rare a handle is, which memories share it, how many handles a memory has) is a query over Keywords
    and Nodes. Nothing is cached between recalls, so a tag written a moment ago is already in play.
    """
    def __init__(self, conn):
        self.conn = conn
        self.mem = _LazyMem(conn)
        self.n = conn.execute("SELECT COUNT(*) FROM Memory").fetchone()[0]
        self._df = {}

    def df(self, stems):
        """{stem: number of memories carrying it}"""
        need = [h for h in set(stems) if h not in self._df]
        for i in range(0, len(need), 500):
            part = need[i:i + 500]
            q = ",".join("?" * len(part))
            for stem_, n in self.conn.execute(
                    f"""SELECT k.Key, COUNT(DISTINCT n.MemID) FROM Keywords k JOIN Nodes n ON n.KeyID = k.KeyID
                        WHERE n.MemID IS NOT NULL AND k.Key IN ({q}) GROUP BY k.Key""", part):
                self._df[stem_] = n
            for h in part:
                self._df.setdefault(h, 0)
        return {h: self._df[h] for h in stems}

    def prime(self, stems):
        """Looks up how common each of these handles is, in as few queries as possible."""
        self.df(list(stems))

    def rarity(self, h):
        return math.log((self.n + 1) / (self.df([h])[h] + 1))

    def stems_of(self, refs):
        """{MemRef: set of handles} for the memories' own tags."""
        out = {r: set() for r in refs}
        for i in range(0, len(refs), 500):
            part = refs[i:i + 500]
            q = ",".join("?" * len(part))
            for ref, stem_ in self.conn.execute(
                    f"SELECT n.MemID, k.Key FROM Nodes n JOIN Keywords k ON k.KeyID = n.KeyID WHERE n.MemID IN ({q})", part):
                if stem_:
                    out[ref].add(stem_)
        return out

    def top_handles(self, refs, per_item):
        """
        Union of each memory's `per_item` rarest handles, read from its tags. A gist (a derived memory with a single
        tag, the thing it is about) has nothing else to follow in the table, so its handles are taken from its text
        instead; this reads the text of the few memories already in hand, never the table.
        """
        out = set()
        tags = self.stems_of(list(refs))
        self.mem.prefetch(list(refs))
        for r in refs:
            m = self.mem.get(r)
            if m and m["gist"] and len(tags.get(r, ())) <= 1:
                tags[r] = set(text_handles(self.conn, m["text"]))
        self.df(set().union(*tags.values()) if tags else [])
        for r in refs:
            out.update(sorted(tags.get(r, ()), key=lambda h: -self.rarity(h))[:per_item])
        return out

    def candidates(self, weights, cid, locked):
        """
        {MemRef: (raw score, OriginTime, handle count)}: for every memory that carries at least one weighted
        handle, the sum of those handles' weights, and what the length and recency terms need. One query.
        """
        if not weights:
            return {}
        self.conn.execute("CREATE TEMP TABLE IF NOT EXISTS _recall_w (Handle TEXT PRIMARY KEY, W REAL)")
        self.conn.execute("DELETE FROM _recall_w")
        self.conn.executemany("INSERT INTO _recall_w (Handle, W) VALUES (?, ?)", list(weights.items()))
        sql = """
            SELECT m.MemRef, SUM(w.W), m.OriginTime,
                   (SELECT COUNT(*) FROM Nodes x WHERE x.MemID = m.MemRef)
            FROM Keywords k JOIN _recall_w w ON w.Handle = k.Key
            JOIN Nodes n ON n.KeyID = k.KeyID
            JOIN Memory m ON m.MemRef = n.MemID"""
        params = ()
        if locked:
            sql += " WHERE m.CanonID = ?"
            params = (cid,)
        sql += " GROUP BY m.MemRef"
        return {ref: (raw, t, hc) for ref, raw, t, hc in self.conn.execute(sql, params)}


# ══════════════════════════════════════════════════════════════════════════════
# Embeddings
# ══════════════════════════════════════════════════════════════════════════════

def _to_blob(vec):
    norm = math.sqrt(sum(x * x for x in vec)) or 1.0
    return array.array("f", [x / norm for x in vec]).tobytes()


def _from_blob(blob):
    a = array.array("f")
    a.frombytes(blob)
    return a


def sync_embedding_meta(conn):
    """
    Compares the configured embedding model with the one the stored vectors came from. If they differ,
    every stored embedding is stale: they are all set to NULL (to be rebuilt by embed_missing) and the
    meta row is updated. Returns True when embeddings are enabled.
    """
    cfg = llm_client.embedding_config()
    if cfg is None:
        return False
    row = conn.execute("SELECT ModelName FROM EmbeddingMeta WHERE Id = 1").fetchone()
    if row is None:
        conn.execute("INSERT INTO EmbeddingMeta (Id, ModelName, Dim) VALUES (1, ?, NULL)", (cfg["model_id"],))
        conn.commit()
    elif row[0] != cfg["model_id"]:
        conn.execute("UPDATE Memory SET Embedding = NULL")
        conn.execute("UPDATE EmbeddingMeta SET ModelName = ?, Dim = NULL WHERE Id = 1", (cfg["model_id"],))
        conn.commit()
        EMBEDDINGS.reset()
    return True


def embed_memory(conn, mem_ref):
    """
    Embeds one stored memory now, as it is rendered (with its speaker, see render). Best effort: a failure
    leaves the column NULL for the dream cycle's embedding pass.
    """
    if not llm_client.embedding_enabled():
        return False
    try:
        vec = llm_client.get_embeddings([_embed_text(rendered(conn, mem_ref))], kind="document")[0]
        conn.execute("UPDATE Memory SET Embedding = ? WHERE MemRef = ?", (_to_blob(vec), mem_ref))
        conn.execute("UPDATE EmbeddingMeta SET Dim = ? WHERE Id = 1 AND Dim IS NULL", (len(vec),))
        conn.commit()
        return True
    except Exception as e:
        print(f"[MEMORY_INDEX][WARNING] Embedding skipped for MemRef={mem_ref}: {e}")
        return False


def embed_missing(conn, limit=500):
    """Embeds memories whose Embedding is NULL, oldest first (dream cycle / catch-up). Returns how many were done."""
    if not llm_client.embedding_enabled():
        return 0
    speakers = speaker_table(conn)
    rows = [(ref, render(text, spk, speakers)) for ref, text, spk in conn.execute(
        "SELECT MemRef, Memory, SpeakerID FROM Memory WHERE Embedding IS NULL ORDER BY MemRef LIMIT ?", (limit,))]
    done = 0
    cfg = llm_client.embedding_config()
    for i in range(0, len(rows), cfg["batch"]):
        part = rows[i:i + cfg["batch"]]
        try:
            vecs = llm_client.get_embeddings([_embed_text(t) for _, t in part], kind="document")
        except Exception as e:
            print(f"[MEMORY_INDEX][WARNING] Embedding batch failed: {e}")
            break
        for (ref, _), vec in zip(part, vecs):
            conn.execute("UPDATE Memory SET Embedding = ? WHERE MemRef = ?", (_to_blob(vec), ref))
            done += 1
        conn.execute("UPDATE EmbeddingMeta SET Dim = ? WHERE Id = 1 AND Dim IS NULL", (len(vecs[0]),))
        conn.commit()
    return done


def _embed_text(text):
    """What actually gets embedded: leading 'Gist:' / 'Derived:' labels (sometimes stacked) are removed."""
    return re.sub(r"^((Gist|GIST|Derived):\s*)+", "", text or "")


class _EmbeddingCache:
    def __init__(self):
        self.lock = threading.RLock()
        self.reset()

    def reset(self):
        self.refs, self.max_ref, self.mat, self.rows = [], 0, None, []

    def refresh(self, conn):
        with self.lock:
            total = conn.execute("SELECT COUNT(*) FROM Memory WHERE Embedding IS NOT NULL").fetchone()[0]
            new = conn.execute("SELECT MemRef, Embedding FROM Memory WHERE Embedding IS NOT NULL AND MemRef > ? ORDER BY MemRef",
                               (self.max_ref,)).fetchall()
            if len(self.refs) + len(new) != total:        # something was cleared or rewritten: reload everything
                self.reset()
                new = conn.execute("SELECT MemRef, Embedding FROM Memory WHERE Embedding IS NOT NULL ORDER BY MemRef").fetchall()
            if not new:
                return
            vecs = [_from_blob(b) for _, b in new]
            self.refs.extend(r for r, _ in new)
            self.max_ref = self.refs[-1]
            if _np is not None:
                add = _np.array([list(v) for v in vecs], dtype="float32")
                self.mat = add if self.mat is None else _np.vstack([self.mat, add])
            else:
                self.rows.extend(vecs)

    def rank(self, qvec):
        """[(MemRef, cosine)] best first. Vectors are stored unit length, so the dot product is the cosine."""
        with self.lock:
            if not self.refs:
                return []
            norm = math.sqrt(sum(x * x for x in qvec)) or 1.0
            if _np is not None:
                q = _np.array(qvec, dtype="float32") / norm
                sims = self.mat @ q
                order = _np.argsort(-sims)
                return [(self.refs[i], float(sims[i])) for i in order]
            q = [x / norm for x in qvec]
            sims = [sum(a * b for a, b in zip(row, q)) for row in self.rows]
            order = sorted(range(len(sims)), key=lambda i: -sims[i])
            return [(self.refs[i], sims[i]) for i in order]


EMBEDDINGS = _EmbeddingCache()
