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
db_setup.py — the only code that creates MEM.db's structure. Run once at start-up by controller.py (and safe to
run by hand: `python3 db_setup.py`, add `--embed` to embed every memory now).

  * No MEM.db: creates a fresh one with every table Lena uses (and the default Director commands).
  * Existing MEM.db: checked (the gist gate's tables are added if missing). A database from before the Canons table,
    or with the older handle form (literal words with a Stem column), is refused with a clear message (move it
    aside and Lena creates a fresh one).
  * A new database also gets the shipped hand labels (data/gist_labels.csv): every handle marked "no" is added with
    ToGist = 0, so it never gets a gist, and the gist gate (gistable.py) is trained from them.
  * Every start: the embedding model in settings.txt is compared with the one the stored vectors came from; if it
    changed, all embeddings are cleared so the dream cycle rebuilds them.

The tables, in one line each:
  Memory     every memory, as bare text: verbatim chunks, gists, summaries. A gist is tagged only with what it is
             about; everything else with the handles of its own text. SpeakerID = who said or wrote it
             (NULL: nobody, e.g. story text or Lena's own notes), PartnerID = for Lena's own lines, the person
             she was answering, CanonID = the canon it belongs to (NULL: open memory). There are no "X said:"
             labels in the text; they are added from Speakers when a memory is shown (memory_index.render).
  Keywords   handles (Key: a name in UPPER CASE or a lower-case Porter2 stem; ToGist: 1/0 is a human's say-so,
             NULL leaves it to the gate).
  Nodes      memory/image <-> keyword links: the associations recall walks.
  Images     image memories, with their CanonID.          Dossiers  state per entity, keyed by its gist.
  Speakers   one row per speaker: Kind 'self' (Lena), 'person' or 'document'.   Canons   one row per canon.
  Personas   how Lena knows one person in one canon (CanonID NULL = open chat): the adopted gist and how far
             the dream has read their statements.
"""

import sqlite3
import sys
from pathlib import Path

ROOT = Path(__file__).parent.resolve()
DB_PATH = ROOT / "MEM.db"

TABLES = {
    "Canons": """CREATE TABLE Canons (
        CanonID INTEGER PRIMARY KEY AUTOINCREMENT,
        Name TEXT NOT NULL,
        NameKey TEXT NOT NULL UNIQUE,
        CreatedAt INTEGER)""",
    "Speakers": """CREATE TABLE Speakers (
        SpeakerID INTEGER PRIMARY KEY AUTOINCREMENT,
        Name TEXT NOT NULL,
        NameKey TEXT NOT NULL UNIQUE,
        Kind TEXT NOT NULL DEFAULT 'person' CHECK (Kind IN ('self', 'person', 'document')))""",
    "Memory": """CREATE TABLE Memory (
        MemRef INTEGER NOT NULL PRIMARY KEY AUTOINCREMENT,
        OriginTime INTEGER NOT NULL,
        Memory TEXT NOT NULL,
        RecallCount INTEGER DEFAULT (1) NOT NULL,
        CompressedAt INTEGER,
        SpeakerID INTEGER REFERENCES Speakers(SpeakerID),
        PartnerID INTEGER REFERENCES Speakers(SpeakerID),
        CanonID INTEGER REFERENCES Canons(CanonID),
        Embedding BLOB)""",
    "Keywords": """CREATE TABLE Keywords (
        KeyID INTEGER NOT NULL PRIMARY KEY AUTOINCREMENT,
        Key TEXT NOT NULL UNIQUE,
        ToGist INTEGER)""",
    "Nodes": """CREATE TABLE Nodes (
        NodeID INTEGER NOT NULL PRIMARY KEY AUTOINCREMENT,
        MemID INTEGER,
        KeyID INTEGER NOT NULL,
        ImageID INTEGER)""",
    "Images": """CREATE TABLE Images (
        ImageID INTEGER NOT NULL PRIMARY KEY AUTOINCREMENT,
        DateCreated INTEGER NOT NULL,
        Filename TEXT NOT NULL,
        Description TEXT,
        CanonID INTEGER REFERENCES Canons(CanonID))""",
    "Dossiers": """CREATE TABLE Dossiers (
        MemRef INTEGER PRIMARY KEY REFERENCES Memory(MemRef),
        Goals TEXT, Limitations TEXT, Assets TEXT, Knowledge TEXT, Location TEXT,
        Relationships TEXT, KnowledgeGaps TEXT, UpdatedAt REAL, FullName TEXT)""",
    "Personas": """CREATE TABLE Personas (
        PersonaID INTEGER PRIMARY KEY AUTOINCREMENT,
        SpeakerID INTEGER NOT NULL REFERENCES Speakers(SpeakerID),
        CanonID INTEGER REFERENCES Canons(CanonID),
        GistRef INTEGER REFERENCES Memory(MemRef),
        LastPersonaRef INTEGER DEFAULT 0)""",
    "GistableModel": "CREATE TABLE GistableModel (Klass INTEGER NOT NULL, Feature TEXT NOT NULL, FeatureCount INTEGER NOT NULL, PRIMARY KEY (Klass, Feature))",
    "GistableModelMeta": """CREATE TABLE GistableModelMeta (
        Klass INTEGER PRIMARY KEY, DocCount INTEGER NOT NULL, TotalFeatureCount INTEGER NOT NULL, VocabSize INTEGER NOT NULL)""",
    # names that are the same subject inside one canon (CanonID NULL = open memory): JAMES and KEETON -> MARLOWE.
    # Written by the dream's deterministic alias detection; a wrong row can simply be deleted.
    "Aliases": """CREATE TABLE Aliases (
        CanonID INTEGER REFERENCES Canons(CanonID),
        KeyID INTEGER NOT NULL REFERENCES Keywords(KeyID),
        MainKeyID INTEGER NOT NULL REFERENCES Keywords(KeyID))""",
    "EmbeddingMeta": """CREATE TABLE EmbeddingMeta (
        Id INTEGER PRIMARY KEY CHECK (Id = 1),
        ModelName TEXT,
        Dim INTEGER)""",
    "DirectorCommands": "CREATE TABLE DirectorCommands (Phrase TEXT PRIMARY KEY, Action TEXT NOT NULL CHECK (Action IN ('COMMIT', 'RETRY')))",
    "DirectorInputLog": """CREATE TABLE DirectorInputLog (
        LogID INTEGER PRIMARY KEY AUTOINCREMENT, LoggedAt REAL, Speaker TEXT, InputText TEXT,
        LLMLabel TEXT, NBLabel TEXT, NBConfidence REAL, HandLabel TEXT)""",
    "DirectorInputModel": """CREATE TABLE DirectorInputModel (
        Klass TEXT NOT NULL, Feature TEXT NOT NULL, FeatureCount INTEGER NOT NULL, PRIMARY KEY (Klass, Feature))""",
    "DirectorInputModelMeta": """CREATE TABLE DirectorInputModelMeta (
        Klass TEXT PRIMARY KEY, DocCount INTEGER NOT NULL, TotalFeatureCount INTEGER NOT NULL, VocabSize INTEGER NOT NULL)""",
}
INDEXES = [

    "CREATE INDEX idx_memory_speaker ON Memory(SpeakerID)",
    "CREATE INDEX idx_memory_partner ON Memory(PartnerID)",
    "CREATE INDEX idx_memory_canon ON Memory(CanonID)",
    "CREATE INDEX idx_images_canon ON Images(CanonID)",
    "CREATE INDEX idx_nodes_mem ON Nodes(MemID)",
    "CREATE INDEX idx_nodes_key ON Nodes(KeyID)",
    "CREATE INDEX idx_nodes_image ON Nodes(ImageID)",
    # one persona per person per canon; IFNULL folds open chat (NULL) into one row too
    "CREATE UNIQUE INDEX idx_personas_scope ON Personas(SpeakerID, IFNULL(CanonID, 0))",
    "CREATE UNIQUE INDEX idx_aliases_scope ON Aliases(IFNULL(CanonID, 0), KeyID)",
]
DEFAULT_DIRECTOR_COMMANDS = {
    "COMMIT": ["done", "commit", "good", "looks good", "looks great", "that works", "that works for me", "perfect", "save it", "save",
               "commit it", "commit that", "approved", "approve", "accept", "accepted", "keep it", "keep that", "ship it", "lock it in",
               "lock it", "finalize", "finalize it", "that is good", "thats good", "thats perfect", "thats it", "good to go", "all good",
               "its good", "i like it", "love it", "yes", "yep", "ok", "okay", "next"],
    "RETRY": ["try again", "start over", "scrap that", "scrap it", "redo that", "redo it", "undo", "undo that", "discard", "discard it",
              "discard that", "throw it out", "throw that out", "never mind", "nevermind", "restart", "from scratch", "start again"],
}


# Lena's own row in Speakers. The key can't come from a typed name (names never contain "#").
SELF_NAME, SELF_KEY = "Lena", "#SELF"
# Columns a current database must have beyond its tables (checked at start-up).
REQUIRED_COLUMNS = {"Speakers": ["Kind"], "Memory": ["PartnerID", "CanonID"]}



# Tables a database may lack and still be upgraded in place by _upgrade (anything else missing means too old).
ADDED_LATER = {"GistableModel", "GistableModelMeta", "Aliases"}


class OutdatedDatabase(Exception):
    """MEM.db was made by an older Lena (before the Canons table) and is not upgraded automatically."""


def _create(conn):
    for ddl in TABLES.values():
        conn.execute(ddl)
    for ddl in INDEXES:
        conn.execute(ddl)
    conn.execute("INSERT INTO Speakers (Name, NameKey, Kind) VALUES (?, ?, 'self')", (SELF_NAME, SELF_KEY))
    for action, phrases in DEFAULT_DIRECTOR_COMMANDS.items():
        conn.executemany("INSERT OR IGNORE INTO DirectorCommands (Phrase, Action) VALUES (?, ?)", [(p, action) for p in phrases])
    conn.commit()


def _check(conn):
    if old_handle_format(conn):
        raise OutdatedDatabase(
            f"{DB_PATH.name} stores its handles in an older form (literal words). Move it out of the Lena folder and "
            f"start Lena again; she creates a fresh, empty database.")
    existing = {r[0] for r in conn.execute("SELECT name FROM sqlite_master WHERE type = 'table'")}
    missing = [t for t in TABLES if t not in existing and t not in ADDED_LATER]
    for table, columns in REQUIRED_COLUMNS.items():
        if table in existing:
            have = {r[1] for r in conn.execute(f"PRAGMA table_info({table})")}
            missing += [f"{table}.{c}" for c in columns if c not in have]
    if missing:
        raise OutdatedDatabase(
            f"{DB_PATH.name} was made by an older version of Lena (missing: {', '.join(missing)}). "
            f"Move it out of the Lena folder and start Lena again; she creates a fresh, empty database.")


def old_handle_format(conn):
    """Whether this database's tags are in an older form: literal words with a separate Stem column (or the
    interim GistKey / Meta structure). Lena refuses it; retag_handles.py converts our test databases."""
    kw = {r[1] for r in conn.execute("PRAGMA table_info(Keywords)")}
    mem = {r[1] for r in conn.execute("PRAGMA table_info(Memory)")}
    tables = {r[0] for r in conn.execute("SELECT name FROM sqlite_master WHERE type = 'table'")}
    return "Stem" in kw or "GistKey" in mem or "Meta" in tables


def _upgrade(conn, report):
    """Brings a database made by an earlier version of this structure up to date in place (no data is lost)."""
    have = {r[0] for r in conn.execute("SELECT name FROM sqlite_master WHERE type = 'table'")}
    for table in ("GistableModel", "GistableModelMeta"):
        if table not in have:
            conn.execute(TABLES[table])
    if "Aliases" not in have:
        conn.execute(TABLES["Aliases"])
        conn.execute("CREATE UNIQUE INDEX IF NOT EXISTS idx_aliases_scope ON Aliases(IFNULL(CanonID, 0), KeyID)")
        report.append("Aliases table added")
    if "ToGist" not in {r[1] for r in conn.execute("PRAGMA table_info(Keywords)")}:
        conn.execute("ALTER TABLE Keywords ADD COLUMN ToGist INTEGER")
        report.append("Keywords.ToGist added")
    conn.commit()
    if conn.execute("SELECT COUNT(*) FROM GistableModelMeta").fetchone()[0] == 0:
        import gistable
        seeded, learned = gistable.seed_database(conn)
        report.append(f"gist gate trained on {learned} labeled stems; {seeded} handles that never get a gist added")
    conn.commit()


def run(db_path=None, quiet=False):
    """Creates the database at db_path (default MEM.db next to this file) or checks an existing one. Returns a
    short report string. Raises OutdatedDatabase for a database from before the Canons table."""
    global DB_PATH
    if db_path is not None:
        DB_PATH = Path(db_path)
    say = (lambda *a: None) if quiet else print
    fresh = not DB_PATH.exists()
    conn = sqlite3.connect(DB_PATH, timeout=30)
    try:
        conn.execute("PRAGMA journal_mode=WAL")
        report = []
        if fresh:
            _create(conn)
            report.append(f"created a new MEM.db at {DB_PATH}")
            import gistable
            seeded, learned = gistable.seed_database(conn)
            report.append(f"{seeded} handles that never get a gist and a gist gate trained on {learned} labeled stems")
        else:
            _check(conn)
            _upgrade(conn, report)

        import llm_client, memory_index
        before = conn.execute("SELECT ModelName FROM EmbeddingMeta WHERE Id = 1").fetchone()
        if memory_index.sync_embedding_meta(conn):
            after = conn.execute("SELECT ModelName FROM EmbeddingMeta WHERE Id = 1").fetchone()
            if before and after and before[0] != after[0]:
                report.append("embedding model changed: all embeddings cleared; the dream cycle rebuilds them")
            missing = conn.execute("SELECT COUNT(*) FROM Memory WHERE Embedding IS NULL").fetchone()[0]
            if missing:
                report.append(f"{missing} memories have no embedding yet (the dream cycle's embedding pass fills them; or run db_setup.py --embed)")
        conn.commit()
        text = "; ".join(report) or "database up to date"
        say(f"[DB_SETUP] {text}")
        return text
    finally:
        conn.close()


if __name__ == "__main__":
    run()
    if "--embed" in sys.argv[1:]:
        import memory_index
        c = memory_index.get_connection()
        total = 0
        while True:
            n = memory_index.embed_missing(c, limit=400)
            total += n
            if not n:
                break
            print(f"[DB_SETUP] embedded {total} so far")
        c.close()
        print(f"[DB_SETUP] embedded {total} memories")
