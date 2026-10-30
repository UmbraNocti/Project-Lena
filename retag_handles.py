"""
Converts a database's tags to the current handle rules, from the text, with no LLM:
  * a handle is a symbol: a name in UPPER CASE (GRETA), every other word its Porter2 stem in lower case (begin);
  * every ordinary memory and summary is tagged with every handle of its own text;
  * a gist keeps exactly what it is about (its subject), turned into the new handle;
  * image tags and the human ToGist say-so are carried over to the new handles (where several old words become one
    handle, a 0 wins over a 1).
Names are decided over the whole database first: a word capitalized in the middle of a sentence anywhere is a name,
so a sentence-initial or lower-case use of it is filed under the name too.

A development tool for our own test databases (the story benchmark), which were built in the older form: literal words
with a separate Stem column, or the interim Memory.GistKey / Meta structure. Lena itself refuses such a database.
usage: python3 retag_handles.py <db> [--no-backup]
"""
import sqlite3
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
import keyword_utils as ku

_LABEL_PREFIXES = ("Gist:", "GIST:", "Derived:")


def _body(text):
    t = text or ""
    while t.startswith(_LABEL_PREFIXES):
        t = t.split(":", 1)[1].lstrip()
    return t


def _has(conn, table):
    return conn.execute("SELECT 1 FROM sqlite_master WHERE type = 'table' AND name = ?", (table,)).fetchone() is not None


def convert(conn):
    """Rebuilds Keywords and Nodes in the current form (see the module docstring). Returns a one-line report."""
    t0 = time.time()
    kw_cols = {r[1] for r in conn.execute("PRAGMA table_info(Keywords)")}
    mem_cols = {r[1] for r in conn.execute("PRAGMA table_info(Memory)")}
    old_key = {kid: key for kid, key in conn.execute("SELECT KeyID, Key FROM Keywords")}
    old_stem_to_keys = {}
    if "Stem" in kw_cols:
        for key, st in conn.execute("SELECT Key, Stem FROM Keywords"):
            old_stem_to_keys.setdefault(st, []).append(key)
    old_togist = conn.execute("SELECT Key, ToGist FROM Keywords WHERE ToGist IS NOT NULL").fetchall() if "ToGist" in kw_cols else []
    rows = conn.execute(f"SELECT MemRef, Memory, RecallCount{', GistKey' if 'GistKey' in mem_cols else ''} FROM Memory").fetchall()
    images = conn.execute("SELECT ImageID, Description FROM Images").fetchall()
    old_tags = {}
    for mid, kid in conn.execute("SELECT MemID, KeyID FROM Nodes WHERE MemID IS NOT NULL"):
        old_tags.setdefault(mid, []).append(old_key.get(kid))
    old_image_tags = {}
    for iid, kid in conn.execute("SELECT ImageID, KeyID FROM Nodes WHERE ImageID IS NOT NULL"):
        old_image_tags.setdefault(iid, []).append(old_key.get(kid))

    # 1. names: capitalized mid-sentence anywhere in the text
    names = set()
    def collect(text):
        for w, cap, start in ku._words(text):
            if cap and not start:
                names.add(w.upper())
    for row in rows:
        if row[2] >= 0 or (row[1] or "").startswith("Derived:"):
            collect(_body(row[1]))
    for _, desc in images:
        collect(desc or "")
    known = names.__contains__

    def convert_word(word):
        w = ku._fold(word or "")
        return w.upper() if w.upper() in names else ku.stem(w)

    # 2. the new tags
    new_tags, new_image_tags = {}, {}
    gists = 0
    for row in rows:
        ref, text, rc = row[0], row[1], row[2]
        if rc >= 0 or (text or "").startswith("Derived:"):
            new_tags[ref] = ku.handle_keywords(_body(text), known)
        else:
            subject = []
            if len(row) > 3 and row[3]:                                  # interim structure: GistKey held old stems
                for part in row[3].split("+"):
                    subject += old_stem_to_keys.get(part, [part])[:1]
            else:
                subject = [k for k in old_tags.get(ref, []) if k]
            new_tags[ref] = list(dict.fromkeys(convert_word(k) for k in subject))
            gists += 1
    for iid, _ in images:
        new_image_tags[iid] = list(dict.fromkeys(convert_word(k) for k in old_image_tags.get(iid, []) if k))
    said = {}
    for key, verdict in old_togist:
        h = convert_word(key)
        said[h] = 0 if 0 in (verdict, said.get(h)) else verdict

    # 3. rebuild Keywords and Nodes
    conn.execute("DROP TABLE IF EXISTS Keywords_new")
    conn.execute("CREATE TABLE Keywords_new (KeyID INTEGER NOT NULL PRIMARY KEY AUTOINCREMENT, Key TEXT NOT NULL UNIQUE, ToGist INTEGER)")
    every = dict.fromkeys(h for tags in list(new_tags.values()) + list(new_image_tags.values()) for h in tags)
    every.update(dict.fromkeys(said))
    conn.executemany("INSERT INTO Keywords_new (Key, ToGist) VALUES (?, ?)", [(h, said.get(h)) for h in every])
    ids = dict(conn.execute("SELECT Key, KeyID FROM Keywords_new"))
    conn.execute("DELETE FROM Nodes")
    conn.executemany("INSERT INTO Nodes (MemID, KeyID) VALUES (?, ?)", [(ref, ids[h]) for ref, tags in new_tags.items() for h in tags])
    conn.executemany("INSERT INTO Nodes (ImageID, KeyID) VALUES (?, ?)", [(iid, ids[h]) for iid, tags in new_image_tags.items() for h in tags])
    conn.execute("DROP TABLE Keywords")
    conn.execute("ALTER TABLE Keywords_new RENAME TO Keywords")
    if "GistKey" in mem_cols:
        conn.execute("DROP INDEX IF EXISTS idx_memory_gistkey")
        conn.execute("ALTER TABLE Memory DROP COLUMN GistKey")
    conn.execute("DROP TABLE IF EXISTS Meta")
    # the gate's model learned from the old form of the handles: retrain it, and pre-flag the shipped 'no' handles
    for table in ("GistableModel", "GistableModelMeta"):
        if _has(conn, table):
            conn.execute(f"DELETE FROM {table}")
    conn.commit()
    if _has(conn, "GistableModel"):
        import gistable
        gistable.seed_database(conn)
    tags = conn.execute("SELECT COUNT(*) FROM Nodes").fetchone()[0]
    return (f"handles converted ({len(rows) - gists} memories and summaries retagged from their text, {gists} gists kept "
            f"their subject; {len(ids)} handles, {tags} tags, {len(names)} names; {time.time() - t0:.0f}s)")


def main(path, backup=True):
    path = Path(path)
    if backup:
        copy = path.with_name(path.name + ".before_handles")
        if not copy.exists():
            src = sqlite3.connect(path); dst = sqlite3.connect(copy); src.backup(dst); dst.close(); src.close()
    conn = sqlite3.connect(path)
    for table, ddl in (("GistableModel", "CREATE TABLE GistableModel (Klass INTEGER NOT NULL, Feature TEXT NOT NULL, FeatureCount INTEGER NOT NULL, PRIMARY KEY (Klass, Feature))"),
                       ("GistableModelMeta", "CREATE TABLE GistableModelMeta (Klass INTEGER PRIMARY KEY, DocCount INTEGER NOT NULL, TotalFeatureCount INTEGER NOT NULL, VocabSize INTEGER NOT NULL)")):
        if not _has(conn, table):
            conn.execute(ddl)
    if "ToGist" not in {r[1] for r in conn.execute("PRAGMA table_info(Keywords)")}:
        conn.execute("ALTER TABLE Keywords ADD COLUMN ToGist INTEGER")
    print(convert(conn))
    conn.close()


if __name__ == "__main__":
    main(sys.argv[1], backup="--no-backup" not in sys.argv)
