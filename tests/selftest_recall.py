"""
Self-test for the deterministic recall stack. Builds a throw-away MEM.db in a temp directory with db_setup.py
(the real MEM.db is never opened), uses a fake bag-of-words embedder and a stub LLM, and checks: fresh-database
creation, seeding, speakers, canons, recall (both lanes, the canon lock, failure handling), the temporal anchor's
cue check, a full dream cycle (pass order, embeddings, person gists per canon and dossiers), the persona leak test,
the old-database refusal and embedder changes.
Run:  python3 tests/selftest_recall.py
"""
import io, os, re, sys, sqlite3, tempfile, time, hashlib, contextlib
from datetime import datetime
from pathlib import Path
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
import llm_client, memory_index, db_setup, seed, recall, persona, dream, temporal_anchor, director_helpers, keyword_utils, gistable
import image_describe, director_input_classifier   # opened lazily by seed / dream / the director

TMP = Path(tempfile.mkdtemp(prefix="lena_recall_test_"))
DB = TMP / "MEM.db"
for mod in (memory_index, seed, recall, dream, image_describe, director_input_classifier, director_helpers, db_setup):
    mod.DB_PATH = DB

# ---- fakes -------------------------------------------------------------------------------
MODEL = {"id": "fake|v1", "up": True}
def fake_vec(text, dim=64):
    v = [0.0] * dim
    for w in re.findall(r"[a-z]{3,}", text.lower()):
        v[int(hashlib.md5(w.encode()).hexdigest(), 16) % dim] += 1.0
    return v
def fake_embed(texts, kind="document", timeout=120):
    if not MODEL["up"]:
        raise ConnectionError("embedding server down")
    return [fake_vec(t) for t in texts]
llm_client.embedding_enabled = lambda: True
llm_client.embedding_config = lambda: {"backend": "fake", "model": "fake", "model_id": MODEL["id"], "doc_prefix": "", "query_prefix": "", "batch": 8}
llm_client.get_embeddings = fake_embed
CALLS = []
THINK_SEEN = []
FAIL_BUILD = False
def fake_completion(system_prompt="", user_text="", **kw):
    CALLS.append(system_prompt[:30])
    if "maintain a gist describing a person" in system_prompt and "as they are in" in (user_text or ""):
        return "On the Discord server Jordan is a moderator who posts bridge memes."
    if "maintain a gist describing a person" in system_prompt:
        return "Jordan prefers to be called Jo. Works as a structural engineer in Portland with partner Casey. Allergic to shellfish. Sister Mara lives in Tucson."
    if "Update the dossier" in system_prompt and "Discord server" in (user_text or ""):
        return "LOCATION: the Discord server\nGOALS: Keep the memes channel tidy"
    if "Update the dossier" in system_prompt:
        return "LOCATION: Portland\nGOALS: Run a half marathon"
    if "GIST_LOOP" in (user_text or ""):
        return "" if kw.get("thinking") else "Gist: a loop-free answer."
    if "Build the dossier" in system_prompt:
        if FAIL_BUILD: raise ConnectionError("model went away")
        THINK_SEEN.append(kw.get("thinking"))
        if "Write only these fields" in (user_text or ""):
            return "ASSETS: the Hasta Aeoli\nKNOWLEDGE: NONE\nRELATIONSHIPS: Nesrin"       # the thinking-off retry answers
        return "GOALS: Rebuild the family trade\nLOCATION: aboard the Hasta Aeoli\nLIMITATIONS: NONE"
    if "Year:" in (user_text or ""):
        return "2026-03-01"
    return "Jordan talked about family, the dog Biscuit and a shellfish allergy."
llm_client.get_completion = fake_completion
temporal_anchor.llm_client.get_completion = fake_completion
dream.get_completion = fake_completion
dream.is_backend_alive = lambda: True
passed = failed = 0
def check(name, cond, extra=""):
    global passed, failed
    if cond: passed += 1; print(f"  ok   {name}")
    else: failed += 1; print(f"  FAIL {name} {extra}")
def quiet(fn, *a, **k):
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        r = fn(*a, **k)
    return r, buf.getvalue()

# ---- fresh database ----------------------------------------------------------------------
print("db_setup")
report, _ = quiet(db_setup.run, DB)
conn = memory_index.get_connection()
tables = {r[0] for r in conn.execute("SELECT name FROM sqlite_master WHERE type='table'")}
check("fresh MEM.db created with every table", {"Memory", "Keywords", "Nodes", "Images", "Dossiers", "Speakers", "Canons", "Personas", "EmbeddingMeta", "DirectorCommands"} <= tables, report)
check("default Director commands seeded", conn.execute("SELECT COUNT(*) FROM DirectorCommands").fetchone()[0] > 40)
check("a new database ships the gist gate: a trained model, and words that never get a gist", conn.execute("SELECT COUNT(*) FROM GistableModel").fetchone()[0] > 500
      and conn.execute("SELECT COUNT(*) FROM Keywords WHERE ToGist = 0").fetchone()[0] > 500)
check("a new database has no Stem column, no GistKey column, no Meta table",
      "Stem" not in {r[1] for r in conn.execute("PRAGMA table_info(Keywords)")} and "GistKey" not in {r[1] for r in conn.execute("PRAGMA table_info(Memory)")}
      and "Meta" not in tables)
check("second run changes nothing", "up to date" in quiet(db_setup.run, DB)[0])

# ---- seeding -----------------------------------------------------------------------------
print("seed")
for q, a in [("My sister Mara is a pediatric nurse in Tucson.", "That sounds like a rewarding job."),
             ("I am allergic to shellfish, it was discovered at a wedding.", "Good to know, I will keep that in mind."),
             ("My dog Biscuit has three legs.", "Biscuit sounds lovely.")] * 6:
    quiet(seed.seed, q, a, speaker="Jordan")
quiet(seed.seed, "At dawn Antonius bought Nesrin at the slave market.", "The sun set over Ostia.", speaker="James", canon="Edge of Empire")
quiet(seed.seed, "By noon Antonius sailed from Ostia with Nesrin aboard.", "The harbour fell away behind them.", speaker="James", canon="Edge of Empire")
quiet(seed.seed, "METRO reports a road closure on I-40.", "NULL", speaker="Metro News")
check("seeding makes no LLM call", not CALLS, CALLS)
check("every memory embedded at storage time", conn.execute("SELECT COUNT(*) FROM Memory WHERE Embedding IS NULL").fetchone()[0] == 0)
jid = conn.execute("SELECT SpeakerID FROM Speakers WHERE NameKey='JORDAN'").fetchone()
check("speaker row for a real conversation partner", jid is not None)
check("a feed is attributed too (a speaker row), even with no reply", conn.execute("SELECT COUNT(*) FROM Speakers WHERE NameKey LIKE 'METRO%' AND Kind='person'").fetchone()[0] == 1)
LENA = conn.execute("SELECT SpeakerID FROM Speakers WHERE Kind='self'").fetchone()
check("Lena has her own speaker row", LENA is not None and conn.execute("SELECT NameKey FROM Speakers WHERE Kind='self'").fetchone()[0] == "#SELF")
check("no memory text carries a speaker label", conn.execute("SELECT COUNT(*) FROM Memory WHERE Memory LIKE '% said:%' OR Memory LIKE 'I said%'").fetchone()[0] == 0)
r = conn.execute("SELECT SpeakerID, PartnerID FROM Memory WHERE Memory = 'That sounds like a rewarding job.' LIMIT 1").fetchone()
check("Lena's reply is hers, answering the person (PartnerID)", r == (LENA[0], jid[0]), r)
quiet(seed.seed, "Narrator: The lighthouse keeper of Skerryvore found a green glass bottle wedged in the rocks.", "NULL",
      speaker="Lighthouse Log", canon="Lighthouse", document=True)
doc = conn.execute("SELECT m.Memory, s.Name FROM Memory m JOIN Speakers s ON s.SpeakerID = m.SpeakerID WHERE m.Memory LIKE '%Skerryvore%'").fetchone()
check("a document's name is its speaker (Speakers row) and its text carries no 'said:' label",
      doc == ("Narrator: The lighthouse keeper of Skerryvore found a green glass bottle wedged in the rocks.", "Lighthouse Log"), doc)
shown = quiet(recall.recall, "What did the keeper of Skerryvore find?", canon="Lighthouse", max_chars=2000)[0]
check("recall shows a document's memory with its name in front", "Lighthouse Log: Narrator: The lighthouse keeper" in shown, shown)
eoe = conn.execute("SELECT CanonID FROM Canons WHERE NameKey='EDGEOFEMPIRE'").fetchone()
check("a canon is a row in Canons", eoe is not None)
EOE = eoe[0] if eoe else None
check("canon memories keep their speaker and carry the CanonID", conn.execute(
    "SELECT COUNT(*) FROM Memory m JOIN Speakers s ON s.SpeakerID = m.SpeakerID WHERE m.Memory LIKE '%Antonius%' AND s.NameKey='JAMES' AND m.CanonID = ?", (EOE,)).fetchone()[0] == 2)
check("no canon keywords in Keywords", conn.execute("SELECT COUNT(*) FROM Keywords WHERE Key LIKE '~%' OR Key = 'EDGEOFEMPIRE'").fetchone()[0] == 0)
keys = {r[0] for r in conn.execute("SELECT Key FROM Keywords")}
check("handles are symbols: a name upper case, every other word its lower-case stem", {"TUCSON", "BISCUIT", "sister", "nurs", "allerg"} <= keys and "SISTER" not in keys, sorted(k for k in keys if k.lower() in ("tucson", "sister", "nurse", "nurs", "biscuit", "allergic", "allerg")))
check("every form of a word is one handle", set(keyword_utils.handle_keywords("They begin; she began beginnings; a beginning.")) == {"begin", "began"})
check("a sentence-initial word is a stem unless the text also uses it as a name", keyword_utils.handle_keywords("Silver shone.") == ["silver", "shone"]
      and keyword_utils.handle_keywords("Silver shone. Then Silver met Greta.") == ["SILVER", "shone", "met", "GRETA"])
check("a lower-case known name that is not a dictionary word is filed under the name; a dictionary word stays a stem",
      keyword_utils.handles_of("where is nesrin?", keyword_utils.known_names(conn)) == ["NESRIN"]
      and keyword_utils.handles_of("where is jordan?", keyword_utils.known_names(conn)) == ["jordan"])
zero = next(h for h, y in gistable.labeled_handles().items() if y == 0 and not keyword_utils.is_name(h))
check("the shipped 'never gist' handles are in a new database with ToGist 0", conn.execute("SELECT ToGist FROM Keywords WHERE Key = ?", (zero,)).fetchone()[0] == 0)
check("an ordinary new handle is left to the gate (ToGist NULL)", conn.execute("SELECT ToGist FROM Keywords WHERE Key = 'TUCSON'").fetchone()[0] is None)

# ---- recall ------------------------------------------------------------------------------
print("recall")
out, log = quiet(recall.recall, "What city does my sister live in?", speaker="Jordan", max_chars=4000)
check("finds the sister memory", "Tucson" in out)
check("recall shows the speaker from the table ('Jordan said: ...')", "Jordan said: My sister Mara is a pediatric nurse in Tucson." in out, out[:300])
lo = quiet(recall.recall, "rewarding job", speaker="Jordan", max_chars=4000)[0]
check("and Lena's own lines as 'I said: ...'", "I said: That sounds like a rewarding job." in lo, lo[:300])
check("recall makes no LLM call", not CALLS)
check("finds the allergy (adjective question)", "shellfish" in quiet(recall.recall, "What am I allergic to?", speaker="Jordan", max_chars=4000)[0])
check("canon memories stay out of an unrelated ordinary question", "Ostia" not in quiet(recall.recall, "What city does my sister live in?", speaker="Jordan", max_chars=2000)[0])
oc = quiet(recall.recall, "Continue the story", speaker="Jordan", canon="Edge of Empire", max_chars=4000)[0]
check("canon recall returns only canon memories", "Antonius" in oc and "Tucson" not in oc)
check("unknown canon returns nothing", quiet(recall.recall, "anything", canon="No Such Canon")[0] == "")
check("tuple shape with images and dossiers", len(quiet(recall.recall, "sister", speaker="Jordan", return_dossiers=True, include_images=True)[0]) == 3)
MODEL["up"] = False
out, log = quiet(recall.recall, "What city does my sister live in?", speaker="Jordan", max_chars=3000)
check("embedder down: recall continues on handles alone", "Tucson" in out and "Vector lane skipped" in log)
orig_open = memory_index.TableStore
memory_index.TableStore = lambda c: (_ for _ in ()).throw(RuntimeError("disk I/O error"))
try:
    quiet(recall.recall, "sister", speaker="Jordan"); raised = None
except recall.RecallUnavailable as e:
    raised = e.args[0]
check("both lanes broken: RecallUnavailable with reasons", raised and "disk I/O error" in raised[0], raised)
memory_index.TableStore = orig_open
MODEL["up"] = True

# ---- temporal anchor ---------------------------------------------------------------------
print("temporal anchor")
CALLS.clear()
t, _ = quiet(temporal_anchor.resolve_anchor, "What is my sister's name?")
check("no time cue: now, no LLM call", abs(t - time.time()) < 5 and not CALLS)
t, _ = quiet(temporal_anchor.resolve_anchor, "What should I cook?", previous="When I get home, what then?")
check("weak cue words in the previous message do not trigger a call", abs(t - time.time()) < 5 and not CALLS)
t, _ = quiet(temporal_anchor.resolve_anchor, "What did we eat there?", previous="Remember our trip last spring?")
check("time cue in the previous exchange: LLM resolves the anchor", datetime.fromtimestamp(t).strftime("%Y-%m-%d") == "2026-03-01" and CALLS)
CALLS.clear()

# ---- dream cycle -------------------------------------------------------------------------
print("dream")
conn.execute("UPDATE Memory SET OriginTime = OriginTime - 86400 * 2"); conn.commit()
conn.execute("UPDATE Memory SET Embedding = NULL WHERE MemRef % 3 = 0"); conn.commit()
cands = gistable.founding_candidates(conn, 10)
check("a name with fewer than gist_min_mentions memories and no gist is not gisted", not dream.worth_a_gist(conn, "ANTONIUS", EOE, cands))
_settings = memory_index.settings; memory_index.settings = lambda: {**_settings(), "gist_min_mentions": 1}   # the toy database is far too small for the real floor
_, log = quiet(dream.run)
memory_index.settings = _settings
check("a handle that already has its gist stays eligible below the floor", dream.worth_a_gist(conn, "ANTONIUS", EOE, cands))
order = [m for m in ["Pass 1 — Temporal", "Pass 2 — Embeddings", "Pass 3 — Open", "Pass 4 — Canon", "Pass 7 — Person gists"] if m in log]
derived = conn.execute("SELECT MemRef, CanonID FROM Memory WHERE Memory LIKE 'Derived:%'").fetchall()
in_canon = [r for r, c in derived if c == EOE]
check("temporal summaries are made per canon and belong to it", len(derived) >= 2 and len(in_canon) >= 1 and len(in_canon) < len(derived), derived)
check("dream runs temporal, embeddings, open gists, canon gists, then person gists", order == ["Pass 1 — Temporal", "Pass 2 — Embeddings", "Pass 3 — Open", "Pass 4 — Canon", "Pass 7 — Person gists"], order)
check("canon gists are made inside the canon, open gists outside it", conn.execute(
    "SELECT COUNT(*) FROM Memory WHERE Memory LIKE 'Gist:%' AND CanonID = ?", (EOE,)).fetchone()[0] >= 1
    and memory_index.gist_with_handles(conn, {"ANTONIUS"}, None) is None, log[-400:])
check("embedding pass filled every gap", conn.execute("SELECT COUNT(*) FROM Memory WHERE Embedding IS NULL").fetchone()[0] == 0)
JID = jid[0]
sp = conn.execute("SELECT GistRef FROM Personas WHERE SpeakerID = ? AND CanonID IS NULL", (JID,)).fetchone()
check("open-chat persona created with its gist", sp and sp[0] is not None)
row = conn.execute("SELECT Memory, RecallCount FROM Memory WHERE MemRef=?", (sp[0],)).fetchone()
check("person gist is a gist with exactly one handle (the name)", row[0].startswith("Gist: Jordan prefers") and row[1] < 0
      and [r[0] for r in conn.execute("SELECT k.Key FROM Nodes n JOIN Keywords k ON k.KeyID = n.KeyID WHERE n.MemID=?", (sp[0],))] == ["JORDAN"])
idents = memory_index.TableStore(conn).top_handles([sp[0]], 14)
check("a one-handle gist is mined from its text, not its single tag", "JORDAN" not in idents or len(idents) > 1, idents)
check("...so the person gist's handles lead to the rest of what is known", {"PORTLAND", "CASEY"} & idents, idents)
check("the name stays an ordinary handle (left to the gate)", conn.execute("SELECT ToGist FROM Keywords WHERE Key='JORDAN'").fetchone()[0] is None)
before = conn.execute("SELECT Memory FROM Memory WHERE MemRef=?", (sp[0],)).fetchone()[0]
quiet(seed.seed, "Jordan's coworker says Jordan once fixed the office printer with a paperclip.", "Ha, that tracks.", speaker="Casey")
quiet(seed.seed, "Casey told me Jordan is the best engineer on the team.", "Nice of Casey to say.", speaker="Casey")
quiet(dream.derive_gist, conn, {"JORDAN"}, "x", label="t")
check("the symbol passes never rewrite an adopted person gist", conn.execute("SELECT Memory FROM Memory WHERE MemRef=?", (sp[0],)).fetchone()[0] == before)
check("dossier created", conn.execute("SELECT Location FROM Dossiers WHERE MemRef=?", (sp[0],)).fetchone()[0] == "Portland")
out = quiet(recall.recall, "Help me write a note", speaker="Jordan", max_chars=3000)[0]
check("identity gist and dossier appear for any question", "prefers to be called Jo" in out and "[DOSSIERS]" in out)
check("the open-chat persona never appears inside a canon", "prefers to be called Jo" not in quiet(recall.recall, "Continue the story", speaker="Jordan", canon="Edge of Empire")[0])

for i in range(6):
    quiet(seed.seed, f"Captain Ishmael Vandervecken charted reef number {i} near the Sargasso shoals at dawn.", "NULL", speaker="Logbook", document=True)
_settings = memory_index.settings; memory_index.settings = lambda: {**_settings(), "gist_fold_chars": 150}
CALLS[:] = []
res, log = quiet(dream.derive_gist, conn, {"VANDERVECKEN"}, "x", label="fold")
memory_index.settings = _settings
check("a gist with more source text than one call holds is folded in over several calls",
      res is not None and len(CALLS) >= 3 and "over" in log and memory_index.gist_with_handles(conn, {"VANDERVECKEN"}) is not None, (len(CALLS), log[-200:]))

check("a word after a period and a non-breaking space starts a sentence (web text): not a name",
      not any(keyword_utils.is_name(h) for h in keyword_utils.handles_of("We sailed at dawn.\xa0 Perhaps the wind died.")))

# the gate's judge: only uncertain cases (a name that is also an ordinary word), asked once, answer stored
for i in range(12):
    quiet(seed.seed, f"Logbook entry {i}: the old Captain checked the barometer again before the watch.", "NULL", speaker="Logbook", document=True)
JUDGED = []
def _judge(conn, key, cid):
    JUDGED.append(key); return False
cands1 = gistable.founding_candidates(conn, 1, 0.043, judge=_judge, margin=1.0)
cands2 = gistable.founding_candidates(conn, 1, 0.043, judge=_judge, margin=1.0)
check("the judge is asked only about uncertain names (CAPTAIN), once, and its no is kept",
      "CAPTAIN" in JUDGED and JUDGED.count("CAPTAIN") == 1 and not any(k == "CAPTAIN" for k, _ in cands1 | cands2)
      and "VANDERVECKEN" not in JUDGED
      and conn.execute("SELECT ToGist FROM Keywords WHERE Key = 'CAPTAIN'").fetchone()[0] == -1, JUDGED[:10])

# aliases: two names that always appear together get one line in their gists, rewritten each cycle
_settings = memory_index.settings; memory_index.settings = lambda: {**_settings(), "gist_min_mentions": 1}
quiet(dream.derive_gist, conn, {"ISHMAEL"}, "x", label="alias")
for i in range(14):
    quiet(seed.seed, f"Captain Ishmael Vandervecken sounded the channel, pass {i}, before the tide turned.", "NULL", speaker="Logbook", document=True)
for i in range(2):
    quiet(seed.seed, f"Ishmael mended the sounding line, knot {i}, by lamplight.", "NULL", speaker="Logbook", document=True)
quiet(seed.seed, "Vandervecken alone logged a storm off Bermuda in the night.", "NULL", speaker="Logbook", document=True)
vand_ref = memory_index.gist_with_handles(conn, {"VANDERVECKEN"}, None)
vand_before = conn.execute("SELECT Memory FROM Memory WHERE MemRef = ?", (vand_ref,)).fetchone()[0]
SEEN_TEXT = []
_gc = dream.get_completion
def _capture(**k):
    SEEN_TEXT.append(k.get("user_text") or "")
    return "Ishmael is a reef charter of the Sargasso."
dream.get_completion = _capture
new_pairs = quiet(dream.refresh_aliases, conn)[0]
check("the dream records an alias pair in the Aliases table, most-mentioned name as the main one",
      new_pairs == [("VANDERVECKEN", "ISHMAEL", None)], new_pairs)
alias, main = new_pairs[0][0], new_pairs[0][1]
quiet(dream.fold_aliases_in, conn, new_pairs, "x")
check("a newly found alias's memories that never name the main one are read into the main gist once",
      any("storm off Bermuda" in t for t in SEEN_TEXT) and "reef charter" in conn.execute(
          "SELECT Memory FROM Memory WHERE MemRef = ?", (memory_index.gist_with_handles(conn, {main}, None),)).fetchone()[0])
check("the alias's own gist is left as it was, and the dream no longer folds into it",
      conn.execute("SELECT Memory FROM Memory WHERE MemRef = ?", (memory_index.gist_with_handles(conn, {alias}, None),)).fetchone()[0]
      == (vand_before if alias == "VANDERVECKEN" else conn.execute("SELECT Memory FROM Memory WHERE MemRef = ?", (memory_index.gist_with_handles(conn, {alias}, None),)).fetchone()[0])
      and quiet(dream.derive_gist, conn, {alias}, "x", label="alias")[0] is None)
conn.execute("UPDATE Memory SET OriginTime = OriginTime - 10 WHERE MemRef = ?", (memory_index.gist_with_handles(conn, {main}, None),)); conn.commit()
quiet(seed.seed, f"{alias.title()} sighted a whale at the edge of the shoals.", "NULL", speaker="Logbook", document=True)
SEEN_TEXT[:] = []
quiet(dream.derive_gist, conn, {main}, "x", label="main")
check("later memories that use only the alias are folded into the main gist", any("sighted a whale" in t for t in SEEN_TEXT))
dream.get_completion = _gc
out = quiet(recall.recall, f"What did {alias.title()} chart?", max_chars=4000)[0]
check("recall by the alias reaches the main name's gist", "reef charter" in out, out[:300])
changed = quiet(dream.annotate_aliases, conn)[0]
alias_text = conn.execute("SELECT Memory FROM Memory WHERE MemRef = ?", (memory_index.gist_with_handles(conn, {"ISHMAEL"}, None),)).fetchone()[0]
check("names that always appear together get an alias line with the full name", "Alias: " in alias_text and "are the same: Ishmael Vandervecken." in alias_text, alias_text[-200:])
check("the alias line is in both gists, once, and a second cycle changes nothing",
      conn.execute("SELECT Memory FROM Memory WHERE MemRef = ?", (memory_index.gist_with_handles(conn, {"VANDERVECKEN"}, None),)).fetchone()[0].count("Alias: ") == 1
      and quiet(dream.annotate_aliases, conn)[0] == 0)
memory_index.settings = _settings

# compression restates a big gist; a rewrite that loses half of it or its names is a summary and is rejected
ref = memory_index.gist_with_handles(conn, {"ISHMAEL"}, None)
long_body = "Gist: " + " ".join(f"Ishmael Vandervecken charted reef {i} near the Sargasso with Captain Ahab, filler filler filler." for i in range(60))
conn.execute("UPDATE Memory SET Memory = ?, CompressedAt = NULL WHERE MemRef = ?", (long_body + "\nAlias: Ishmael and Vandervecken are the same: Ishmael Vandervecken.", ref)); conn.commit()
_gc = dream.get_completion
dream.get_completion = lambda **k: "Ishmael charted some reefs."
quiet(dream.compress_gist, conn, ref, conn.execute("SELECT Memory FROM Memory WHERE MemRef=?", (ref,)).fetchone()[0], "x", label="c")
kept = conn.execute("SELECT Memory FROM Memory WHERE MemRef=?", (ref,)).fetchone()[0]
check("a rewrite too short to be a gist, or that drops a name the gist kept returning to, is rejected", kept.startswith(long_body))
dream.get_completion = lambda **k: " ".join(f"Ishmael Vandervecken charted reef {i} near the Sargasso." for i in range(40))
quiet(dream.compress_gist, conn, ref, kept, "x", label="c")
check("a long enough rewrite that drops a name the gist kept returning to (Ahab) is rejected",
      conn.execute("SELECT Memory FROM Memory WHERE MemRef=?", (ref,)).fetchone()[0] == kept)
dream.get_completion = lambda **k: " ".join(f"Ishmael Vandervecken charted reef {i}, Sargasso, with Captain Ahab." for i in range(40))
quiet(dream.compress_gist, conn, ref, kept, "x", label="c")
new = conn.execute("SELECT Memory FROM Memory WHERE MemRef=?", (ref,)).fetchone()[0]
check("a faithful compression is kept, and the alias line stays exactly once at the end",
      len(new) < len(kept) and new.startswith("Gist: Ishmael Vandervecken charted reef 0") and new.count("Alias: ") == 1 and new.split("\n")[-1].startswith("Alias: "), new[-120:])
dream.get_completion = _gc

# ---- the terminal log: one file, timestamps, nothing lost on the terminal, never in the way ------------
import consolelog, tempfile, os as _os
def _run_tee(path, enabled, writes, stdout=None):
    real_out, real_err = sys.stdout, sys.stderr
    term_out, term_err = io.StringIO() if stdout is None else stdout, io.StringIO()
    sys.stdout, sys.stderr = term_out, term_err
    try:
        consolelog.install(path, enabled)
        for stream, text in writes:
            (sys.stdout if stream == "out" else sys.stderr).write(text)
        sys.stdout.flush()
    finally:
        sys.stdout, sys.stderr = real_out, real_err
    return term_out, term_err
_d = tempfile.mkdtemp(); _log = _os.path.join(_d, "CONSOLE.txt")
_out, _err = _run_tee(_log, lambda: True, [("out", "[INFO] hello\n"), ("out", "\x1b[33mwarn\x1b[0m par"), ("out", "tial\n"), ("err", "Traceback boom\n")])
_txt = open(_log, encoding="utf-8").read()
check("the terminal gets exactly what was printed, color codes and all", _out.getvalue() == "[INFO] hello\n\x1b[33mwarn\x1b[0m partial\n" and _err.getvalue() == "Traceback boom\n")
check("the log gets each line with a timestamp, color codes stripped, partial writes joined, stderr included",
      "] [INFO] hello\n" in _txt and "] warn partial\n" in _txt and "] Traceback boom\n" in _txt and "\x1b" not in _txt and _txt.startswith("[20") and "=== Lena started" in _txt, _txt)
_log2 = _os.path.join(_d, "OFF.txt")
_run_tee(_log2, lambda: False, [("out", "secret\n")])
check("with logging off nothing is written (not even a header)", not _os.path.exists(_log2) or open(_log2).read() == "")
_flag = {"on": True}; _log3 = _os.path.join(_d, "LIVE.txt")
real_out, real_err = sys.stdout, sys.stderr
try:
    sys.stdout, sys.stderr = io.StringIO(), io.StringIO(); consolelog.install(_log3, lambda: _flag["on"])
    print("one"); _flag["on"] = False; print("two"); _flag["on"] = True; print("three"); sys.stdout.flush()
finally:
    sys.stdout, sys.stderr = real_out, real_err
_t3 = open(_log3).read()
check("switching the setting takes effect on the very next line", "] one\n" in _t3 and "two" not in _t3 and "] three\n" in _t3, _t3)
_t, _ = _run_tee(_os.path.join(_d, "no_such_dir", "x.txt"), lambda: True, [("out", "still fine\n")])
check("an unwritable log file never breaks the terminal", _t.getvalue() == "still fine\n")
_log4 = _os.path.join(_d, "BIG.txt"); open(_log4, "w").write("x" * (consolelog.MAX_BYTES + 10))
_run_tee(_log4, lambda: True, [("out", "fresh\n")])
check("a log past 10 MB is moved aside at startup, and the new one starts small",
      _os.path.exists(_os.path.join(_d, "BIG.old.txt")) and _os.path.getsize(_log4) < 1000)

# ---- personas per canon: the leak test ------------------------------------------------------
print("personas per canon")
for i in range(16):
    quiet(seed.seed, f"Mod note {i}: please keep bridge memes in the memes channel, thread {i}.", f"Got it, noted for thread {i}.", speaker="Jordan", canon="Discord")
CALLS.clear(); quiet(persona.run)
DIS = memory_index.canon_id(conn, "Discord")
dp = conn.execute("SELECT GistRef FROM Personas WHERE SpeakerID = ? AND CanonID = ?", (JID, DIS)).fetchone()
check("a conversation partner inside a canon gets a persona there", dp and dp[0] is not None and dp[0] != sp[0])
check("the canon persona is a gist inside that canon", conn.execute("SELECT CanonID FROM Memory WHERE MemRef=?", (dp[0],)).fetchone()[0] == DIS)
inside = quiet(recall.recall, "Who am I?", speaker="Jordan", canon="Discord", max_chars=4000)[0]
check("LEAK TEST: inside the canon you get the canon persona and nothing of the open-chat one",
      "moderator" in inside and "Tucson" not in inside and "prefers to be called Jo" not in inside and "Portland" not in inside, inside[:400])
check("open chat still uses the open-chat persona as identity", "prefers to be called Jo" in quiet(recall.recall, "Help me write a note", speaker="Jordan", max_chars=3000)[0])
for i in range(5):
    quiet(seed.seed, f"Reddit thread {i}: hot take about structural engineering salaries number {i}.", "Interesting point.", speaker="u/randomposter")
for i in range(20):
    quiet(seed.seed, f"Reddit thread {i}: another unanswered comment about bridges and load paths {i}.", "NULL", speaker="u/randomposter")
CALLS.clear(); quiet(persona.run)
rp = conn.execute("SELECT COUNT(*) FROM Personas p JOIN Speakers s ON s.SpeakerID = p.SpeakerID WHERE s.NameKey='U/RANDOMPOSTER'").fetchone()[0]
check("a passing commenter (5 replies, 20%) gets no person gist or dossier", rp == 0 and not CALLS, (rp, CALLS))
n_before = conn.execute("SELECT COUNT(*) FROM Memory WHERE Memory LIKE 'Gist: Jordan prefers%'").fetchone()[0]
quiet(persona.run)
check("a second person pass adds no second gist", conn.execute("SELECT COUNT(*) FROM Memory WHERE Memory LIKE 'Gist: Jordan prefers%'").fetchone()[0] == n_before == 1)

# ---- Director-style recall ---------------------------------------------------------------
print("director recall")
from keyword_utils import ensure_keyword
for i, txt in enumerate(["Antonius haggled with a Phoenician trader over amphorae of olive oil at the Ostia docks.",
                         "Nesrin counted the ledger twice and found the Phoenician had shorted them three amphorae.",
                         "The Hasta Aeoli sat low in the water, her triangular sail furled against the evening wind."]):
    quiet(seed.seed, txt, "NULL", speaker=None, canon="Edge of Empire")      # a Director commit
check("a Director commit is stored with no speaker and no 'said:' label", conn.execute(
    "SELECT SpeakerID, Memory FROM Memory WHERE Memory LIKE '%counted the ledger%'").fetchone() == (None, "Nesrin counted the ledger twice and found the Phoenician had shorted them three amphorae."))
g = conn.execute("INSERT INTO Memory (OriginTime, Memory, RecallCount, CanonID) VALUES (?, ?, -1, ?)", (int(time.time()), "Gist: Nesrin is a Syrian dyer's daughter who keeps Antonius' ledgers.", EOE)).lastrowid
conn.execute("INSERT INTO Nodes (MemID, KeyID) VALUES (?, ?)", (g, ensure_keyword(conn, "NESRIN")))
conn.execute("INSERT INTO Dossiers (MemRef, FullName, Location, UpdatedAt) VALUES (?, 'Nesrin', 'aboard the Hasta Aeoli', 0)", (g,))
conn.commit(); memory_index.embed_memory(conn, g)
tail = "The Hasta Aeoli sat low in the water, her triangular sail furled against the evening wind."
out, log = quiet(recall.recall, "Antonius orders the crew aboard", canon="Edge of Empire", max_chars=6000,
                 vector_queries=[("Antonius orders the crew aboard " + tail, 2), ("Phoenician trader amphorae", 1)],
                 context_text=tail, exclude_texts=[tail], entity_text="Antonius orders the crew aboard. Nesrin follows.")
check("text already in the prompt is never recalled again", "furled against the evening wind" not in out and "excluded 1" in log, log[-300:])
check("standing slice: the dossier of a character named in the input arrives", "aboard the Hasta Aeoli" in out and "[DOSSIERS]" in out)
check("several vector queries each contribute hits", "Phoenician" in out and "vector 3" in log, log[-300:])
check("no memory appears twice", len(re.findall(r"counted the ledger twice", out)) <= 1 and out.count("Syrian dyer") == 1)
img = conn.execute("INSERT INTO Images (DateCreated, Filename, Description, CanonID) VALUES (?, 'ship_plan.png', 'deck plan of the Hasta Aeoli', ?)", (int(time.time()), EOE)).lastrowid
for key in ("AEOLI", "HASTA"):
    conn.execute("INSERT INTO Nodes (ImageID, KeyID) VALUES (?, ?)", (img, ensure_keyword(conn, key)))
conn.commit()
res = quiet(recall.recall, "Write the next scene.", canon="Edge of Empire", include_images=True, return_dossiers=True, max_images=2,
            context_text="The Hasta Aeoli sat low in the water.")[0]
check("Director image lookup uses the newest text, so a ship plan is found for 'write the next scene'", any(i["filename"] == "ship_plan.png" for i in res[1]), res[1])
check("identity weight setting is 0.4 by default", memory_index.settings()["recall_identity_weight"] == 0.4)
check("a canon image stays out of another canon's recall", not any(i["filename"] == "ship_plan.png" for i in quiet(
    recall.recall, "Tell me about the Hasta Aeoli", canon="Discord", include_images=True, max_images=2)[0][1]))
check("open chat reads every canon (no lock)", any(i["filename"] == "ship_plan.png" for i in quiet(
    recall.recall, "Tell me about the Hasta Aeoli", include_images=True, max_images=2)[0][1]))

check("a dossier line of NONE counts as empty however the model punctuates it",
      director_helpers._parse_dossier_fields("GOALS: NONE.\nASSETS: **None**\nLOCATION: n/a\nKNOWLEDGE: Knows the ledger is false.")
      == {"Goals": None, "Assets": None, "Location": None, "Knowledge": "Knows the ledger is false."})

check("a gist call that comes back empty with thinking on is asked again with thinking off",
      dream.llm_gist("p", "GIST_LOOP memories") == "Gist: a loop-free answer.")

# ---- the Director's dossier button: a full rebuild ---------------------------------------------
print("dossier button")
director_helpers.DB_PATH = DB
old = memory_index.gist_with_handles(conn, {"ANTONIUS"}, EOE)
conn.execute("INSERT OR IGNORE INTO Dossiers (MemRef, FullName) VALUES (?, 'ANTONIUS')", (old,))
conn.execute("UPDATE Memory SET Memory = 'Gist: Antonius is a baker in Lyon.' WHERE MemRef = ?", (old,))
conn.execute("UPDATE Dossiers SET Assets = 'a stale bakery', Location = 'Lyon' WHERE MemRef = ?", (old,)); conn.commit()
res, log = quiet(director_helpers.create_or_update_dossier_on_demand, "Antonius", "Edge of Empire")
d = conn.execute("SELECT Goals, Location, Assets FROM Dossiers WHERE MemRef = ?", (old,)).fetchone()
check("button rebuilds the gist in place and from scratch", res["status"] == "ok" and res["mem_ref"] == old
      and "baker" not in conn.execute("SELECT Memory FROM Memory WHERE MemRef = ?", (old,)).fetchone()[0], (res, log[-300:]))
check("button rebuilds the dossier: new fields written, stale ones cleared", d[:2] == ("Rebuild the family trade", "aboard the Hasta Aeoli") and d[2] == "the Hasta Aeoli", d)
d2 = conn.execute("SELECT Limitations, Knowledge, Relationships, KnowledgeGaps FROM Dossiers WHERE MemRef = ?", (old,)).fetchone()
check("verification: fields the thinking pass left out are asked again, thinking off; NONE is an answer, not a gap",
      THINK_SEEN[-2:] == [True, False] and d2 == (None, None, "Nesrin", None) and "without ASSETS" in log, (THINK_SEEN, d2, log[-400:]))
FAIL_BUILD = True
quiet(director_helpers.create_or_update_dossier_on_demand, "Antonius", "Edge of Empire")
FAIL_BUILD = False
check("a failed dossier build leaves the dossier as it was",
      conn.execute("SELECT Location FROM Dossiers WHERE MemRef = ?", (old,)).fetchone()[0] == "aboard the Hasta Aeoli")
check("button on an unknown name aborts without touching anything",
      quiet(director_helpers.create_or_update_dossier_on_demand, "Nobodyever", "Edge of Empire")[0]["status"] == "aborted")

# ---- handles: every tag kept, compounds split, accents folded --------------------------------
print("handles")
import keyword_utils, retag_handles
long_chunk = ("Antonius, Nesrin and Bran counted twenty amphorae of wine near the harbor while the Gauls stacked cheese "
              "on the quay, and Rooper watched the tide turn against the mole beside Pollux and Briccus.")
check("a rich passage keeps every distinct handle (no cap)", len(keyword_utils.handle_keywords(long_chunk)) == len(set(keyword_utils.handles_of(long_chunk))) > 12)
check("a very common name is still tagged on a long passage (a known name at the start of a sentence too)",
      {"ANTONIUS", "NESRIN", "BRAN"} <= set(keyword_utils.handle_keywords(long_chunk, keyword_utils.known_names(conn))))
check("hyphenated compound: the whole and each part", {"cryo-pod", "cryo", "pod"} <= set(keyword_utils.handle_keywords("The cryo-pod hummed.")))
check("accents fold and a trailing apostrophe is dropped", set(keyword_utils.handle_keywords("We followed Jules' trip to Yucatán")) >= {"JULES", "YUCATAN"})
sq = quiet(seed.seed, "Her cryo-pod hummed in the cold bay near the Celestún coast.", "Noted.", speaker="James", canon="Edge of Empire")
check("recall finds 'cryo-pod' from the words 'cryo pod'", "cryo-pod hummed" in quiet(recall.recall, "Where is the cryo pod?", canon="Edge of Empire", max_chars=2000)[0])

# ---- one-time upgrades and embedder changes ----------------------------------------------
print("old databases and embedder changes")
OLD = TMP / "old_MEM.db"
oc_ = sqlite3.connect(OLD); oc_.execute("CREATE TABLE Memory (MemRef INTEGER PRIMARY KEY, OriginTime INTEGER, Memory TEXT, RecallCount INTEGER)"); oc_.commit(); oc_.close()
try:
    quiet(db_setup.run, OLD); refused = None
except db_setup.OutdatedDatabase as e:
    refused = str(e)
db_setup.DB_PATH = DB
check("a database from before the Canons table is refused with a clear message", refused and "older version of Lena" in refused, refused)
OLD2 = TMP / "old2_MEM.db"
o2 = sqlite3.connect(OLD2)
for name, ddl in db_setup.TABLES.items():
    o2.execute(ddl.replace(",\n        Kind TEXT NOT NULL DEFAULT 'person' CHECK (Kind IN ('self', 'person', 'document'))", ""))
o2.commit(); o2.close()
try:
    quiet(db_setup.run, OLD2); refused = None
except db_setup.OutdatedDatabase as e:
    refused = str(e)
db_setup.DB_PATH = DB
check("a database missing a newer column (Speakers.Kind) is refused too", refused and "Speakers.Kind" in refused, refused)
MODEL["id"] = "fake|v2"
report, _ = quiet(db_setup.run, DB)
check("a changed embedding model clears the stored embeddings at start-up", conn.execute("SELECT COUNT(*) FROM Memory WHERE Embedding IS NOT NULL").fetchone()[0] == 0, report)
check("recall keeps working while they are rebuilt", "Tucson" in quiet(recall.recall, "What city does my sister live in?", speaker="Jordan", max_chars=3000)[0])
check("embed_missing rebuilds them", memory_index.embed_missing(conn, limit=10000) == conn.execute("SELECT COUNT(*) FROM Memory").fetchone()[0])

# ---- a database with the older handle form is converted in place ----------------------------------
print("conversion")
OLD3 = TMP / "old3_MEM.db"
quiet(db_setup.run, OLD3)
c3 = sqlite3.connect(OLD3)
c3.execute("ALTER TABLE Keywords ADD COLUMN Stem TEXT")                       # the older form: literal words, a Stem column
def oldkey(k):
    row = c3.execute("SELECT KeyID FROM Keywords WHERE Key = ?", (k,)).fetchone()
    return row[0] if row else c3.execute("INSERT INTO Keywords (Key, Stem) VALUES (?, ?)", (k, k)).lastrowid
c3.execute("INSERT INTO Memory (OriginTime, Memory, RecallCount) VALUES (1, 'Gist: Geralt hunts monsters for coin.', -1)")
gref = c3.execute("SELECT MAX(MemRef) FROM Memory").fetchone()[0]
c3.execute("INSERT INTO Nodes (MemID, KeyID) VALUES (?, ?)", (gref, oldkey("GERALT")))                       # a gist: its subject
c3.execute("INSERT INTO Memory (OriginTime, Memory, RecallCount) VALUES (2, 'Then Geralt hunts a griffin near the village of Blaviken at dusk.', 1)")
oref = c3.execute("SELECT MAX(MemRef) FROM Memory").fetchone()[0]
c3.execute("INSERT INTO Nodes (MemID, KeyID) VALUES (?, ?)", (oref, oldkey("GERALT")))                       # the old 12-handle rule kept few
c3.execute("UPDATE Keywords SET ToGist = 0 WHERE KeyID = ?", (oldkey("HUNTS"),))
c3.commit(); c3.close()
try:
    quiet(db_setup.run, OLD3); refused3 = None
except db_setup.OutdatedDatabase as e:
    refused3 = str(e)
check("Lena refuses a database with the older handle form", refused3 and "older form" in refused3, refused3)
rep = quiet(retag_handles.main, OLD3)[1]
c3 = sqlite3.connect(OLD3)
tags = lambda ref: sorted(r[0] for r in c3.execute("SELECT k.Key FROM Nodes n JOIN Keywords k ON k.KeyID = n.KeyID WHERE n.MemID = ?", (ref,)))
check("an older database is converted in place: no Stem column, a copy kept", "Stem" not in {r[1] for r in c3.execute("PRAGMA table_info(Keywords)")}
      and (TMP / "old3_MEM.db.before_handles").exists(), rep)
check("...an ordinary memory is tagged with every handle of its text", {"GERALT", "BLAVIKEN", "griffin", "hunt", "villag", "dusk"} <= set(tags(oref)), tags(oref))
check("...a gist keeps exactly its subject and is still found by it", tags(gref) == ["GERALT"] and memory_index.gist_with_handles(c3, {"GERALT"}, None) == gref, tags(gref))
check("...a human's ToGist carries over to the new handle", c3.execute("SELECT ToGist FROM Keywords WHERE Key = 'hunt'").fetchone()[0] == 0)
check("...and the gist gate is retrained", c3.execute("SELECT COUNT(*) FROM GistableModel").fetchone()[0] > 500)
c3.close()
db_setup.DB_PATH = DB

# ---- problems Lena explains herself ---------------------------------------------------------
print("errors")
import requests
llm_client._SETTINGS["default_backend"] = "127.0.0.1:8099"
msg = llm_client.describe_failure(requests.exceptions.ConnectionError("refused"))
check("an unreachable model gets a plain message with the address and the fix", "nothing is answering at 127.0.0.1:8099" in msg and "Start your model server" in msg, msg)
bad = type("R", (), {"status_code": 401, "text": "no"})()
check("a refused API key is named as such", "API key" in llm_client.describe_failure(requests.exceptions.HTTPError("401", response=bad)))
saved_get = llm_client.get_embeddings
def broken_embed(texts, kind="document", timeout=120):
    raise requests.exceptions.ConnectionError("embedding server down")
llm_client.get_embeddings = broken_embed
recall.LAST_PROBLEMS.clear()
mem = quiet(recall.recall, "What city does my sister live in?", speaker="Jordan", max_chars=3000)[0]
check("an embedding outage is reported to the caller while handles still answer", recall.LAST_PROBLEMS and "vector search failed" in recall.LAST_PROBLEMS[0] and "Tucson" in mem, recall.LAST_PROBLEMS)
llm_client.get_embeddings = saved_get
conn.close()
print(f"\n{passed} passed, {failed} failed")
sys.exit(1 if failed else 0)
