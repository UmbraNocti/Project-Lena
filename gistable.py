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
gistable.py - which handles deserve a gist of their own. Deterministic except for one narrow case (step 5).

The pipeline (founding_candidates), for every handle in every canon (open memory counts as one):

  1. The mention floor: at least `gist_min_mentions` ordinary memories carry it, counted over all time.
  2. The say-so: Keywords.ToGist. A human's (or the shipped seed's, data/gist_labels.csv): 0 never, 1 yes. The
     judge's (step 5): -1 no, 2 yes. A human's word is never overwritten.
  3. Names pass: a name is stored upper case (keyword_utils), so this is just the handle's case.
  4. Every other handle (a lower-case stem) is asked of a naive Bayes classifier that sees its spelling
     (character 1-3-grams) and its part of speech. It is trained on hand labels (data/gist_labels.csv) and
     kept in plain SQL tables (GistableModel, GistableModelMeta), like the Director's input classifier. Its prior
     is the share of such words that deserve a gist (setting gist_prior), not the share in the training set,
     because the words it is asked about are nearly all 'no'.
  5. Only in the dream, and only for a handle with no say-so and no gist yet whose answer is uncertain: a name
     that is also an ordinary word (BISCUIT, CAPTAIN, SUNDAY), or a stem whose Bayes score is within
     `gist_judge_margin` of the line. A model reads a few of the sentences it appears in and answers whether it is
     a subject worth a standing note. The answer is stored in ToGist (-1 / 2), so each handle is asked once.
     A margin of 0 turns the judge off.
  The final set is the union of 3 and 4, minus anything 2 vetoed, plus anything 2 approved.

The labels file holds words as you judged them; they are turned into handles with the current rules when loaded,
so a change of stemmer does not invalidate them.
Retraining: python3 gistable.py --retrain
"""

import csv
import math
import sqlite3
import sys
from collections import Counter
from pathlib import Path

import keyword_utils as ku

ROOT = Path(__file__).parent.resolve()
LABELS_FILE = ROOT / "data" / "gist_labels.csv"
SOURCE_ORDER = {"original": 0, "round1": 1, "round2": 2, "round3": 3}


# ══════════════════════════════════════════════════════════════════════════════
# The classifier
# ══════════════════════════════════════════════════════════════════════════════

def features(handle, pos):
    """Character 1-3-grams of the handle (boundary-padded), plus its part of speech (weighted x3)."""
    p = "^" + handle.upper() + "$"
    grams = [p[i:i + n] for n in (1, 2, 3) for i in range(len(p) - n + 1)]
    return grams + [f"POS={pos or '?'}"] * 3


def pos_string(handle):
    return "".join(sorted(ku.pos_of(handle.lower()))) or "?"


def label_handle(word, cap="", pos=""):
    """The handle a labeled word is filed under: a name (upper case) when it was capitalized in its text (or, where that
    was not recorded, when it is not a dictionary word), else its stem."""
    w = ku._fold(word)
    if cap == "CAP" or (not cap and not ku.pos_of(w)):      # capitalization unknown: not a dictionary word reads as a name
        return w.upper()
    return ku.stem(w)


def train(conn, rows):
    """rows: [(handle, label 0/1)], stems only (names never reach the classifier). Writes the model tables."""
    counts = {0: Counter(), 1: Counter()}
    totals = {0: 0, 1: 0}
    docs = {0: 0, 1: 0}
    vocab = set()
    for handle, label in rows:
        f = features(handle, pos_string(handle))
        counts[label].update(f)
        totals[label] += len(f)
        docs[label] += 1
        vocab.update(f)
    if not docs[0] or not docs[1]:
        return 0
    conn.execute("DELETE FROM GistableModel")
    conn.execute("DELETE FROM GistableModelMeta")
    for klass in (0, 1):
        conn.executemany("INSERT INTO GistableModel (Klass, Feature, FeatureCount) VALUES (?, ?, ?)",
                         [(klass, f, n) for f, n in counts[klass].items()])
        conn.execute("INSERT INTO GistableModelMeta (Klass, DocCount, TotalFeatureCount, VocabSize) VALUES (?, ?, ?, ?)",
                     (klass, docs[klass], totals[klass], len(vocab)))
    conn.commit()
    return docs[0] + docs[1]


def load_model(conn):
    meta = {k: (d, t, v) for k, d, t, v in conn.execute("SELECT Klass, DocCount, TotalFeatureCount, VocabSize FROM GistableModelMeta")}
    if 0 not in meta or 1 not in meta:
        return None
    counts = {0: {}, 1: {}}
    for klass, feat, n in conn.execute("SELECT Klass, Feature, FeatureCount FROM GistableModel"):
        counts[klass][feat] = n
    return {"counts": counts, "totals": {k: meta[k][1] for k in meta}, "vocab": meta[0][2]}


def log_odds(model, handle):
    """log P(features | gistable) - log P(features | not): the evidence, before any prior."""
    s = 0.0
    c1, c0 = model["counts"][1], model["counts"][0]
    t1, t0, v = model["totals"][1], model["totals"][0], model["vocab"]
    for f in features(handle, pos_string(handle)):
        s += math.log((c1.get(f, 0) + 1) / (t1 + v)) - math.log((c0.get(f, 0) + 1) / (t0 + v))
    return s


def labeled_handles():
    """{handle: label} from the shipped hand labels. Where two labeled words share a handle, the later round wins;
    within one round, yes wins."""
    best = {}
    try:
        with open(LABELS_FILE, newline="", encoding="utf-8") as f:
            for r in csv.DictReader(f):
                h = label_handle(r["word"], r.get("cap", ""), r.get("pos", ""))
                if not h or len(h) < 2:
                    continue
                rank = (SOURCE_ORDER.get(r.get("source", ""), 0), int(r["label"]))
                if h not in best or rank > best[h][0]:
                    best[h] = (rank, int(r["label"]))
    except OSError:
        pass
    return {h: lab for h, (_, lab) in best.items()}


def seed_database(conn):
    """
    For a new database: every handle the shipped labels mark 0 gets ToGist = 0 (it never gets a gist), and the
    classifier is trained from the labeled stems. Returns (handles vetoed, stems trained on).
    """
    labels = labeled_handles()
    seeded = 0
    for h, label in labels.items():
        if label == 0:
            if conn.execute("SELECT 1 FROM Keywords WHERE Key = ?", (h,)).fetchone():
                conn.execute("UPDATE Keywords SET ToGist = 0 WHERE Key = ? AND ToGist IS NULL", (h,))
            else:
                conn.execute("INSERT INTO Keywords (Key, ToGist) VALUES (?, 0)", (h,))
            seeded += 1
    conn.commit()
    learned = train(conn, [(h, y) for h, y in labels.items() if not ku.is_name(h)])
    return seeded, learned


# ══════════════════════════════════════════════════════════════════════════════
# The pipeline
# ══════════════════════════════════════════════════════════════════════════════

def founding_candidates(conn, min_mentions=10, prior=0.043, judge=None, margin=0.0):
    """{(handle, CanonID)}: the handles that may FOUND a gist (see the module docstring). CanonID None is open memory.
    A handle that already has its gist is not this function's business; the dream always keeps refreshing those.
    judge: optional callable(conn, handle, cid) -> True / False / None (no answer), asked for the uncertain cases
    (step 5) when margin > 0; its answers are written to Keywords.ToGist."""
    counted = conn.execute(
        "SELECT k.Key, m.CanonID FROM Nodes n JOIN Keywords k ON k.KeyID = n.KeyID JOIN Memory m ON m.MemRef = n.MemID "
        "WHERE m.RecallCount >= 0 GROUP BY k.KeyID, m.CanonID HAVING COUNT(DISTINCT m.MemRef) >= ?", (min_mentions,)).fetchall()
    said = dict(conn.execute("SELECT Key, ToGist FROM Keywords WHERE ToGist IS NOT NULL"))
    model = load_model(conn)
    shift = math.log(prior / (1 - prior))
    out = set()
    def has_gist(key, cid):
        return conn.execute(
            "SELECT 1 FROM Memory m JOIN Nodes n ON n.MemID = m.MemRef JOIN Keywords k ON k.KeyID = n.KeyID "
            "WHERE k.Key = ? AND m.CanonID IS ? AND m.RecallCount < 0 AND m.Memory NOT LIKE 'Derived:%' "
            "AND (SELECT COUNT(*) FROM Nodes x WHERE x.MemID = m.MemRef) = 1", (key, cid)).fetchone() is not None
    for key, cid in counted:
        verdict = said.get(key)
        if verdict in (0, -1):
            continue
        if verdict in (1, 2):
            out.add((key, cid))
            continue
        name = ku.is_name(key)
        score = None if name or not model else log_odds(model, key) + shift
        uncertain = (name and ku.pos_of(key.lower()) is not None) or (score is not None and abs(score) < margin)
        if judge and margin > 0 and uncertain and not has_gist(key, cid):
            answer = judge(conn, key, cid)
            if answer is not None:
                said[key] = 2 if answer else -1
                conn.execute("UPDATE Keywords SET ToGist = ? WHERE Key = ? AND ToGist IS NULL", (said[key], key))
                conn.commit()
                if answer:
                    out.add((key, cid))
                continue
        if name or (score is not None and score > 0):
            out.add((key, cid))
    return out


if __name__ == "__main__":
    if "--retrain" in sys.argv[1:]:
        import memory_index
        conn = memory_index.get_connection()
        n = train(conn, [(h, y) for h, y in labeled_handles().items() if not ku.is_name(h)])
        print(f"[GISTABLE] model retrained on {n} labeled stems")
        conn.close()
    else:
        print(__doc__)
