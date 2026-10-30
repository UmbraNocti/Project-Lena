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
director_input_classifier.py - Hand-rolled multinomial naive bayes for the
Director Mode input type (DIALOGUE / ACTION / DIRECTION / SCENE / PROSE).
Same philosophy as gistable.py: stdlib only, deterministic,
trained on James's own labels, model stored as plain SQL tables.

Tables (created in MEM.db, not in code):

  DirectorInputLog (LogID, LoggedAt, Speaker, InputText,
                    LLMLabel, NBLabel, NBConfidence, HandLabel)
      One row per classified Director input. LLMLabel is what the LLM
      decision tree said (NULL when the NB answered live), NBLabel /
      NBConfidence are this classifier's verdict if a model was trained
      at the time. HandLabel is written ONLY by a human (DBeaver).

  DirectorInputModel (Klass, Feature, FeatureCount)
  DirectorInputModelMeta (Klass, DocCount, TotalFeatureCount, VocabSize)

train_model() reads HandLabel and nothing else — never LLMLabel or
NBLabel — so the classifier can't learn from its own (or the LLM's)
guesses. Same guarantee as Keywords.ToGist: only a human writes it, and the gist gate never writes it.

Rollout (controller setting director_classifier_mode):
  shadow - LLM tree decides; NB verdict is logged next to it. Default.
  live   - NB decides when NBConfidence >= director_classifier_min_confidence,
           otherwise the LLM tree runs as before (the edge-case fallback).

Retrain after a labeling session:  python3 director_input_classifier.py
"""

import math
import re
import sqlite3
import time
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).parent.resolve()
DB_PATH = ROOT / "MEM.db"

LABELS = ("DIALOGUE", "ACTION", "DIRECTION", "SCENE", "PROSE")

_WORD = re.compile(r"[a-z']+")


def extract_features(text):
    """Word unigrams + bigrams, plus a few structural tokens that words
    alone don't capture (quotes, length bucket, first-person opener)."""
    text = text or ""
    words = _WORD.findall(text.lower().replace("’", "'"))
    feats = list(words)
    feats += [f"{a}_{b}" for a, b in zip(words, words[1:])]
    if '"' in text or "“" in text or "”" in text:
        feats.append("__QUOTE__")
    if words and words[0] == "i":
        feats.append("__STARTS_I__")
    feats.append(f"__LEN{min(len(text) // 100, 10)}__")
    return feats


def train_model(conn):
    rows = conn.execute(
        "SELECT InputText, HandLabel FROM DirectorInputLog WHERE HandLabel IN (%s)"
        % ",".join("?" * len(LABELS)), LABELS,
    ).fetchall()
    if not rows:
        print("[DIRECTOR_NB] No hand-labeled rows found — nothing to train on.")
        return False

    counts = {k: Counter() for k in LABELS}
    totals = Counter()
    docs = Counter()
    vocab = set()
    for text, label in rows:
        feats = extract_features(text)
        counts[label].update(feats)
        totals[label] += len(feats)
        docs[label] += 1
        vocab.update(feats)

    conn.execute("DELETE FROM DirectorInputModel")
    conn.execute("DELETE FROM DirectorInputModelMeta")
    for k in LABELS:
        if not docs[k]:
            continue
        conn.executemany(
            "INSERT INTO DirectorInputModel (Klass, Feature, FeatureCount) VALUES (?, ?, ?)",
            [(k, f, c) for f, c in counts[k].items()],
        )
        conn.execute(
            "INSERT INTO DirectorInputModelMeta (Klass, DocCount, TotalFeatureCount, VocabSize) "
            "VALUES (?, ?, ?, ?)", (k, docs[k], totals[k], len(vocab)),
        )
    conn.commit()
    print(f"[DIRECTOR_NB] Trained on {len(rows)} rows, vocab_size={len(vocab)}, "
          f"per-class: {dict(docs)}")
    return True


def load_model(conn):
    meta = {k: (d, t, v) for k, d, t, v in conn.execute(
        "SELECT Klass, DocCount, TotalFeatureCount, VocabSize FROM DirectorInputModelMeta")}
    if len(meta) < 2:
        return None
    feature_counts = {k: {} for k in meta}
    for k, f, c in conn.execute("SELECT Klass, Feature, FeatureCount FROM DirectorInputModel"):
        feature_counts[k][f] = c
    total_docs = sum(d for d, _, _ in meta.values())
    return {
        "priors": {k: d / total_docs for k, (d, _, _) in meta.items()},
        "class_totals": {k: t for k, (_, t, _) in meta.items()},
        "vocab_size": next(iter(meta.values()))[2],
        "feature_counts": feature_counts,
    }


def predict(model, text):
    """Returns (label, confidence) with confidence = the winning class's
    normalized posterior. Naive bayes posteriors run overconfident, so
    pick the live threshold from logged agreement data, not from intuition."""
    feats = extract_features(text)
    vocab = model["vocab_size"]
    log_scores = {}
    for k, prior in model["priors"].items():
        fc = model["feature_counts"][k]
        total = model["class_totals"][k]
        s = math.log(prior)
        for f in feats:
            s += math.log((fc.get(f, 0) + 1) / (total + vocab))
        log_scores[k] = s
    top = max(log_scores.values())
    exp = {k: math.exp(v - top) for k, v in log_scores.items()}
    norm = sum(exp.values())
    label = max(exp, key=exp.get)
    return label, exp[label] / norm


def log_decision(conn, speaker, text, llm_label, nb_label, nb_confidence):
    conn.execute(
        "INSERT INTO DirectorInputLog (LoggedAt, Speaker, InputText, LLMLabel, NBLabel, NBConfidence) "
        "VALUES (?, ?, ?, ?, ?, ?)",
        (time.time(), speaker, text, llm_label, nb_label, nb_confidence),
    )
    conn.commit()


if __name__ == "__main__":
    conn = sqlite3.connect(DB_PATH)
    conn.execute("PRAGMA journal_mode=WAL")
    try:
        train_model(conn)
    finally:
        conn.close()
