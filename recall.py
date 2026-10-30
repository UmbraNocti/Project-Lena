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
recall.py — deterministic memory recall. Called by controller.py before each LLM query; makes no LLM call.

For one message:
  1. IDENTITY  the speaker's person gist in this canon (their persona's adopted gist, or the gist whose only
               handle is their name) and its dossier.
  2. VECTOR    embed the message, take the top K memories (default 6) by cosine over ALL memories
               (verbatim, gists, derived). No embedder, or a failure, simply frees the budget for step 3.
  3. HANDLES   the message's own handles (content words, stop words dropped, stemmed) at full weight;
               the rarest handles of the vector hits and of the identity gist at low weight.
               weight(h) = rarity(h) * (1 + 0.6 * (sources - 1)),  rarity = ln((N + 1) / (df + 1))
               score(m)  = sum(weights shared) / (distinct handles in m) ^ 0.25
                           * (0.6 + 0.4 * exp(-days from the temporal anchor / 120))
  4. FILL      vector hits first, then the ranked memories, until max_chars; a dossier rides with its gist.

Canon is a hard gate: with a canon active only memories whose CanonID is that canon are eligible (the
person gist included, so a person's open-chat gist never shows up inside a canon). With no canon, everything is.
If both lanes fail (or the database is unreachable) RecallUnavailable is raised and controller.py
answers in amnesia mode with a diagnosis. Tuning values: memory_index.settings() (settings.txt).
The long-lived index and embedding caches live in memory_index.py; tables are created by db_setup.py.
"""

import math
import re
import sqlite3
import time
from collections import defaultdict
from datetime import datetime
from pathlib import Path

import llm_client
import memory_index
from keyword_utils import handles_of, handle_keywords, strip_speaker_prefix, known_names

# ── Paths ─────────────────────────────────────────────────────────────────────
ROOT = Path(__file__).parent.resolve()
DB_PATH = ROOT / "MEM.db"

# ── Settings ──────────────────────────────────────────────────────────────────
TOP_N = 10000                        # default character budget for recalled memories
IMAGE_REASONING_MIN_SCORE = 0.02     # recalled images below this percentile are dropped; low on purpose —
                                     # the model reasons over them, the show-gate decides what the user sees


# Problems that reduced (but did not stop) the last recall, for controller.py to pass on to the model.
LAST_PROBLEMS = []


class RecallUnavailable(Exception):
    """The memory system cannot recall at all. args[0] is a list of human-readable reasons."""


# ══════════════════════════════════════════════════════════════════════════════
# Database and canon
# ══════════════════════════════════════════════════════════════════════════════

def get_connection():
    conn = sqlite3.connect(DB_PATH)
    conn.execute("PRAGMA journal_mode=WAL")
    return conn


def check_connection():
    """Called by controller on startup to verify the DB is reachable and has the tables recall reads (read-only)."""
    try:
        if not DB_PATH.exists():
            return False, f"MEM.db not found at {DB_PATH}"
        conn = get_connection()
        for table in ("Keywords", "Memory", "Nodes", "Dossiers", "Speakers", "Canons", "Personas", "EmbeddingMeta"):
            conn.execute(f"SELECT 1 FROM {table} LIMIT 1")
        conn.close()
        return True, "ok"
    except Exception as e:
        return False, str(e)


def lookup_keyword_ids(conn, handles):
    """KeyIDs of these handles (the ones that exist), for fetch_images_for_keywords."""
    handles = list(dict.fromkeys(handles))
    found = []
    for i in range(0, len(handles), 500):
        part = handles[i:i + 500]
        found += [r[0] for r in conn.execute(f"SELECT KeyID FROM Keywords WHERE Key IN ({','.join('?' * len(part))})", part)]
    return found


# ══════════════════════════════════════════════════════════════════════════════
# Images, dossiers, formatting
# ══════════════════════════════════════════════════════════════════════════════

def build_dossier_text(dossier):
    """
    Multi-line dossier block, headed by FullName, one indented line per
    populated field. Returns "" for a None dossier so it's always safe to
    call unconditionally and concatenate the result.
    """
    if not dossier:
        return ""
    fields = [
        ("Goals", dossier["goals"]),
        ("Limitations", dossier["limitations"]),
        ("Assets", dossier["assets"]),
        ("Knowledge", dossier["knowledge"]),
        ("Location", dossier["location"]),
        ("Relationships", dossier["relationships"]),
        ("Knowledge Gaps", dossier["knowledge_gaps"]),
    ]
    lines = [dossier["full_name"] or "(unnamed)"]
    lines += [f"    {label}: {val}" for label, val in fields if val]
    return "\n".join(lines)


def fetch_images_for_keywords(conn, keyword_ids, cid=None, canon_locked=False):
    """
    Pulls images sharing a keyword, via Nodes.ImageID
    instead of Nodes.MemID. Returns list of dicts, same shape pattern as
    memory dicts so score_images() can reuse the same scoring math.

    canon_locked: only images whose CanonID is cid — images participate
    in the same canon lock as text memories.
    """
    if not keyword_ids:
        return []

    placeholders = ",".join("?" * len(keyword_ids))
    params = list(keyword_ids)

    canon_clause = ""
    if canon_locked:
        canon_clause = "AND i.CanonID = ?"
        params.append(cid)

    query = f"""
        SELECT
            i.ImageID,
            i.Filename,
            i.DateCreated,
            COUNT(n.KeyID) AS MatchCount,
            (SELECT COUNT(*) FROM Nodes n2 WHERE n2.ImageID = i.ImageID) AS TotalKeywordCount,
            i.Description
        FROM Images i
        JOIN Nodes n ON i.ImageID = n.ImageID
        WHERE n.KeyID IN ({placeholders})
        {canon_clause}
        GROUP BY i.ImageID
    """
    cursor = conn.execute(query, params)
    rows = cursor.fetchall()

    images = []
    for row in rows:
        images.append({
            "image_id":             row[0],
            "filename":             row[1],
            "origin_time":          row[2],   # reuse "origin_time" name so
                                               # the same scoring math works
            "match_count":          row[3],
            "memory_keyword_count": row[4],   # "total keywords on this
                                               # record", regardless of type
            "description":          row[5],   # supplemental, may be None
        })
    return images


def score_images(images, total_keywords, anchor_time=None):
    """
    Keyword-overlap score with distance decay (the original recall formula, kept for images) —
    images have no RecallCount sign concept, so there's no "derived" image.
    """
    anchor = anchor_time if anchor_time is not None else time.time()
    seconds_per_day = 86400
    max_possible_score = total_keywords if total_keywords > 0 else 1

    for img in images:
        days_since   = abs(anchor - img["origin_time"]) / seconds_per_day
        match_count  = img["match_count"]
        img_kw_count = img["memory_keyword_count"]

        concentration_ratio   = total_keywords / img_kw_count if img_kw_count > 0 else 1
        effective_match_count = match_count * concentration_ratio
        base_decay             = 1 / (1 + days_since)

        if effective_match_count > 0:
            score = effective_match_count * (base_decay ** (1 / effective_match_count))
        else:
            score = 0

        img["score"]      = score
        img["percentile"] = round(score / max_possible_score, 4)

    images.sort(key=lambda i: i["score"], reverse=True)
    return images


def increment_recall_counts(conn, mem_refs):
    """Increment recall count magnitude, preserving sign (derived stays negative)."""
    for ref in mem_refs:
        conn.execute("""
            UPDATE Memory
            SET RecallCount = CASE
                WHEN RecallCount < 0 THEN RecallCount - 1
                ELSE RecallCount + 1
            END
            WHERE MemRef = ?
        """, (ref,))
    conn.commit()


def format_timestamp(unix_epoch):
    return datetime.fromtimestamp(unix_epoch).strftime("%Y-%m-%d")


def format_memory_blocks(memories):
    """
    Three sections, in order: Dossiers -> Gists -> Memories.

    Dossier and gist are not mutually exclusive — an entity with a
    dossier still shows up in [GISTS] too, since the two hold different
    information (dossier: derived, future-facing state; gist: durable
    descriptive/sensory facts). [GISTS] vs [MEMORIES] is still the
    original factual/derived split.
    """
    dossiers = [m for m in memories if m["has_dossier"]]
    gists    = [m for m in memories if m["is_derived"]]
    factual  = [m for m in memories if not m["is_derived"]]

    lines = []

    if dossiers:
        lines.append("[DOSSIERS]")
        for m in dossiers:
            lines.append(m["dossier_text"])
        lines.append("")

    if gists:
        lines.append("[GISTS]")
        for m in gists:
            date_str = format_timestamp(m["origin_time"])
            lines.append(f"({m['percentile']*100:.2f}% | {date_str}) {m['memory']}")

    if factual:
        if lines and lines[-1] != "":
            lines.append("")
        lines.append("[MEMORIES]")
        for m in factual:
            date_str = format_timestamp(m["origin_time"])
            lines.append(f"({m['percentile']*100:.2f}% | {date_str}) {m['memory']}")

    return "\n".join(lines)


def _shape_recall_return(formatted, best_images, dossier_records, include_images, return_dossiers):
    """Centralizes recall()'s four possible return shapes so every return
    point in the function doesn't have to repeat the branching."""
    if include_images and return_dossiers:
        return formatted, best_images, dossier_records
    if include_images:
        return formatted, best_images
    if return_dossiers:
        return formatted, dossier_records
    return formatted


# ══════════════════════════════════════════════════════════════════════════════
# Recall
# ══════════════════════════════════════════════════════════════════════════════

def _dossier_rows(conn, refs):
    out = {}
    if not refs:
        return out
    q = ",".join("?" * len(refs))
    for row in conn.execute(
            f"SELECT MemRef, FullName, Goals, Limitations, Assets, Knowledge, Location, Relationships, KnowledgeGaps "
            f"FROM Dossiers WHERE MemRef IN ({q})", list(refs)):
        out[row[0]] = {"full_name": row[1], "goals": row[2], "limitations": row[3], "assets": row[4], "knowledge": row[5],
                       "location": row[6], "relationships": row[7], "knowledge_gaps": row[8]}
    return out


def _empty_result(include_images, return_dossiers):
    if include_images and return_dossiers:
        return "", [], []
    if include_images:
        return "", []
    if return_dossiers:
        return "", []
    return ""


def _memory_dict(mem, ref, pct, dossier, speakers):
    m = mem[ref]
    return {
        "mem_ref": ref,
        "memory": memory_index.render(m["text"], m["speaker"], speakers),   # "James said: ...", "I said: ..."
        "origin_time": m["t"],
        "recall_count": -1 if m["gist"] else 1,
        "percentile": round(pct, 4),
        "is_derived": m["gist"],
        "dossier": dossier,
        "dossier_text": build_dossier_text(dossier) if dossier else "",
        "has_dossier": dossier is not None,
    }


def _normalize(text):
    return re.sub(r"\s+", " ", text or "").strip().lower()


def _mentions(name, text):
    """Whether an entity's name (or, for a multi-word name, its first word) appears as a whole word in text."""
    name = (name or "").strip()
    if not name:
        return False
    if re.search(r"\b" + re.escape(name) + r"\b", text, re.I):
        return True
    first = name.split()[0]
    return " " in name and len(first) >= 3 and re.search(r"\b" + re.escape(first) + r"\b", text, re.I) is not None


def recall(user_message: str, last_user: str = "", last_ai: str = "",
           max_chars: int = TOP_N, anchor_time: float = None,
           include_images: bool = False, canon: str = None,
           return_dossiers: bool = False, max_images: int = 1,
           speaker: str = None, vector_queries=None, context_text: str = None,
           exclude_texts=None, entity_text: str = None):
    """
    Entry point called by controller before each LLM query. Returns the formatted memory block
    (or "" when nothing matched), with the recalled images and/or dossier records appended to a
    tuple when asked for (see _shape_recall_return).

    max_chars: character budget for the recalled memories (identity and standing dossiers come on top).
    anchor_time: unix epoch the recency decay is measured from (the temporal anchor); now if None.
    canon: when set, only memories of that canon are eligible, and the speaker's person gist is the one for
        this canon. A canon that has never been seeded returns nothing.
    speaker: the person speaking; their person gist and dossier (for this canon) are always included.
    vector_queries: [(text, k), ...] searched in order, k hits each; default [(user_message, recall_vector_hits)].
        Director passes several: its input with the end of the story, then earlier story blocks.
    context_text: text that should shape recall without being the question (Director: the story tail or
        the draft being revised); its rarest handles join the handle lane at the mined weight.
    exclude_texts: texts already in the prompt (Director: the raw story tail, the draft). A verbatim memory
        whose text is contained in one of them is never returned, so nothing is shown twice.
    entity_text: text searched for named entities; every dossier whose FullName appears in it is included
        with its gist, on top of the budget (Director's standing slice: the characters in play).

    Raises RecallUnavailable when the memory system itself is broken (database unreachable, or both the
    vector and the handle lanes failed). Controller turns that into amnesia mode with a diagnosis.
    """
    cfg = memory_index.settings()
    problems = []
    if not DB_PATH.exists():
        raise RecallUnavailable([f"MEM.db not found at {DB_PATH}"])
    try:
        conn = get_connection()
    except Exception as e:
        raise RecallUnavailable([f"MEM.db could not be opened: {e}"])

    try:
        locked = bool(memory_index.canon_key(canon))
        cid = memory_index.canon_id(conn, canon) if locked else None
        if locked and cid is None:
            print(f"[RECALL] Canon '{canon}' has no memories seeded yet — returning empty.")
            return _empty_result(include_images, return_dossiers)

        try:
            store = memory_index.TableStore(conn)      # associations come from Keywords and Nodes
        except Exception as e:
            raise RecallUnavailable([f"reading the Memory table failed: {e}"])
        anchor = anchor_time if anchor_time is not None else time.time()
        speakers = memory_index.speaker_table(conn)
        shown_len = lambda r: len(memory_index.render(store.mem[r]["text"], store.mem[r]["speaker"], speakers))   # what the model sees
        eligible = lambda r: r in store.mem and (not locked or store.mem[r]["canon"] == cid)

        known = known_names(conn)

        # 0. memories already in the prompt (Director: the raw tail, the draft) are never repeated. Checked only on
        #    the memories about to be taken (their text is fetched for the budget anyway), never across the table.
        excluded = set()
        hay = [_normalize(t) for t in (exclude_texts or []) if t]
        def is_excluded(r):
            if not hay or r in excluded:
                return r in excluded
            m = store.mem[r]
            if m["gist"]:
                return False
            body = _normalize(strip_speaker_prefix(m["text"]))
            if len(body) >= 30 and any(body in h for h in hay):
                excluded.add(r)
                return True
            return False

        # 1. identity: the speaker's person gist for this canon (open chat: their open-chat gist)
        ident = memory_index.find_identity(conn, speaker, cid)
        if ident is not None and not eligible(ident):
            ident = None

        # 1b. standing entities: dossiers of everyone named in entity_text (Director)
        standing = []
        if entity_text:
            for ref, name in conn.execute("SELECT MemRef, FullName FROM Dossiers WHERE FullName IS NOT NULL").fetchall():
                if ref != ident and eligible(ref) and _mentions(name, entity_text):
                    standing.append(ref)
            standing = standing[:int(cfg["recall_standing_entities"])]
        taken = set(standing) | ({ident} if ident is not None else set())

        # 2. vector lane: k hits per query. Skipped without an embedder; a failure only frees budget.
        vec_hits = []
        if llm_client.embedding_enabled():
            try:
                memory_index.EMBEDDINGS.refresh(conn)
                queries = vector_queries or [(user_message, int(cfg["recall_vector_hits"]))]
                queries = [(q, k) for q, k in queries if q and q.strip() and k > 0]
                qvecs = llm_client.get_embeddings([q for q, _ in queries], kind="query")
                used_v = 0
                for (q, k), qv in zip(queries, qvecs):
                    n = 0
                    ranked = memory_index.EMBEDDINGS.rank(qv)
                    for pos, (r, sim) in enumerate(ranked):
                        if n >= k:
                            break
                        if pos % 200 == 0:
                            store.mem.prefetch([x for x, _ in ranked[pos:pos + 200]])
                        if r in taken or not eligible(r) or is_excluded(r):
                            continue
                        size = shown_len(r) + 12
                        if used_v + size > max_chars:
                            continue
                        vec_hits.append((r, max(0.0, min(1.0, sim))))
                        taken.add(r)
                        used_v += size
                        n += 1
            except Exception as e:
                problems.append(f"vector search failed (embedding backend {llm_client.embedding_config()['backend']}): {e}")
                print(f"[RECALL][WARNING] Vector lane skipped: {e}")
                LAST_PROBLEMS.append(problems[-1])

        # 3. handle lane. Full weight: the question's handles. Identity weight: handles of the person gist and of
        #    the standing entities' gists. Mined weight: handles of the vector hits and of context_text.
        scored = []
        try:
            sources = {"Q": set(handles_of(user_message, known))}
            if vec_hits:
                sources["A"] = store.top_handles([r for r, _ in vec_hits][:5], int(cfg["recall_mined_per_hit"]))
            people = ([ident] if ident is not None else []) + standing
            if people:
                sources["I"] = store.top_handles(people, int(cfg["recall_mined_identity"]))
            if context_text:
                ch = set(handles_of(context_text, known))
                store.prime(ch)
                sources["C"] = set(sorted(ch, key=lambda h: -store.rarity(h))[:int(cfg["recall_context_handles"])])
            seen_in = defaultdict(set)
            for name, hs in sources.items():
                for h in hs:
                    seen_in[h].add(name)
            # an alias also walks its main name, in this canon only: "Keeton" reaches what is filed under MARLOWE
            main_of = memory_index.alias_groups(conn, cid)[0]
            for h in [h for h in seen_in if h in main_of]:
                seen_in[main_of[h]] |= seen_in[h]
            store.prime(seen_in)
            weights = {}
            for h, src in seen_in.items():
                rar = store.rarity(h)
                if rar <= float(cfg["recall_min_rarity"]):
                    continue
                w = rar * (1 + float(cfg["recall_source_bonus"]) * (len(src) - 1))
                if "Q" not in src:
                    w *= float(cfg["recall_identity_weight"]) if "I" in src else float(cfg["recall_mined_weight"])
                weights[h] = w
            max_possible = sum(weights.values()) or 1.0
            floor, scale, lexp = float(cfg["recall_recency_floor"]), float(cfg["recall_recency_days"]), float(cfg["recall_length_exponent"])
            for r, (raw, t, handle_count) in store.candidates(weights, cid, locked).items():
                if r in taken or not eligible(r):
                    continue
                days = abs(anchor - t) / 86400
                s = raw / (max(1, handle_count) ** lexp) * (floor + (1 - floor) * math.exp(-days / scale))
                scored.append((s / max_possible, r))
            scored.sort(reverse=True)
        except Exception as e:
            problems.append(f"handle recall failed: {e}")
            print(f"[RECALL][WARNING] Handle lane failed: {e}")
            if not vec_hits:
                raise RecallUnavailable(problems)

        # 4. fill the budget: vector hits first, then the ranked memories; a dossier rides with its gist.
        #    The identity gist and the standing entities (with their dossiers) come on top of the budget.
        doss = _dossier_rows(conn, [r for r, _ in vec_hits] + [r for _, r in scored[:400]] + standing + ([ident] if ident is not None else []))
        def footprint(ref):
            d = doss.get(ref)
            return shown_len(ref) + (len(build_dossier_text(d)) if d else 0)
        chosen, used = [], 0
        for r, sim in vec_hits:
            chosen.append((r, sim))
            used += footprint(r)
        for s, r in scored:
            if is_excluded(r):
                continue
            fp = footprint(r)
            if used + fp > max_chars:
                continue
            chosen.append((r, min(1.0, s)))
            used += fp
        memories = []
        for r in ([ident] if ident is not None else []) + standing:
            memories.append(_memory_dict(store.mem, r, 1.0, doss.get(r), speakers))
        for r, pct in chosen:
            memories.append(_memory_dict(store.mem, r, pct, doss.get(r), speakers))

        best_images = []
        if include_images:
            try:
                # the message's words, plus the context text's in Director Mode ("write the next scene" names nothing)
                words = handle_keywords(user_message + "\n" + (context_text or ""), known)
                kw_ids = lookup_keyword_ids(conn, words)
                images = fetch_images_for_keywords(conn, kw_ids, cid=cid, canon_locked=locked)
                images = score_images(images, len(words), anchor_time=anchor_time)
                best_images = [i for i in images if i["percentile"] >= IMAGE_REASONING_MIN_SCORE][:max_images]
            except Exception as e:
                print(f"[RECALL][WARNING] Image lookup failed: {e}")

        if not memories:
            print("[RECALL] No memories to return.")
            return _shape_recall_return("", best_images, [], include_images, return_dossiers)

        dossier_records = []
        if return_dossiers:
            for m in memories:
                if m["has_dossier"]:
                    dossier_records.append({"mem_ref": m["mem_ref"], **m["dossier"]})
        increment_recall_counts(conn, [m["mem_ref"] for m in memories])
        summary = ", ".join(f"MemRef={m['mem_ref']}({m['percentile'] * 100:.0f}%)" for m in memories)
        print(f"[RECALL] {len(memories)} memories returned (vector {len(vec_hits)}, identity {'yes' if ident else 'no'}, standing {len(standing)}, excluded {len(excluded)}): {summary}")
        return _shape_recall_return(format_memory_blocks(memories), best_images, dossier_records, include_images, return_dossiers)
    finally:
        conn.close()


if __name__ == "__main__":
    import sys
    if len(sys.argv) >= 2:
        user_msg  = sys.argv[1]
        last_user = sys.argv[2] if len(sys.argv) > 2 else ""
        last_ai   = sys.argv[3] if len(sys.argv) > 3 else ""
        result = recall(user_msg, last_user, last_ai)
        print("\n── Final memory block returned to controller ──")
        print(result if result else "(empty)")
    else:
        print("Usage: python3 recall.py \"user message\" [\"last user msg\"] [\"last ai msg\"]")
