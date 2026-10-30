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
keyword_utils.py - the handle rules, one copy

Single source of truth for the handles seed.py, recall.py, dream.py and persona.py all share: the stop list,
deterministic handle extraction (handles_of / handle_keywords), the stemmer (Porter2, porter2.py), part of
speech, and keyword-row creation (ensure_keyword). Storage and recall must make identical handles, or a
memory stops being found. (Canons are not keywords: they are rows in the Canons table.)

A handle is a symbol, not a word:
  * a name is stored as itself, in UPPER CASE: GRETA, ANTONIUS, FSB. A word is a name when it is capitalized
    in the middle of a sentence; at the start of a sentence, when the same text also capitalizes it mid-sentence
    or it is already a known name; in lower case, when it is a known name and not a dictionary word.
  * every other word is its Porter2 stem, in lower case: begin, begins, beginning, beginnings -> begin.
So SILVER (the otter girl) and silver (the colour), or WILL and will, are different handles.
"""

import re
import unicodedata
from pathlib import Path

import porter2


# ══════════════════════════════════════════════════════════════════════════════
# Deterministic handles
# ══════════════════════════════════════════════════════════════════════════════
# Handles are what recall and seeding share: every word of a text, minus stop words; names as names, the
# rest stemmed. No LLM is involved. The stop list below is the one used in every test of the deterministic
# recall (general English stop words, conversational filler, relative-time words and task verbs such as "write").

_HANDLE_STOP_TEXT = """
a about above after again against all also am an and any are aren't as at be because been before being below between both but by can can't cannot could
couldn't did didn't do does doesn't doing don't down during each few for from further had hadn't has hasn't have haven't having he he'd he'll he's her here
here's hers herself him himself his how how's i i'd i'll i'm i've if in into is isn't it it's its itself just let's me more most mustn't my myself no nor not
of off on once only or other ought our ours ourselves out over own same shan't she she'd she'll she's should shouldn't so some such than that that's the
their theirs them themselves then there there's these they they'd they'll they're they've this those through to too under until up very was wasn't we we'd
we'll we're we've were weren't what what's when when's where where's which while who who's whom why why's with won't would wouldn't you you'd you'll
you're you've your yours yourself yourselves
said say says tell told ask asked asking get got getting make made makes know knew think thought want wanted need needed going gone went come came take took
like really good nice great much many thing things something anything everything lot lots kind sort bit little way ways well yeah yes okay ok sure maybe
actually probably still even back right able try tried trying look looking looks feel feeling feels seem seems let put give gave given use used using help
helps sorry thanks thank please hey hello
write writing wrote word words draft compose suggest short
city name place time day today tonight tomorrow yesterday week year ago before after previous previously former old new now current currently recent
recently already ever never last first next another around always often sometimes lately
"""
HANDLE_STOPWORDS = frozenset(_HANDLE_STOP_TEXT.split())
_HANDLE_TOKEN = re.compile(r"[^\W\d_]+(?:['’\-][^\W\d_]+)*")   # letters of any alphabet; ' and - only between letters
_SPEAKER_PREFIX = re.compile(r"^([^:\n]{0,60}? said|I said):")
_SENTENCE_BREAK = set('\n.!?:"“”*—(')


def strip_speaker_prefix(text):
    """Removes a leading 'Name said:' / 'I said:' label so it never becomes a handle."""
    return _SPEAKER_PREFIX.sub("", text or "", count=1)


def _fold(word):
    """Lower-case, curly apostrophes straightened, accents removed: 'Yucatán' and 'Yucatan' are one word."""
    w = unicodedata.normalize("NFKD", word.lower().replace("’", "'"))
    return "".join(ch for ch in w if not unicodedata.combining(ch))


def stem(word):
    """The Porter2 stem of a word, lower case: begin, begins, beginning, beginnings -> begin. A comparison key, not a word."""
    return porter2.stem(_fold(word))


def is_name(handle):
    """Whether a stored handle is a name (names are upper case, stems lower case)."""
    return handle.isupper()


def _words(text):
    """
    (word, capitalized, starts a sentence) for every word of the text, folded and with a possessive 's removed.
    A hyphenated compound gives the whole and then each part, so 'cryo-pod' is found by 'cryo-pod', 'cryo' and 'pod'.
    """
    text = strip_speaker_prefix(text)
    for m in _HANDLE_TOKEN.finditer(text or ""):
        j = m.start()
        while j > 0 and text[j - 1].isspace() and text[j - 1] != "\n":   # any space, including the non-breaking
            j -= 1                                                       # spaces pasted web text puts after a period
        start = j == 0 or text[j - 1] in _SENTENCE_BREAK
        raw = m.group(0)
        pieces = [raw] + (raw.split("-") if "-" in raw else [])
        for k, piece in enumerate(pieces):
            w = _fold(piece)
            if w.endswith("'s"):
                w = w[:-2]
            if len(w) < 3 or w in HANDLE_STOPWORDS:
                continue
            yield w, piece[:1].isupper(), start and k <= 1


def handles_of(text, known=None):
    """
    The handles of a text, in order, repeats kept. known: optional callable(NAME) -> bool, whether that name is
    already a handle in the database (lets a sentence-initial or lower-case name be recognised).
    """
    words = list(_words(text))
    capitalized_mid = {w for w, cap, start in words if cap and not start}
    out = []
    for w, cap, start in words:
        name = w.upper()
        if cap and not start:
            out.append(name)
        elif cap:
            out.append(name if (w in capitalized_mid or (known is not None and known(name))) else stem(w))
        elif known is not None and not pos_of(w) and known(name):
            out.append(name)
        else:
            out.append(stem(w))
    return out


def handle_keywords(text, known=None):
    """
    The handles to store in Keywords/Nodes for a chunk of text: every distinct one. A tag is an association; a
    passage about three characters is tagged with all three, however often one of them is already mentioned.
    Telling common handles from rare ones is recall's job (rarity weighting there), not the storage's.
    """
    return list(dict.fromkeys(handles_of(text, known)))


def known_names(conn):
    """A known(NAME) callable for handles_of: is this name already a handle in the database? Cached per call site."""
    cache = {}
    def known(name):
        if name not in cache:
            cache[name] = conn.execute("SELECT 1 FROM Keywords WHERE Key = ?", (name,)).fetchone() is not None
        return cache[name]
    return known


# ══════════════════════════════════════════════════════════════════════════════
# Part of speech (deterministic, from a WordNet-derived word list)
# ══════════════════════════════════════════════════════════════════════════════
# data/pos_lexicon.tsv.gz lists every single-word WordNet 3.0 lemma with the parts of speech it can
# take (n noun, v verb, a adjective, r adverb). Inflected forms are reduced with WordNet's own
# suffix rules before lookup. A word not in the list (a name, a coinage, a typo) is "unknown".

_POS = None
_POS_FILE = Path(__file__).parent / "data" / "pos_lexicon.tsv.gz"
_MORPH = {
    "n": [("ses", "s"), ("xes", "x"), ("zes", "z"), ("ches", "ch"), ("shes", "sh"), ("men", "man"), ("ies", "y"), ("s", "")],
    "v": [("ies", "y"), ("es", "e"), ("es", ""), ("ed", "e"), ("ed", ""), ("ing", "e"), ("ing", ""), ("s", "")],
    "a": [("er", ""), ("est", ""), ("er", "e"), ("est", "e")],
    "r": [],
}


def _pos_table():
    global _POS
    if _POS is None:
        import gzip
        table = {}
        try:
            with gzip.open(_POS_FILE, "rt", encoding="utf-8") as f:
                for line in f:
                    if line and not line.startswith("#"):
                        word, _, tags = line.rstrip("\n").partition("\t")
                        table[word] = tags
        except OSError:
            print(f"[KEYWORDS][WARNING] {_POS_FILE} missing — every word is treated as unknown.")
        _POS = table
    return _POS


def pos_of(word):
    """Set of parts of speech ('n', 'v', 'a', 'r') a word can take; empty when the word is unknown."""
    w = (word or "").lower()
    table = _pos_table()
    found = set(table.get(w, ""))
    for pos, rules in _MORPH.items():
        if pos in found:
            continue
        for suffix, repl in rules:
            if w.endswith(suffix) and len(w) > len(suffix) + 1:
                base = w[: len(w) - len(suffix)] + repl
                if pos in table.get(base, ""):
                    found.add(pos)
                    break
    return found


def ensure_keyword(conn, key):
    """
    The one place a Keywords row is created. `key` is a handle as handles_of makes it (a lower-case stem or an
    upper-case name). Returns the KeyID (existing or new). Does not commit.
    """
    row = conn.execute("SELECT KeyID FROM Keywords WHERE Key = ?", (key,)).fetchone()
    if row:
        return row[0]
    return conn.execute("INSERT INTO Keywords (Key) VALUES (?)", (key,)).lastrowid
