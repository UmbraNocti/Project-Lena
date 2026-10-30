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
porter2.py - the Porter2 ("Snowball English") stemmer, plain Python, no dependencies.

Martin Porter's revision of his 1980 algorithm, as published at snowballstem.org (BSD-licensed algorithm
description). Deterministic. It reduces a word to a comparison key, not to a real word: begin, begins, beginning
and beginnings are all "begin"; organize and organization are "organ"; argue and argued are "argu".

stem(word) takes a lower-case word and returns its stem in lower case.
"""

_VOWELS = "aeiouy"
_DOUBLES = ("bb", "dd", "ff", "gg", "mm", "nn", "pp", "rr", "tt")
_LI_ENDINGS = "cdeghkmnrt"
_EXCEPTIONS = {
    "skis": "ski", "skies": "sky", "dying": "die", "lying": "lie", "tying": "tie",
    "idly": "idl", "gently": "gentl", "ugly": "ugli", "early": "earli", "only": "onli", "singly": "singl",
    "sky": "sky", "news": "news", "howe": "howe", "atlas": "atlas", "cosmos": "cosmos", "bias": "bias", "andes": "andes",
}
_EXCEPTIONS_AFTER_1A = {"inning", "outing", "canning", "herring", "earring", "proceed", "exceed", "succeed"}

_STEP2 = (
    ("ization", "ize"), ("ational", "ate"), ("fulness", "ful"), ("ousness", "ous"), ("iveness", "ive"),
    ("tional", "tion"), ("biliti", "ble"), ("lessli", "less"), ("entli", "ent"), ("ation", "ate"),
    ("alism", "al"), ("aliti", "al"), ("ousli", "ous"), ("iviti", "ive"), ("fulli", "ful"),
    ("enci", "ence"), ("anci", "ance"), ("abli", "able"), ("izer", "ize"), ("ator", "ate"),
    ("alli", "al"), ("bli", "ble"), ("ogi", "og"), ("li", ""),
)
_STEP3 = (
    ("ational", "ate"), ("tional", "tion"), ("alize", "al"), ("icate", "ic"), ("iciti", "ic"),
    ("ative", ""), ("ical", "ic"), ("ness", ""), ("ful", ""),
)
_STEP4 = ("ement", "ance", "ence", "able", "ible", "ment", "ant", "ent", "ism", "ate", "iti", "ous", "ive", "ize",
          "ion", "al", "er", "ic")


def _is_vowel(c):
    return c in _VOWELS


def _regions(w):
    """R1 and R2 as start offsets (len(w) when empty)."""
    def after_vc(start):
        for i in range(start + 1, len(w)):
            if not _is_vowel(w[i]) and _is_vowel(w[i - 1]):
                return i + 1
        return len(w)
    for prefix in ("gener", "commun", "arsen"):
        if w.startswith(prefix):
            r1 = len(prefix)
            break
    else:
        r1 = after_vc(0)
    r2 = after_vc(r1) if r1 < len(w) else len(w)
    return r1, r2


def _short_syllable_at_end(w):
    if len(w) == 2:
        return _is_vowel(w[0]) and not _is_vowel(w[1])
    if len(w) >= 3:
        return (not _is_vowel(w[-3]) and _is_vowel(w[-2]) and not _is_vowel(w[-1]) and w[-1] not in "wxY")
    return False


def _is_short(w, r1):
    return r1 >= len(w) and _short_syllable_at_end(w)


def stem(word):
    w = word.lower()
    if len(w) <= 2:
        return w
    if w in _EXCEPTIONS:
        return _EXCEPTIONS[w]
    w = w.lstrip("'")
    if w.startswith("y"):
        w = "Y" + w[1:]
    w = "".join("Y" if c == "y" and i > 0 and _is_vowel(w[i - 1]) else c for i, c in enumerate(w))
    r1, r2 = _regions(w)

    # Step 0: possessive
    for suf in ("'s'", "'s", "'"):
        if w.endswith(suf):
            w = w[: -len(suf)]
            break

    # Step 1a
    if w.endswith("sses"):
        w = w[:-2]
    elif w.endswith("ied") or w.endswith("ies"):
        w = w[:-2] if len(w) > 4 else w[:-1]
    elif w.endswith("us") or w.endswith("ss"):
        pass
    elif w.endswith("s"):
        if any(_is_vowel(c) for c in w[:-2]):
            w = w[:-1]
    if w in _EXCEPTIONS_AFTER_1A:
        return w

    # Step 1b
    for suf in ("eedly", "ingly", "edly", "eed", "ing", "ed"):
        if w.endswith(suf):
            if suf in ("eed", "eedly"):
                if len(w) - len(suf) >= r1:
                    w = w[: -len(suf)] + "ee"
            else:
                base = w[: -len(suf)]
                if any(_is_vowel(c) for c in base):
                    w = base
                    if w.endswith(("at", "bl", "iz")):
                        w += "e"
                    elif w.endswith(_DOUBLES):
                        w = w[:-1]
                    elif _is_short(w, r1):
                        w += "e"
            break

    # Step 1c
    if len(w) > 2 and w[-1] in "yY" and not _is_vowel(w[-2]):
        w = w[:-1] + "i"

    # Step 2
    for suf, rep in _STEP2:
        if w.endswith(suf):
            if len(w) - len(suf) >= r1:
                if suf == "ogi":
                    if len(w) > 3 and w[-4] == "l":
                        w = w[:-3] + "og"
                elif suf == "li":
                    if len(w) > 2 and w[-3] in _LI_ENDINGS:
                        w = w[:-2]
                else:
                    w = w[: -len(suf)] + rep
            break

    # Step 3
    for suf, rep in _STEP3:
        if w.endswith(suf):
            if len(w) - len(suf) >= r1:
                if suf == "ative":
                    if len(w) - len(suf) >= r2:
                        w = w[: -len(suf)]
                else:
                    w = w[: -len(suf)] + rep
            break

    # Step 4
    for suf in _STEP4:
        if w.endswith(suf):
            if len(w) - len(suf) >= r2:
                if suf == "ion":
                    if len(w) > 3 and w[-4] in "st":
                        w = w[:-3]
                else:
                    w = w[: -len(suf)]
            break

    # Step 5
    if w.endswith("e"):
        if len(w) - 1 >= r2 or (len(w) - 1 >= r1 and not _short_syllable_at_end(w[:-1])):
            w = w[:-1]
    elif w.endswith("l"):
        if len(w) - 1 >= r2 and w.endswith("ll"):
            w = w[:-1]

    return w.replace("Y", "y")
