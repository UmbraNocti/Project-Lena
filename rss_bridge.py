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
rss_bridge.py — watches any number of RSS/Atom feeds, same run() contract
as every other bridge:

    def run(mode, canon, process_comment, stop_event):
        ...

There is no posting destination for a feed — no thread to reply into, no
comment section to post to — so this bridge behaves identically no matter
what mode it's started in. It calls process_comment(text, speaker) for
every new entry and never acts on whatever comes back. Controller never
needs to know RSS is different from any other bridge; the "there is no
full mode" behavior lives entirely here, not as a special case anywhere
in controller.py.

Parses both RSS 2.0 and Atom with stdlib's xml.etree.ElementTree — no
feedparser dependency. Handles the common field-name variants of each
(pubDate vs updated, guid vs id, link-as-text vs link href=) but this is
a lightweight parser, not a full implementation of either spec; a feed
with an unusual structure may not parse cleanly.

HTML-to-text extraction (entry bodies and, optionally, related-link
pages) goes through BeautifulSoup4 rather than hand-rolled regex — it
decodes HTML entities as part of parsing (curly quotes etc. arrive
already-decoded, not as literal "&#8220;" sequences), and lets structural
noise (video embeds, WordPress photo captions, "Related:" boilerplate,
nav/header/footer on a scraped page) be removed by tag/class rather than
pattern-matched text.

    pip install beautifulsoup4

Each entry is deduped by its own guid/id (falling back to its link, then
to a title+date combination) and persisted per-feed to rss_seen.json, so
a feed is only ever read once. A feed's own <title> becomes the speaker/
attribution for everything it publishes (optionally overridden per-feed
via "Alias|https://..." in the feed list), refined further with the
entry's own byline when the feed provides one (RSS's <dc:creator>, Atom's
<author><name>) — e.g. "Daily Wire" alone, or "Daily Wire: Matt Walsh"
when that entry names an author. Every source gets its own persistent,
recallable identity rather than collapsing into one generic speaker.

When follow_related_links is on, a same-domain "Related: <a href=...>"
link inside an entry's body (captured before that boilerplate is
stripped) is fetched and seeded as its own separate entry, up to
max_related_per_entry per parent entry, one level deep only — this
follows the pattern once, like reading one more page, not a crawl.
Useful for feeds (Quanta Magazine's, notably) that only publish a teaser
and never embed the full article in <content:encoded> at all.
"""

import json
import re
import time
import xml.etree.ElementTree as ET
from pathlib import Path
from urllib.parse import urljoin, urlparse

import requests
from bs4 import BeautifulSoup

ROOT = Path(__file__).parent.resolve()
BRIDGE_SETTINGS_FILE = ROOT / "rss_settings.txt"
SEEN_FILE = ROOT / "rss_seen.json"

DEFAULTS = {
    "feeds":                       "",
    "poll_interval":               "300",  # 5 min between full cycles through every feed
    "feed_delay":                  "10",   # polite pause — between feeds, AND between each related-page fetch
    "max_new_per_feed_per_cycle":  "10",   # cap so a feed's entire backlog doesn't flood in at once
    "follow_related_links":        "true", # fetch a same-domain "Related:" link and seed it too
    "max_related_per_entry":       "2",    # cap per parent entry — this isn't meant to become a crawler
}


# ── Settings ──────────────────────────────────────────────────────────────────

def _load_settings():
    settings = dict(DEFAULTS)
    if not BRIDGE_SETTINGS_FILE.exists():
        print(f"[RSS] No {BRIDGE_SETTINGS_FILE.name} found — cannot start.")
        return settings
    for line in BRIDGE_SETTINGS_FILE.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, val = line.partition("=")
        key, val = key.strip(), val.strip()
        if key in settings:
            settings[key] = val
    return settings


def _parse_feed_list(raw):
    """
    "https://a.com/feed.xml, My Nickname|https://b.com/rss" ->
    [(None, "https://a.com/feed.xml"), ("My Nickname", "https://b.com/rss")]
    """
    feeds = []
    for entry in raw.split(","):
        entry = entry.strip()
        if not entry:
            continue
        if "|" in entry:
            alias, url = entry.split("|", 1)
            feeds.append((alias.strip() or None, url.strip()))
        else:
            feeds.append((None, entry))
    return feeds


# ── Dedupe state (per-feed set of guids/ids already seeded) ──────────────────
# Unbounded growth over long-term "leave it running indefinitely" use is
# accepted here, same as reddit_seen.json/email_seen.json — even a feed
# publishing dozens of items a day for a year is a small JSON file of
# short ID strings, not a real storage concern.

def _load_seen():
    if not SEEN_FILE.exists():
        return {}
    try:
        return json.loads(SEEN_FILE.read_text(encoding="utf-8"))
    except Exception:
        return {}


def _save_seen(seen):
    try:
        SEEN_FILE.write_text(json.dumps(seen), encoding="utf-8")
    except Exception as e:
        print(f"[RSS] Could not save seen-entries state: {e}")


def _responsive_sleep(seconds, stop_event, tick=2.0):
    waited = 0.0
    while waited < seconds and not stop_event.is_set():
        time.sleep(min(tick, seconds - waited))
        waited += tick


# ── HTML extraction (entry bodies and, optionally, scraped related pages) ────

def _same_domain(url_a, url_b):
    """www.example.com and example.com count as the same domain — this
    is the guard against a "Related:" link turning into a hop onto some
    entirely different site."""
    def norm(u):
        try:
            netloc = urlparse(u).netloc.lower()
            return netloc[4:] if netloc.startswith("www.") else netloc
        except Exception:
            return ""
    a, b = norm(url_a), norm(url_b)
    return bool(a) and a == b


def _guess_page_author(soup):
    """Best-effort author lookup for a scraped related-link page. There's
    no shared convention the way RSS's dc:creator is one — this only
    catches sites using one of a few common meta tags. None found is a
    normal, expected outcome, not a failure."""
    for attrs, key in (
        ({"name": "author"}, "content"),
        ({"property": "article:author"}, "content"),
        ({"name": "twitter:creator"}, "content"),
    ):
        tag = soup.find("meta", attrs=attrs)
        if tag and tag.get(key):
            val = tag[key].strip().lstrip("@")
            if val:
                return val
    return None


def _extract_content(html_text, prefer_article=False, capture_related=False):
    """
    Turns an entry's (or a scraped page's) HTML into clean prose, using
    BeautifulSoup rather than regex — entities are decoded automatically
    as part of parsing, and structural noise is removed by tag/class
    instead of pattern-matched text.

    prefer_article: for a full scraped webpage (not a feed's own
    <content:encoded>), narrow to <article>/<main> and strip
    <nav>/<header>/<footer>/<aside> first — a real page has far more
    surrounding chrome than an isolated feed field does.

    capture_related: pull href(s) out of any "Related: <a href=...>"
    paragraph BEFORE removing it as boilerplate, returning them alongside
    the cleaned text. Only meaningful for feed entries — a page already
    reached via a related link isn't searched for further related links,
    since depth is capped at one hop.

    Returns (clean_text, related_urls) — related_urls is always [] unless
    capture_related=True and at least one was found.

    This is tuned to common WordPress conventions (the majority of news/
    blog output), not a general-purpose readability algorithm — some
    sites' boilerplate or article markup will still slip through.
    """
    if not html_text:
        return "", []

    soup = BeautifulSoup(html_text, "html.parser")

    if prefer_article:
        for tag in soup.find_all(["nav", "header", "footer", "aside"]):
            tag.decompose()
        root = soup.find("article") or soup.find("main") or soup
    else:
        root = soup

    for tag in root.find_all(["script", "style", "iframe"]):
        tag.decompose()
    for tag in root.find_all(class_=lambda c: c and "wp-caption" in c):
        tag.decompose()

    related_urls = []
    for p_tag in root.find_all("p"):
        if p_tag.get_text(strip=True).lower().startswith("related:"):
            if capture_related:
                for a in p_tag.find_all("a", href=True):
                    related_urls.append(a["href"])
            p_tag.decompose()

    text = root.get_text(separator=" ", strip=True)
    text = re.sub(r"\s+", " ", text).strip()
    return text, related_urls


def _fetch_related_page(url):
    """Fetches and extracts a related-link page's article text plus a
    best-effort author guess. Returns (text, author) — author may be
    None. Any failure returns ("", None) rather than raising, since a
    related link not panning out shouldn't stop the parent entry."""
    try:
        r = requests.get(
            url, timeout=20,
            headers={"User-Agent": "ProjectLena-RSSBridge/1.0 (+https://github.com/UmbraNocti/Project-Lena)"},
        )
        r.raise_for_status()
        soup = BeautifulSoup(r.text, "html.parser")
        author = _guess_page_author(soup)
        text, _ = _extract_content(str(soup), prefer_article=True, capture_related=False)
        return text, author
    except Exception as e:
        print(f"[RSS] Could not fetch related link {url}: {e}")
        return "", None


# ── RSS / Atom parsing ─────────────────────────────────────────────────────────

def _local(tag):
    return tag.split('}')[-1] if '}' in tag else tag


def _child_text(entry_el, names):
    """First direct child of entry_el whose local tag is in `names`. Handles
    both RSS-style text content (<link>http://...</link>) and Atom-style
    href attributes (<link href="http://..."/>)."""
    for child in entry_el:
        if _local(child.tag) in names:
            text = (child.text or "").strip()
            if text:
                return text
            href = child.get("href")
            if href:
                return href.strip()
    return ""


def _entry_body(entry_el):
    """
    Prefer the full article body over the short teaser most feeds also
    include alongside it. RSS/WordPress feeds carry the full body as
    <content:encoded>, whose local name (after the namespace prefix is
    stripped) is "encoded", not "content". Atom puts full content in a
    plain <content> element. <description>/<summary> is only used as a
    fallback when a feed doesn't provide either — this is exactly the
    case for feeds (Quanta Magazine's, for instance) that only ever
    publish a teaser, which is what related-link-following is for.
    """
    full = _child_text(entry_el, {"encoded", "content"})
    if full:
        return full
    return _child_text(entry_el, {"description", "summary"})


def _entry_author(entry_el):
    """RSS's <dc:creator> (local name "creator") or Atom's nested
    <author><name>...</name></author>. None if the entry doesn't name
    one — plenty don't (wire copy, live-blog placeholders)."""
    creator = _child_text(entry_el, {"creator"})
    if creator:
        return creator
    for child in entry_el:
        if _local(child.tag) == "author":
            name = _child_text(child, {"name"})
            if name:
                return name
            text = (child.text or "").strip()
            if text:
                return text
    return None


def _feed_title(root):
    for channel in root:
        if _local(channel.tag) == "channel":         # RSS 2.0
            for child in channel:
                if _local(child.tag) == "title":
                    return (child.text or "").strip()
    for child in root:                                 # Atom
        if _local(child.tag) == "title":
            return (child.text or "").strip()
    return ""


def _find_entries(root):
    # Deliberately forgiving: matches "item" (RSS 2.0 / RSS 1.0-RDF) or
    # "entry" (Atom) anywhere in the tree, rather than requiring an exact
    # channel/feed structure, so mildly nonstandard feeds still parse.
    return [el for el in root.iter() if _local(el.tag) in ("item", "entry")]


def _fetch_and_parse(url):
    r = requests.get(
        url, timeout=20,
        headers={"User-Agent": "ProjectLena-RSSBridge/1.0 (+https://github.com/UmbraNocti/Project-Lena)"},
    )
    r.raise_for_status()
    return ET.fromstring(r.content)


# ── Main loop ─────────────────────────────────────────────────────────────────

def run(mode, canon, process_comment, stop_event):
    """
    Entry point controller.py calls in its own background thread.

    Cycles through every configured feed once per poll_interval, with a
    polite feed_delay pause between each individual feed fetch (and,
    when related-link-following is on, that same pause between each
    related-page fetch too — there's no user waiting on any of this).
    Every new entry (by guid/id, not already in rss_seen.json) is handed
    to process_comment(text, speaker) exactly once, speaker being the
    feed's name alone, or "FeedName: Author" when the entry names one.
    Whatever comes back is never used — there's nowhere to post a reply
    for a feed — so this bridge behaves identically regardless of what
    mode it was started in.

    When follow_related_links is on, a same-domain "Related:" link found
    in the entry's body is fetched and seeded as its own separate entry
    (own speaker, own guid — added to the same per-feed seen-set so it's
    never re-fetched even if cited again later), one hop deep, capped at
    max_related_per_entry per parent entry.
    """
    if mode == "full":
        print("[RSS] No posting destination exists for a feed — running "
              "listen-only regardless of mode. This is normal.")

    settings = _load_settings()
    feeds = _parse_feed_list(settings["feeds"])
    if not feeds:
        print("[RSS] No feeds configured in rss_settings.txt — cannot start.")
        return

    poll_interval = float(settings["poll_interval"])
    feed_delay = float(settings["feed_delay"])
    max_new = int(settings["max_new_per_feed_per_cycle"])
    follow_related = settings["follow_related_links"].strip().lower() != "false"
    max_related = int(settings["max_related_per_entry"])
    seen = _load_seen()

    print(f"[RSS] Bridge running — canon={canon or '(blank)'}, "
          f"watching {len(feeds)} feed(s), "
          f"related-link-following {'on' if follow_related else 'off'}.")

    while not stop_event.is_set():
        for alias, url in feeds:
            if stop_event.is_set():
                break

            try:
                root = _fetch_and_parse(url)
            except Exception as e:
                print(f"[RSS] Could not fetch/parse {url}: {e}")
                _responsive_sleep(feed_delay, stop_event)
                continue

            feed_name = alias or _feed_title(root) or url
            seen_ids = set(seen.get(url, []))
            # Feeds conventionally list newest-first; reverse so entries
            # are handed over oldest-to-newest within this fetch.
            entries = list(reversed(_find_entries(root)))

            new_count = 0
            for entry_el in entries:
                if stop_event.is_set() or new_count >= max_new:
                    break

                title = _child_text(entry_el, {"title"})
                link = _child_text(entry_el, {"link"})
                date_str = _child_text(entry_el, {"pubDate", "updated", "published", "date"})
                guid = _child_text(entry_el, {"guid", "id"}) or link or f"{title}|{date_str}"

                if not guid or guid in seen_ids:
                    continue
                seen_ids.add(guid)
                new_count += 1

                raw_body = _entry_body(entry_el)
                body, related_urls = _extract_content(
                    raw_body, prefer_article=False, capture_related=follow_related
                )

                author = _entry_author(entry_el)
                speaker = f"{feed_name}: {author}" if author else feed_name

                text = title
                if body:
                    text += f"\n\n{body}"
                meta = []
                if date_str:
                    meta.append(f"Published: {date_str}")
                if link:
                    meta.append(f"Link: {link}")
                if meta:
                    text += f"\n\n({' | '.join(meta)})"

                process_comment(text, speaker)

                # Paces entries WITHIN a feed, not just between feeds —
                # without this, a first-run backlog (or any feed with
                # several new items in one cycle) fires all of that
                # feed's seed calls back-to-back with nothing pacing
                # them, which is exactly what floods a single local
                # backend and starts producing timeouts further down the
                # pipeline (temporal resolution, recall, generation).
                # Reuses feed_delay rather than adding a separate knob.
                _responsive_sleep(feed_delay, stop_event)
                if stop_event.is_set():
                    break

                # ── Related links: one hop, same domain only, capped ──
                if follow_related and related_urls and link:
                    followed = 0
                    for related_url in related_urls:
                        if stop_event.is_set() or followed >= max_related:
                            break
                        absolute_url = urljoin(link, related_url)
                        if not _same_domain(absolute_url, link):
                            continue
                        if absolute_url in seen_ids:
                            continue

                        _responsive_sleep(feed_delay, stop_event)
                        if stop_event.is_set():
                            break

                        related_text, related_author = _fetch_related_page(absolute_url)
                        seen_ids.add(absolute_url)
                        if not related_text:
                            continue

                        related_speaker = (
                            f"{feed_name}: {related_author}" if related_author else feed_name
                        )
                        process_comment(
                            f"{related_text}\n\n(Link: {absolute_url})", related_speaker
                        )
                        followed += 1

            if new_count:
                print(f"[RSS] {feed_name}: {new_count} new entr{'y' if new_count == 1 else 'ies'} seeded.")

            seen[url] = list(seen_ids)
            _save_seen(seen)

            _responsive_sleep(feed_delay, stop_event)

        _responsive_sleep(poll_interval, stop_event)

    print("[RSS] Bridge stopped.")
