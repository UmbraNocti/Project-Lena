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
file_bridge.py — the simplest possible bridge, and meant as the reference
implementation of the contract for anyone writing their own:

    def run(mode, canon, process_comment, stop_event):
        ...

No network, no auth. It just watches one or more directories (configured
as a list in file_bridge_settings.txt) in round-robin, treats each
dropped file as a comment, hands it to process_comment(text, speaker),
and — if a reply comes back in full mode — writes it to a companion file
next to the original. Everything else (canon gating, the reply gate,
recall, generation, seeding) lives in controller.py exactly as it does
for every other bridge; this file only transduces "a text-bearing file
appeared" into the same shape Reddit comments, emails, and Discord
messages already arrive in.

Supported formats: .txt (and any other plain-text extension you list),
.pdf (embedded text only — a scanned/image-only PDF has no text layer to
extract; OCR is a heavier dependency and deliberately left as an open
extension point), and .docx. Legacy .doc (pre-2007 binary Word) isn't
handled — no good pure-Python reader exists for it, only shelling out to
antiword/LibreOffice, which is more than this bridge is trying to carry.
PDF and .docx support are optional dependencies (pypdf, python-docx) —
if either isn't installed, files of that type are skipped with a console
warning rather than crashing the bridge.

    pip install pypdf python-docx

Speaker is the filename itself (without extension) — see
_speaker_from_filename for why.

Processed files are moved into a "_processed" subfolder within whichever
directory they came from — visible, inspectable, and doubles as the
dedupe mechanism, so this bridge needs no separate seen-file/cursor state
at all, unlike the network bridges.
"""

import time
from pathlib import Path

DEFAULTS = {
    "directories":     "",       # comma-separated list of folders to watch
    "extensions":      ".txt",   # comma-separated list of extensions to treat as input
    "poll_interval":   "15",
}


def _settings_file():
    return Path(__file__).parent.resolve() / "file_bridge_settings.txt"


def _load_settings():
    settings = dict(DEFAULTS)
    path = _settings_file()
    if not path.exists():
        print(f"[FILE] No {path.name} found — cannot start.")
        return settings
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, val = line.partition("=")
        key, val = key.strip(), val.strip()
        if key in settings:
            settings[key] = val
    return settings


def _responsive_sleep(seconds, stop_event, tick=2.0):
    waited = 0.0
    while waited < seconds and not stop_event.is_set():
        time.sleep(min(tick, seconds - waited))
        waited += tick


def _speaker_from_filename(path):
    """
    The document's own filename (without extension) is the speaker —
    e.g. "FAR.pdf" is recorded in the Speakers table as "FAR". This gives
    every dropped document its own persistent, recallable identity in
    memory, the same as a person's name would, rather than collapsing
    everything into one generic speaker. In listen mode the text itself
    is stored without a "FAR said:" label; recall shows the name from
    the Speakers table ("FAR: ..."), so Lena can still cite it.
    """
    return path.stem


def _extract_pdf_text(path):
    # Lazy import — the bridge shouldn't hard-fail at startup over a
    # dependency nobody's dropped a PDF to need yet.
    try:
        from pypdf import PdfReader
    except ImportError:
        print(f"[FILE] pypdf not installed — skipping {path.name}. "
              f"Install with: pip install pypdf")
        return None
    try:
        reader = PdfReader(str(path))
        pages = [(page.extract_text() or "").strip() for page in reader.pages]
        pages = [p for p in pages if p]
        if not pages:
            # Embedded-text only, by design — a scanned/image-only PDF has
            # no text layer at all. OCR is a different, heavier problem
            # (pytesseract + the tesseract binary + poppler) and deliberately
            # not attempted here; left as an open extension point.
            print(f"[FILE] {path.name} has no embedded text — likely a "
                  f"scanned/image-only PDF. OCR isn't supported here; "
                  f"leaving it in place.")
            return None
        return "\n\n".join(pages)
    except Exception as e:
        print(f"[FILE] Could not extract PDF text from {path.name}: {e}")
        return None


def _extract_docx_text(path):
    try:
        import docx
    except ImportError:
        print(f"[FILE] python-docx not installed — skipping {path.name}. "
              f"Install with: pip install python-docx")
        return None
    try:
        document = docx.Document(str(path))
        paragraphs = [p.text.strip() for p in document.paragraphs if p.text.strip()]
        if not paragraphs:
            print(f"[FILE] {path.name} has no readable text — leaving it in place.")
            return None
        return "\n".join(paragraphs)
    except Exception as e:
        print(f"[FILE] Could not extract .docx text from {path.name}: {e}")
        return None


def _extract_text(path):
    """
    Dispatches by extension. Plain text files are read directly; .pdf and
    .docx go through their own extractors. Legacy .doc (the pre-2007
    binary Word format) isn't handled — there's no good pure-Python
    reader for it, only shelling out to antiword/LibreOffice, which is a
    heavier dependency than this bridge is trying to carry. Left as an
    open extension point, same as OCR for scanned PDFs.
    """
    suffix = path.suffix.lower()
    if suffix == ".pdf":
        return _extract_pdf_text(path)
    if suffix == ".docx":
        return _extract_docx_text(path)
    try:
        return path.read_text(encoding="utf-8", errors="replace").strip()
    except Exception as e:
        print(f"[FILE] Could not read {path.name}: {e}")
        return None


def run(mode, canon, process_comment, stop_event):
    """
    Entry point controller.py calls in its own background thread.

    Cycles through every configured directory once per poll, oldest-
    looking files first within each (alphabetical, which for typically
    timestamp-prefixed filenames also means chronological). Each new file
    gets handed to process_comment(text, speaker) exactly once, then moved
    to that directory's "_processed" subfolder regardless of what
    controller decides — the move itself is the dedupe record. Only in
    full mode, and only if process_comment returned reply text, is a
    companion "<name>.reply.txt" written alongside the moved original.
    """
    settings = _load_settings()

    raw_dirs = [d.strip() for d in settings["directories"].split(",") if d.strip()]
    if not raw_dirs:
        print("[FILE] No directories configured in file_bridge_settings.txt — cannot start.")
        return

    directories = [Path(d).expanduser() for d in raw_dirs]
    extensions = tuple(
        e.strip().lower() if e.strip().startswith(".") else f".{e.strip().lower()}"
        for e in settings["extensions"].split(",") if e.strip()
    )
    poll_interval = float(settings["poll_interval"])

    for d in directories:
        try:
            d.mkdir(parents=True, exist_ok=True)
            (d / "_processed").mkdir(exist_ok=True)
        except Exception as e:
            print(f"[FILE] Could not prepare directory {d}: {e}")

    print(f"[FILE] Bridge running — mode={mode}, canon={canon or '(blank)'}, "
          f"watching {len(directories)} folder(s): "
          + ", ".join(str(d) for d in directories))

    while not stop_event.is_set():
        for d in directories:
            if stop_event.is_set():
                break
            if not d.exists():
                continue

            try:
                candidates = sorted(
                    p for p in d.iterdir()
                    if p.is_file() and p.suffix.lower() in extensions
                )
            except Exception as e:
                print(f"[FILE] Could not scan {d}: {e}")
                continue

            for path in candidates:
                if stop_event.is_set():
                    break

                try:
                    text = _extract_text(path)
                except Exception as e:
                    print(f"[FILE] Unexpected error extracting {path.name}: {e}")
                    continue
                if not text:
                    # Extraction failed (missing dependency, corrupt file,
                    # scanned PDF with no text layer) or genuinely empty.
                    # Left in place rather than moved — a silent archive
                    # would hide exactly the cases someone needs to notice
                    # and fix (install a package, replace a bad file).
                    continue

                speaker = _speaker_from_filename(path)
                reply_text = process_comment(text, speaker)

                processed_dir = d / "_processed"
                dest = processed_dir / path.name
                try:
                    path.rename(dest)
                except Exception as e:
                    print(f"[FILE] Could not move {path.name} to _processed: {e}")
                    continue

                if mode == "full" and reply_text:
                    reply_path = processed_dir / f"{path.stem}.reply.txt"
                    try:
                        reply_path.write_text(reply_text, encoding="utf-8")
                        print(f"[FILE] Replied to {speaker} -> {reply_path.name}")
                    except Exception as e:
                        print(f"[FILE] Could not write reply file: {e}")
                else:
                    print(f"[FILE] Read {path.name} from {speaker} — seeded, no reply.")

        _responsive_sleep(poll_interval, stop_event)

    print("[FILE] Bridge stopped.")
