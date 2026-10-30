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
consolelog.py - a quiet record of everything Lena prints to the terminal (startup, the dream, bridges, errors,
tracebacks), for when she runs without a console window.

install() wraps sys.stdout and sys.stderr so every line still reaches the terminal exactly as before and is also
appended to a file with a timestamp. Whether the file is written is asked on every line (the `enabled` callable),
so turning the setting off stops it at once. Logging never gets in the way: any problem writing the file turns the
file off for the rest of the run and the terminal carries on untouched.

The file is moved to <name>.old.txt at startup when it has grown past MAX_BYTES, which keeps the total bounded.
"""

import os
import re
import sys
import threading
from datetime import datetime

MAX_BYTES = 10 * 1024 * 1024
_ANSI = re.compile(r"\x1b\[[0-9;]*[A-Za-z]")


class _Shared:
    """One file, one lock, shared by the stdout and stderr wrappers."""

    def __init__(self, path, enabled):
        self.path, self.enabled, self.lock, self.fh, self.failed = path, enabled, threading.Lock(), None, False

    def line(self, text):
        if self.failed:
            return
        try:
            if not self.enabled():
                return
            if self.fh is None:
                self.fh = open(self.path, "a", encoding="utf-8", errors="replace")
            self.fh.write(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] {text}\n")
        except Exception:
            self.failed = True                       # never let the log break the program

    def flush(self):
        try:
            if self.fh is not None:
                self.fh.flush()
        except Exception:
            self.failed = True


class _Tee:
    def __init__(self, stream, shared):
        self._stream, self._shared, self._partial = stream, shared, ""

    def write(self, s):
        n = self._stream.write(s) if self._stream is not None else len(s)
        try:
            with self._shared.lock:
                self._partial += _ANSI.sub("", s).replace("\r", "")
                *lines, self._partial = self._partial.split("\n")
                for ln in lines:
                    self._shared.line(ln)
        except Exception:
            pass
        return n

    def flush(self):
        if self._stream is not None:
            self._stream.flush()
        with self._shared.lock:
            self._shared.flush()

    def __getattr__(self, name):                    # isatty, fileno, encoding, ... behave as the real stream's
        if self._stream is None:
            raise AttributeError(name)
        return getattr(self._stream, name)


def install(path, enabled):
    """Wraps stdout and stderr. path: the log file. enabled: callable returning True while logging is wanted.
    Safe to call twice (the second call does nothing)."""
    if isinstance(sys.stdout, _Tee):
        return
    try:
        if os.path.exists(path) and os.path.getsize(path) > MAX_BYTES:
            os.replace(path, os.path.splitext(str(path))[0] + ".old.txt")
    except OSError:
        pass
    shared = _Shared(path, enabled)
    sys.stdout, sys.stderr = _Tee(sys.stdout, shared), _Tee(sys.stderr, shared)
    with shared.lock:
        shared.line(f"=== Lena started (pid {os.getpid()}) ===")
