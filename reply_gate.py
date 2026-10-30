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
reply_gate.py — should_reply(comment_text, memories, speaker) -> bool

Pre-reasoning classifier for external-bridge comments (Reddit, Discord,
etc.). Runs BEFORE any response is generated — controller.py's
process_bridge_comment() only proceeds to recall-informed generation if
this returns True. Same shape and calling convention as image_gate.py's
should_show_image(): a narrow, single-purpose LLM call, loaded dynamically
by controller.py via importlib and never imported the other direction.

Default is silence. An ambiguous or low-confidence read, low-effort bait,
or any error talking to the model should all resolve to False — better to
be thought quiet than to open your mouth and remove all doubt. Nothing is
lost by staying silent: the comment gets seeded either way (controller.py
handles that regardless of what this function returns), so DreamCycle
still has it to work with. All this gate controls is whether she speaks.
"""

from pathlib import Path
import llm_client

ROOT = Path(__file__).parent.resolve()
PROMPT_FILE = ROOT / "prompts" / "reply_gate.txt"

DEFAULT_PROMPT = (
    "You are deciding whether a specific reply is worth writing — not "
    "writing it. You'll be given a comment, who said it, and any recalled "
    "memories about the topic or the person. Answer only YES or NO.\n\n"
    "Answer YES only if a reply would clearly add something worth "
    "reading: a real question worth answering, a substantive point worth "
    "engaging, or a direct address worth responding to.\n\n"
    "Answer NO for: low-effort bait, hostility with no real question in "
    "it, or comments that don't need a response. If you are unsure, answer NO."
)


def _load_prompt():
    if PROMPT_FILE.exists():
        text = PROMPT_FILE.read_text(encoding="utf-8").strip()
        if text:
            return text
    return DEFAULT_PROMPT


def should_reply(comment_text, memories, speaker):
    prompt = _load_prompt()

    user_content = f"Comment from {speaker}: {comment_text}"
    if memories:
        user_content += f"\n\nRecalled memories:\n{memories}"

    try:
        answer = llm_client.get_completion(
            system_prompt=prompt,
            user_text=user_content,
            max_tokens=10,
            temperature=0.1,
            timeout=120,
        ).strip().upper()
        return answer.startswith("YES")
    except Exception as e:
        print(f"[REPLY_GATE] Classifier call failed, defaulting to silence: {e}")
        return False
