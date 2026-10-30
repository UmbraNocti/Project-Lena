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
image_gate.py - Decides whether a retrieved image is worth showing the user.
Never sees the image itself. Only sees metadata: the image's match score,
its date, its stored description (if any), the top two scoring memories'
text, the current user prompt, and the reply that was already sent.
Runs AFTER the reply has been posted, so a "show" decision arrives a moment
later as a follow-up. Fails closed — any error or ambiguous response means
DO NOT show the image.
"""

import time
import llm_client
from pathlib import Path
from datetime import datetime

ROOT = Path(__file__).parent.resolve()
IMAGE_GATE_PROMPT_FILE = ROOT / "prompts" / "image_gate_prompt.txt"


def load_prompt(path):
    if not path.exists():
        print(f"[IMAGE_GATE][WARNING] Prompt file not found: {path}")
        return ""
    return path.read_text(encoding="utf-8").strip()


def should_show_image(image_score_pct, image_date_epoch, top_memories_text,
                       user_prompt, ai_response=None, description=None,
                       model_saw_image=False) -> bool:
    """
    Decides whether to show a retrieved image to the user.
    
    Args:
        image_score_pct: match confidence percentage (0-100)
        image_date_epoch: unix timestamp when image was saved
        top_memories_text: list of up to 2 memory text strings (already formatted)
        user_prompt: current user prompt for context
        ai_response: the reply already sent to the user, if any
        description: the image's stored description, if it has one
        model_saw_image: True if the image was attached to the model's
            reasoning call (vision path), False if the model worked blind
    
    Returns:
        bool: True only on an explicit, parseable "YES". Everything else is False.
    """
    prompt = load_prompt(IMAGE_GATE_PROMPT_FILE)
    if not prompt:
        print("[IMAGE_GATE][WARNING] image_gate_prompt.txt missing, defaulting to NO.")
        return False

    date_str = datetime.fromtimestamp(image_date_epoch).strftime("%Y-%m-%d")
    memories_block = "\n".join(f"- {m}" for m in top_memories_text) if top_memories_text else "(none)"

    context = (
        f"Image match confidence: {image_score_pct:.1f}%\n"
        f"Image date: {date_str}\n"
        f"The assistant {'already looked at this image while writing its reply' if model_saw_image else 'could NOT see this image when writing its reply'}.\n"
    )
    if description:
        context += f"Image description: {description}\n"
    context += f"Top related memories:\n{memories_block}\n\n"
    context += f"Current user prompt: {user_prompt}"
    if ai_response:
        context += f"\n\nReply already sent to the user: {ai_response}"

    try:
        raw = llm_client.get_completion(
            system_prompt=prompt,
            user_text=context,
            max_tokens=8,
            temperature=0.0,
            timeout=30,
            preferred_tier="small",
        )
    except Exception as e:
        print(f"[IMAGE_GATE][WARNING] LLM call failed: {e}")
        return False

    if not raw:
        return False

    cleaned = raw.strip().upper()
    return cleaned.startswith("YES")


if __name__ == "__main__":
    import sys
    if len(sys.argv) >= 3:
        score = float(sys.argv[1])
        prompt = sys.argv[2]
        result = should_show_image(score, time.time(), [], prompt)
        print(f"Show image: {result}")
    else:
        print("Usage: python3 image_gate.py <score_pct> \"user prompt\"")
