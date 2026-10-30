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
llm_client.py — Single point of contact with the inference backend.

Every call (controller.py, seed.py, recall.py, dream.py, image_gate.py, reply_gate.py,
image_describe.py, director_helpers.py, temporal_anchor.py) comes through here. Backend independence is maintained: any OpenAI-compatible
/v1/chat/completions endpoint works (llama.cpp, KoboldCPP, OpenAI API, etc).

Tiered routing: Automatically routes requests to different backends based on
payload size. Useful for running lightweight models (7B) for simple operations
and heavier models (26B+) or cloud APIs for complex reasoning. All tiers must
speak OpenAI format. Automatic fallback to default port on any error.

Vision: a call can carry images. Each ordinary tier can be flagged
<tier>_vision in settings.txt, and a fourth "specialist" route is used only
when no ordinary tier can see images (see vision_mode()). With no vision
route, images are skipped and the call goes through blind.

Settings are read from settings.txt at module load time (and again on
reload_settings()), so all callers get automatic routing without needing to
pass anything.
"""

import re
import base64
import io
import requests
import json
from pathlib import Path


# ── Module-level settings (read from settings.txt at startup) ─────────────────
_SETTINGS = {}


def _load_settings():
    """Load backend, auth, model, and threshold config from settings.txt at module startup."""
    global _SETTINGS
    
    root = Path(__file__).parent
    settings_file = root / "settings.txt"
    
    # Defaults (IP:port format)
    _SETTINGS = {
        "default_backend": "127.0.0.1:5001",
        "default_api_key": "",
        "default_model": "",
        "small_backend": "",  # "" = disabled
        "small_api_key": "",
        "small_model": "",
        "large_backend": "",  # "" = disabled
        "large_api_key": "",
        "large_model": "",
        "specialist_backend": "",  # "" = disabled; vision-only fourth route
        "specialist_api_key": "",
        "specialist_model": "",
        "default_vision": False,
        "small_vision": False,
        "large_vision": False,
        "specialist_send_recalled": True,
        "vision_max_images": 2,
        "vision_max_px": 1024,
        "small_threshold": 5000,
        "large_threshold": 50000,
        "min_output_tokens": 500,
        # Embedding route (recall's vector lane). "" = disabled: recall then runs on
        # handles alone. OpenAI-compatible /v1/embeddings, e.g. a llama.cpp
        # server started with --embeddings, or any hosted embeddings API.
        "embedding_backend": "",
        "embedding_api_key": "",
        "embedding_model": "",
        "embedding_doc_prefix": "",
        "embedding_query_prefix": "",
        "embedding_batch_size": 16,
    }
    
    # Read settings.txt if it exists
    if settings_file.exists():
        try:
            with open(settings_file, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line or line.startswith("#"):
                        continue
                    if "=" in line:
                        key, value = line.split("=", 1)
                        key = key.strip().strip('"').strip("'")
                        value = value.strip().strip('"').strip("'")
                        
                        # Parse backend, API key, and model settings
                        if key in ["default_backend", "default_api_key", "default_model",
                                   "small_backend", "small_api_key", "small_model",
                                   "large_backend", "large_api_key", "large_model",
                                   "specialist_backend", "specialist_api_key", "specialist_model",
                                   "embedding_backend", "embedding_api_key", "embedding_model",
                                   "embedding_doc_prefix", "embedding_query_prefix"]:
                            _SETTINGS[key] = value
                        elif key in ["default_vision", "small_vision", "large_vision",
                                     "specialist_send_recalled"]:
                            _SETTINGS[key] = value.lower() == "true"
                        elif key in ["small_threshold", "large_threshold", "min_output_tokens",
                                     "vision_max_images", "vision_max_px", "embedding_batch_size"]:
                            try:
                                _SETTINGS[key] = int(value)
                            except ValueError:
                                pass
        except Exception as e:
            print(f"[LLM_CLIENT][WARNING] Could not read settings.txt, using defaults: {e}")
    else:
        print("[LLM_CLIENT][WARNING] settings.txt not found, using defaults.")


# Load settings at module import time
_load_settings()


def reload_settings():
    """Re-reads settings.txt (used by the !reload command). Callers see the
    new values on their next call; nothing needs restarting."""
    _load_settings()
    _warned_fallbacks.clear()


_warned_fallbacks = set()  # tiers we've already printed a fallback warning for


def _warn_once(key, message):
    if key not in _warned_fallbacks:
        _warned_fallbacks.add(key)
        print(f"[LLM_CLIENT][WARNING] {message}")


def startup_report():
    """
    Sanity-checks the loaded settings and returns (problems, notes): lists
    of human-readable lines. Problems are things that will not work as the
    config seems to intend; notes are expected-but-worth-knowing fallbacks.
    Printed at startup and by the !reload command.
    """
    problems, notes = [], []
    if not _SETTINGS["default_backend"]:
        problems.append("default_backend is empty — calls with no other route will fail.")
    for tier in ("small", "large"):
        if not _SETTINGS[f"{tier}_backend"]:
            notes.append(f"{tier}_backend not set — {tier}-tier calls use default_backend.")
    if (_SETTINGS["small_backend"] and _SETTINGS["large_backend"]
            and _SETTINGS["small_threshold"] >= _SETTINGS["large_threshold"]):
        problems.append("small_threshold >= large_threshold — size routing will behave oddly.")
    for tier in ("default", "small", "large"):
        if _SETTINGS[f"{tier}_vision"] and not _SETTINGS[f"{tier}_backend"]:
            problems.append(f"{tier}_vision = true but {tier}_backend is empty — flag ignored.")
    flagged = _flagged_vision_tiers()
    if flagged and _SETTINGS["specialist_backend"]:
        notes.append(f"specialist_backend is set but unused: {', '.join(flagged)} can already see images.")
    mode = vision_mode()
    if mode is None:
        notes.append("No vision route configured — images are skipped and Lena works blind.")
    else:
        notes.append(f"Vision route: {mode}.")
        if mode == "specialist" and _SETTINGS["specialist_send_recalled"]:
            notes.append("specialist_send_recalled = true — recalled images will be sent to the specialist backend.")
        try:
            import PIL  # noqa: F401
        except ImportError:
            notes.append("Pillow not installed — images are sent without downscaling.")
    return problems, notes


def get_completion(system_prompt=None, user_text=None, messages=None,
                    max_tokens=2048, temperature=0.7, top_p=0.9,
                    repetition_penalty=1.1, timeout=180, preferred_tier=None,
                    images=None, return_meta=False, vision_failure_note=None, thinking=None):
    """
    Single entry point for all LLM requests. No port passing, no mode switching.
    Settings are read from settings.txt automatically.
    
    Two call styles (provide exactly one):
      1. system_prompt + user_text: simple operations
         llm_client.get_completion(system_prompt="...", user_text="...")
         Used by: seed.py, dream.py, image_gate.py, reply_gate.py
      
      2. messages: pre-built chat history
         llm_client.get_completion(messages=[{role: "system", ...}, ...])
         Used by: controller.py chat pipeline
    
    Routing is automatic and deterministic by default:
      - Measures payload size (in bytes, UTF-8)
      - Routes to small/medium/large backend based on thresholds in settings.txt
      - Fallback to default (medium) port on any error
    
    A caller can bypass size-based routing entirely by passing preferred_tier.
    This is a request-level override, not a negation of fallback behavior:
      - If the requested tier's backend isn't configured in settings.txt,
        the call transparently uses default instead (config-level fallback).
      - If the requested tier's backend IS configured but the request itself
        fails (timeout, connection error, bad response), the normal
        fallback-to-default retry still applies.
    
    Args:
        system_prompt: system instruction (used with user_text only)
        user_text: user query (used with system_prompt only)
        messages: pre-built messages list [{role, content}, ...]
        max_tokens: max output length (default 2048)
        temperature: sampling temperature (default 0.7)
        top_p: nucleus sampling (default 0.9)
        repetition_penalty: penalty for repeated tokens (default 1.1)
        timeout: request timeout in seconds (default 180)
        thinking: optional True/False that overrides the thinking mode preferred_tier would pick
            (e.g. preferred_tier="large", thinking=False asks the large model without its reasoning
            block). None leaves the mode to preferred_tier.
        preferred_tier: optional override, one of None, "small", "large".
            When set, skips payload-size-based tier selection for the
            *backend* and requests that tier directly (falling back to
            default if unconfigured). It also fully determines thinking
            mode: "large" -> thinking on, "small" -> thinking off (dumdum
            mode) — regardless of payload size. Leave as None for normal
            automatic backend routing; in that case no thinking flags are
            sent at all, and the backend's own configured/model default
            applies untouched.
    
        images: optional list of images to attach to the last user message.
            Each entry is a file path (treated as a fresh upload) or a dict
            {"path": ..., "recalled": bool}. Routed per the vision rules in
            vision_mode(); if no route can see images, or the vision call
            fails, the call is retried/sent blind (text only) rather than
            failing. See filter_images() for the cap and privacy filter.
        return_meta: when True, returns (text, {"images_sent": int,
            "vision_tier": str|None}) instead of just text, so the caller
            can tell whether the model actually saw the images.
    
    Returns:
        str: model response, stripped of whitespace
    
    Raises:
        ValueError: if neither (system_prompt+user_text) nor messages provided,
            or if preferred_tier is not one of None, "small", "large"
        Exception: if all backends fail (original error from default port attempt)
    """
    if preferred_tier is not None and preferred_tier not in ("small", "large"):
        raise ValueError('preferred_tier must be None, "small", or "large"')
    
    # Build messages list if not provided
    if messages is None:
        if system_prompt is None or user_text is None:
            raise ValueError(
                "get_completion requires either messages=, or both "
                "system_prompt= and user_text="
            )
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_text},
        ]
    
    # Determine which backend to use (size-based, unless overridden by preferred_tier)
    target_backend, backend_tier = _select_backend(messages, max_tokens, preferred_tier)
    
    # Determine thinking mode (size-based, unless overridden by preferred_tier)
    thinking_enabled = _select_thinking_mode(messages, max_tokens, preferred_tier)
    if thinking is not None:
        thinking_enabled = bool(thinking)
    
    # Vision attempt — only when images were supplied AND some route can see.
    # Any failure here falls through to the ordinary blind path below.
    if images:
        route = _select_vision_route(backend_tier)
        allowed = filter_images(images)
        if route and allowed:
            vision_backend, vision_tier = route
            try:
                vision_messages = _attach_images(messages, allowed)
                text = _send_request(vision_backend, vision_messages, max_tokens, temperature,
                                     top_p, repetition_penalty, timeout, thinking_enabled,
                                     vision_tier)
                return (text, {"images_sent": len(allowed), "vision_tier": vision_tier}) if return_meta else text
            except Exception as e:
                print(f"[LLM_CLIENT][WARNING] Vision call failed on {vision_tier}, going blind: {e}")
                if vision_failure_note:
                    # the caller wants the model told, so it can explain to the user why it cannot see
                    messages = _with_system_note(messages, vision_failure_note.format(error=failure_reason(e)[0]))
    
    meta = {"images_sent": 0, "vision_tier": None}
    
    # Try target backend first
    try:
        text = _send_request(target_backend, messages, max_tokens, temperature,
                             top_p, repetition_penalty, timeout, thinking_enabled, backend_tier)
        return (text, meta) if return_meta else text
    except Exception as e:
        # Fallback: if not already on default, retry default backend
        default_backend = _SETTINGS["default_backend"]
        if target_backend != default_backend:
            print(f"[LLM_CLIENT][WARNING] {backend_tier} backend ({target_backend}) failed: {e} "
                  f"— retrying on default_backend.")
            try:
                text = _send_request(default_backend, messages, max_tokens, temperature,
                                     top_p, repetition_penalty, timeout, thinking_enabled, "default")
                return (text, meta) if return_meta else text
            except Exception as e2:
                # Default also failed, propagate original error
                print(f"[LLM_CLIENT][ERROR] default_backend also failed: {e2}")
                raise e
        else:
            # Already on default backend, no fallback available
            raise


def _with_system_note(messages, note):
    """Copy of messages with note appended to the system message (or a system message added)."""
    out = [dict(m) for m in messages]
    if out and out[0].get("role") == "system" and isinstance(out[0].get("content"), str):
        out[0]["content"] += f"\n\n{note}"
    else:
        out.insert(0, {"role": "system", "content": note})
    return out


def failure_reason(e):
    """
    (what went wrong, what to do about it) for a failed LLM call, in plain words. Used when the model itself
    cannot be reached, so the explanation has to come from Lena's own code.
    """
    request = getattr(e, "request", None)
    where = re.sub(r"^https?://|/v1/.*$", "", getattr(request, "url", "") or "") or _SETTINGS.get("default_backend") or ""
    if not _SETTINGS.get("default_backend"):
        return ("no language model is configured",
                "Open the settings with !settings, set default_backend to your model server's address "
                "(for example 127.0.0.1:8080) or an API's URL, then type !reload.")
    if isinstance(e, requests.exceptions.ConnectionError):
        return (f"nothing is answering at {where}",
                "Start your model server (or check its address in settings.txt), then send your message again.")
    if isinstance(e, requests.exceptions.Timeout):
        return (f"the model server at {where} did not answer in time",
                "It may still be loading the model, or the request was too much for it. Give it a minute and try again.")
    response = getattr(e, "response", None)
    if isinstance(e, requests.exceptions.HTTPError) and response is not None:
        code, body = response.status_code, (response.text or "")[:200].strip()
        if code in (401, 403):
            return (f"the server at {where} refused my API key (HTTP {code})", "Check the api_key lines in settings.txt.")
        if code == 404:
            return (f"the server at {where} has no such endpoint or model (HTTP 404)",
                    "Check the backend address and the model name in settings.txt.")
        if "context" in body.lower() or "too long" in body.lower():
            return (f"the request was too long for the model at {where} ({body})",
                    "Start the model server with a larger context (its -c option); recall alone needs about 4,000 tokens.")
        return (f"the server at {where} returned an error (HTTP {code}: {body})",
                "The model server's own console usually says more.")
    return (f"the call to {where} failed ({type(e).__name__}: {e})",
            "The console where Lena is running shows the full error.")


def describe_failure(e, saved=True):
    """The message shown to the user when no reply could be generated. saved: the message was still remembered."""
    why, fix = failure_reason(e)
    kept = "Your message is saved, so I'll still remember it. " if saved else "Nothing was written. "
    return (f"[Lena] I can't reach my language model: {why}. {fix} "
            f"{kept}(!settings opens the settings file, !reload re-reads it.)")


# ══════════════════════════════════════════════════════════════════════════════
# Vision routing
# ══════════════════════════════════════════════════════════════════════════════
#
# Three ordinary tiers can each be flagged <tier>_vision = true in settings.txt.
# A fourth "specialist" route exists purely as a vision backstop: it is used
# if and only if NO ordinary tier is flagged. It never participates in
# size-based routing, preferred_tier, or text fallback.

_VISION_TIER_ORDER = ("large", "default", "small")  # tie-break after the call's own tier
_IMAGE_EXTS = {".jpg": "image/jpeg", ".jpeg": "image/jpeg", ".png": "image/png",
               ".webp": "image/webp", ".gif": "image/gif"}
SUPPORTED_IMAGE_EXTS = frozenset(_IMAGE_EXTS)  # also the upload allowlist in controller.py


def _tier_backend(tier):
    return _SETTINGS.get(f"{tier}_backend", "")


def _flagged_vision_tiers():
    return [t for t in _VISION_TIER_ORDER
            if _SETTINGS.get(f"{t}_vision") and _tier_backend(t)]


def vision_mode():
    """"native" if any ordinary tier is flagged vision, "specialist" if only
    the specialist route can see, None if nothing can (blind)."""
    if _flagged_vision_tiers():
        return "native"
    if _SETTINGS.get("specialist_backend"):
        return "specialist"
    return None


def _select_vision_route(normal_tier):
    """(backend, tier) for an image-bearing call, or None. Prefers the
    call's own normal tier, then large -> default -> small, then specialist."""
    flagged = _flagged_vision_tiers()
    if normal_tier in flagged:
        return _tier_backend(normal_tier), normal_tier
    if flagged:
        return _tier_backend(flagged[0]), flagged[0]
    if _SETTINGS.get("specialist_backend"):
        return _SETTINGS["specialist_backend"], "specialist"
    return None


def vision_max_images():
    return _SETTINGS["vision_max_images"]


def filter_images(images):
    """
    Normalizes the caller's image list to [{"path": Path, "recalled": bool}],
    dropping missing files and unsupported formats, dropping recalled images
    when only the specialist can see and specialist_send_recalled is false,
    and capping at vision_max_images. Idempotent, and independent of
    message content, so callers can use it up front to decide what to
    describe to the model before the call is made.
    """
    out = []
    mode = vision_mode()
    for item in images or []:
        entry = {"path": item, "recalled": False} if not isinstance(item, dict) else item
        path = Path(entry["path"])
        recalled = bool(entry.get("recalled"))
        if path.suffix.lower() not in _IMAGE_EXTS or not path.is_file():
            continue
        if recalled and mode == "specialist" and not _SETTINGS["specialist_send_recalled"]:
            continue
        out.append({"path": path, "recalled": recalled})
        if len(out) >= _SETTINGS["vision_max_images"]:
            break
    return out


def _encode_image(path):
    """Returns a data: URI for the image, downscaled if Pillow is available."""
    mime = _IMAGE_EXTS[path.suffix.lower()]
    raw = path.read_bytes()
    max_px = _SETTINGS["vision_max_px"]
    try:
        from PIL import Image  # optional dependency
        with Image.open(io.BytesIO(raw)) as img:
            if max(img.size) > max_px:
                img.thumbnail((max_px, max_px))
                buf = io.BytesIO()
                if img.mode in ("RGBA", "LA", "P"):
                    img.convert("RGBA").save(buf, format="PNG")
                    mime = "image/png"
                else:
                    img.convert("RGB").save(buf, format="JPEG", quality=88)
                    mime = "image/jpeg"
                raw = buf.getvalue()
    except ImportError:
        pass
    except Exception as e:
        print(f"[LLM_CLIENT][WARNING] Could not downscale {path.name}, sending as-is: {e}")
    return f"data:{mime};base64,{base64.b64encode(raw).decode('ascii')}"


def _attach_images(messages, allowed):
    """Returns a copy of messages with the images added as content parts on
    the last user message. The original list is not mutated."""
    parts = [{"type": "image_url", "image_url": {"url": _encode_image(e["path"])}} for e in allowed]
    out = [dict(m) for m in messages]
    for m in reversed(out):
        if m.get("role") == "user":
            content = m["content"]
            text_part = [{"type": "text", "text": content}] if isinstance(content, str) else list(content)
            m["content"] = text_part + parts
            return out
    out.append({"role": "user", "content": parts})
    return out


def _calculate_effective_payload(messages, max_tokens):
    """
    Blend real payload size with requested output size into a single
    comparable number for threshold routing.
    
    Raw payload size alone isn't a perfect proxy for how "complex" a call
    is — a gate/classifier call can arrive with a huge recall dump attached
    (large payload) but only ever wants a couple tokens back. Folding in
    max_tokens as a signal lets a tiny requested output pull routing toward
    the small tier even when the caller didn't set preferred_tier.
    
    The weighting ratio is derived live from the real payload itself rather
    than a hardcoded constant, calibrated so that a "reference" large call
    (payload / 4 bytes-per-token ≈ 2048 output tokens, i.e. a full memory
    pass) lands with the input and output terms roughly equal — the trough
    point where neither term dominates. Smaller real payloads shrink the
    ratio, so an unexamined default max_tokens on a genuinely small call
    doesn't get inflated into a bigger tier.
    
    The "+1" in the denominator is a cheap, negligible-at-scale guard
    against a div/0 — not a meaningful part of the math.
    
    Args:
        messages: list of message dicts [{role, content}, ...]
        max_tokens: requested output token count for this call
    
    Returns:
        float: calculated payload size (same units/scale as _measure_payload,
            for direct comparison against small_threshold/large_threshold)
    """
    real_payload = _measure_payload(messages)
    output_weight_ratio = (real_payload / 4) / (2048 + 1)
    return (real_payload + max_tokens * 4 * output_weight_ratio) / 2


def _select_backend(messages, max_tokens, preferred_tier=None):
    """
    Backend routing. Normally deterministic based on payload size; a caller
    can override this with preferred_tier to force a specific tier regardless
    of payload size.
    
    Override logic (preferred_tier is "small" or "large"):
      - If that tier's backend is configured → use it directly.
      - Else → default_backend, "default" (config-level fallback; the tier
        simply isn't available, same as if size-based routing had never
        selected it).
    
    Normal (preferred_tier is None) logic:
      - Blends real payload size with requested max_tokens via
        _calculate_effective_payload() into a single comparable number.
      - If small_backend enabled and calculated payload < small_threshold → small_backend, "small"
      - Else if large_backend enabled and calculated payload > large_threshold → large_backend, "large"
      - Else → default_backend, "default" (fallback)
    
    Disabled backends (empty string) are skipped either way. Uses settings
    loaded from settings.txt.
    
    Note: this is the config-level "tier unavailable" fallback. The separate
    request-failure fallback (tier configured but the HTTP call itself fails)
    is handled by the caller in get_completion() and is unaffected by
    preferred_tier.
    
    Args:
        messages: list of message dicts [{role, content}, ...]
        max_tokens: requested output token count, factored into routing
            alongside payload size (see _calculate_effective_payload)
        preferred_tier: optional override, one of None, "small", "large"
    
    Returns:
        tuple: (backend_url, tier_name)
          - backend_url: str in IP:port format (e.g. "127.0.0.1:5001")
          - tier_name: str in ("default", "small", "large") for settings lookup
    """
    default_backend = _SETTINGS["default_backend"]
    
    if preferred_tier is not None:
        override_backend = _SETTINGS[f"{preferred_tier}_backend"]
        if override_backend:
            return override_backend, preferred_tier
        _warn_once(preferred_tier, f"{preferred_tier}_backend is not set; "
                                   f"{preferred_tier}-tier calls are using default_backend.")
        return default_backend, "default"
    
    payload_size = _calculate_effective_payload(messages, max_tokens)
    
    small_backend = _SETTINGS["small_backend"]
    small_threshold = _SETTINGS["small_threshold"]
    large_backend = _SETTINGS["large_backend"]
    large_threshold = _SETTINGS["large_threshold"]
    
    # Small model route (if enabled)
    if small_backend and payload_size < small_threshold:
        return small_backend, "small"
    
    # Large model route (if enabled)
    if large_backend and payload_size > large_threshold:
        return large_backend, "large"
    
    # Default (medium) route
    return default_backend, "default"


def _select_thinking_mode(messages, max_tokens, preferred_tier=None):
    """
    Thinking mode routing. Thinking is now purely a function of an
    explicit preferred_tier request — the payload-size classifier has been
    removed from this decision entirely.
    
    Override logic (preferred_tier is "small" or "large"):
      - "small" → thinking_enabled = False (dumdum mode)
      - "large" → thinking_enabled = True
    This tracks the requested tier itself, not whether that tier's backend
    turned out to be configured — thinking is a property of the kind of
    work being requested, and is preserved even if _select_backend had to
    fall back to default for the actual connection.
    
    Normal (preferred_tier is None) logic:
      - Returns None. No thinking flags are sent at all (see _send_request),
        so the backend's own configured/model default applies untouched.
        A caller that needs a guaranteed thinking state for a specific
        call (e.g. a gate that must reason) should request preferred_tier
        explicitly rather than relying on this to infer it from size.
    
    Args:
        messages: list of message dicts [{role, content}, ...]
        max_tokens: requested output token count (unused when preferred_tier
            is None now, kept in the signature for call-site stability)
        preferred_tier: optional override, one of None, "small", "large"
    
    Returns:
        bool or None: True/False to force thinking on/off, or None to send
            no thinking flags and defer to the backend's own default.
    """
    if preferred_tier is not None:
        return preferred_tier == "large"
    
    # No explicit tier requested: stop guessing from payload size. Return
    # None so _send_request sends no thinking flags at all and the
    # backend's own configured default (per-model / server-side) decides.
    # Thinking is now purely a function of an explicit preferred_tier
    # request, not something the router infers.
    return None


def _measure_payload(messages):
    """
    Measure messages payload size in bytes (UTF-8 encoded).
    
    Rough token conversion: 1 token ≈ 4 bytes
    Examples:
      5,000 bytes   ≈ 1,250 tokens   (simple: keyword extraction, gates)
      50,000 bytes  ≈ 12,500 tokens  (complex: reasoning with full memories)
    
    Args:
        messages: list of message dicts
    
    Returns:
        int: size in bytes
    """
    return len(json.dumps(messages).encode('utf-8'))


def _send_request(backend, messages, max_tokens, temperature, top_p,
                   repetition_penalty, timeout, thinking_enabled=None, backend_tier="default"):
    """
    Send request to specific backend.
    Uses OpenAI-compatible /v1/chat/completions format.
    
    Automatically handles authentication: if api_key is set in settings for this tier,
    adds Authorization header. If model is set in settings, uses it; otherwise defaults to "local".
    
    Args:
        backend: backend URL (IP:port format, e.g. "127.0.0.1:5001")
        messages: list of message dicts
        max_tokens: max output length
        temperature: sampling temperature
        top_p: nucleus sampling parameter
        repetition_penalty: token repetition penalty
        timeout: request timeout in seconds
        thinking_enabled: True/False to force thinking on/off, or None to
            send no thinking flags at all and let the backend's own
            configured/model default apply untouched (default False)
        backend_tier: str in ("default", "small", "large", "specialist") to look up api_key and model from settings
    
    Returns:
        str: model response, stripped
    
    Raises:
        Exception: on any request failure (connection, timeout, bad response)
    """
    # Determine URL scheme and port based on backend
    if backend.startswith("http://") or backend.startswith("https://"):
        url = f"{backend}/v1/chat/completions"
    else:
        # Default to http for bare IP:port
        url = f"http://{backend}/v1/chat/completions"
    
    # Get api_key and model for this tier from settings
    api_key = _SETTINGS.get(f"{backend_tier}_api_key", "")
    model = _SETTINGS.get(f"{backend_tier}_model", "") or "local"
    
    # Build headers
    headers = {"Content-Type": "application/json"}
    if api_key:  # Non-empty api_key means auth is required
        headers["Authorization"] = f"Bearer {api_key}"
    
    # Floor the outgoing max_tokens so a reasoning model has room to finish
    # its <think> block before the real answer, even when the caller
    # requested a tiny max_tokens (e.g. a gate/classifier call). This is a
    # wire-only adjustment — the caller's original max_tokens is what
    # routing already saw upstream and is left untouched as a signal.
    effective_max_tokens = max(max_tokens, _SETTINGS["min_output_tokens"])
    
    payload = {
        "model": model,
        "messages": messages,
        "max_tokens": effective_max_tokens,
        "temperature": temperature,
        "top_p": top_p,
    }
    # The specialist is usually a hosted API that rejects unknown fields, so
    # it only gets standard OpenAI keys. Local tiers keep the extras.
    if backend_tier != "specialist":
        payload["repetition_penalty"] = repetition_penalty
    
    # thinking_enabled is None when the caller didn't request an explicit
    # tier — in that case we send no thinking-related keys at all so the
    # backend's own configured/model default decides. Only an explicit
    # True/False (derived from preferred_tier) gets broadcast here.
    if thinking_enabled is not None and backend_tier != "specialist":
        payload["chat_template_kwargs"] = {
            # Different model templates check different key names for the
            # thinking on/off switch (Qwen3-style vs others). Templates that
            # don't reference a given key silently ignore it, so broadcasting
            # every known name is safe — only the one the template actually
            # checks has any effect.
            "enable_thinking": thinking_enabled,
            "thinking": thinking_enabled,
            "enable_reasoning": thinking_enabled,
            "reasoning": thinking_enabled,
            "preserve_reasoning": thinking_enabled,
        }
    
    r = requests.post(url, json=payload, headers=headers, timeout=timeout)
    r.raise_for_status()
    
    response_json = r.json()
    
    # Some models/templates inline reasoning as a <think>...</think> block
    # at the start of content rather than emitting it as a separate field.
    # Strip it unconditionally rather than chasing per-model field names —
    # a no-op for content that never had one.
    content = response_json["choices"][0]["message"].get("content", "").strip()
    content = re.sub(r"<think>.*?</think>", "", content, flags=re.DOTALL).strip()
    return content


def is_backend_alive(backend=None, backend_tier="default"):
    """
    Health check: is the backend listening and responding?
    
    Useful for startup validation (controller.py, dream.py) to warn if the
    inference backend isn't running before user starts typing.
    
    If backend is specified, checks that specific backend. Otherwise checks default.
    Uses a minimal timeout (2s) to fail fast if backend is down.
    Automatically includes auth header if api_key is set for this tier.
    
    Args:
        backend: specific backend to check (IP:port format). If None, checks default from settings.txt
        backend_tier: str in ("default", "small", "large", "specialist") for auth lookup
    
    Returns:
        bool: True if backend responds, False otherwise
    """
    check_backend = backend if backend is not None else _SETTINGS["default_backend"]
    
    try:
        # Determine URL scheme
        if check_backend.startswith("http://") or check_backend.startswith("https://"):
            url = f"{check_backend}/v1/chat/completions"
        else:
            url = f"http://{check_backend}/v1/chat/completions"
        
        # Get api_key for this tier if needed
        api_key = _SETTINGS.get(f"{backend_tier}_api_key", "")
        headers = {"Content-Type": "application/json"}
        if api_key:
            headers["Authorization"] = f"Bearer {api_key}"
        
        # Minimal test: single message, instant response (1 token max)
        r = requests.post(
            url,
            json={
                "model": "local",
                "messages": [{"role": "user", "content": "ok"}],
                "max_tokens": 1
            },
            headers=headers,
            timeout=2
        )
        return r.status_code == 200
    except:
        return False



# ══════════════════════════════════════════════════════════════════════════════
# Embeddings (recall's vector lane)
# ══════════════════════════════════════════════════════════════════════════════

def embedding_enabled():
    """True when an embedding backend is configured in settings.txt."""
    return bool(_SETTINGS.get("embedding_backend"))


def embedding_config():
    """
    The configured embedding route as a dict, or None when disabled.
    model_id is the identity stored in the EmbeddingMeta table: if it changes,
    every stored embedding is stale and is cleared and rebuilt.
    """
    if not embedding_enabled():
        return None
    backend = _SETTINGS["embedding_backend"]
    model = _SETTINGS.get("embedding_model") or "local"
    return {
        "backend": backend,
        "model": model,
        "model_id": f"{backend}|{model}|{_SETTINGS.get('embedding_doc_prefix', '')}|{_SETTINGS.get('embedding_query_prefix', '')}",
        "doc_prefix": _SETTINGS.get("embedding_doc_prefix", ""),
        "query_prefix": _SETTINGS.get("embedding_query_prefix", ""),
        "batch": max(1, int(_SETTINGS.get("embedding_batch_size", 16))),
    }


def get_embeddings(texts, kind="document", timeout=120):
    """
    Embeds a list of texts. kind is "document" (a stored memory) or "query"
    (a question being recalled for); the matching prefix from settings.txt is
    put in front of each text (some models, such as nomic-embed, need them).
    Returns a list of float lists in the same order. Raises on any failure, so
    callers decide whether a missing embedding is fatal (recall: it is not).
    """
    cfg = embedding_config()
    if cfg is None:
        raise RuntimeError("no embedding backend configured")
    prefix = cfg["query_prefix"] if kind == "query" else cfg["doc_prefix"]
    if prefix and not prefix[-1].isspace():
        prefix += " "          # settings values are stripped, so a trailing space is restored here
    backend = cfg["backend"]
    url = f"{backend}/v1/embeddings" if backend.startswith(("http://", "https://")) else f"http://{backend}/v1/embeddings"
    headers = {"Content-Type": "application/json"}
    if _SETTINGS.get("embedding_api_key"):
        headers["Authorization"] = f"Bearer {_SETTINGS['embedding_api_key']}"
    out = []
    for i in range(0, len(texts), cfg["batch"]):
        batch = [prefix + (t or "")[:3000] for t in texts[i:i + cfg["batch"]]]
        r = requests.post(url, json={"input": batch, "model": cfg["model"]}, headers=headers, timeout=timeout)
        r.raise_for_status()
        data = sorted(r.json()["data"], key=lambda d: d["index"])
        out.extend(d["embedding"] for d in data)
    return out
