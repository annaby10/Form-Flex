"""
audio_feedback.py — FormFlex Audio Feedback Engine
Multi-modal audio alerts using the best available system:
  1. Pygame (if installed & has wheels for Python version)
  2. winsound (Windows built-in beeper, no extra install)
  3. Silent fallback (no audio device / unsupported OS)

This means audio will ALWAYS work on Windows without extra packages.
"""

import threading
import sys

# ── Try pygame first (best quality, cross-platform) ──────────────
_pygame_ok = False
try:
    import pygame
    import pygame.sndarray
    import numpy as np
    pygame.mixer.pre_init(frequency=44100, size=-16, channels=1, buffer=512)
    pygame.mixer.init()
    _pygame_ok = True
except Exception:
    pass

# ── Windows winsound fallback ─────────────────────────────────────
_winsound_ok = False
if not _pygame_ok and sys.platform == "win32":
    try:
        import winsound
        _winsound_ok = True
    except Exception:
        pass

# ── Rate-limiting: don't spam audio ──────────────────────────────
_last_played: dict = {}
_MIN_INTERVAL = 1.5   # seconds between same alert type


def _can_play(key: str) -> bool:
    import time
    now = time.time()
    if now - _last_played.get(key, 0) >= _MIN_INTERVAL:
        _last_played[key] = now
        return True
    return False


# ── Pygame beep builder ──────────────────────────────────────────
def _make_beep_pygame(freq=440, duration=0.3, volume=0.6):
    if not _pygame_ok:
        return None
    import numpy as np
    sr = 44100
    n = int(sr * duration)
    t = __import__('numpy').linspace(0, duration, n, endpoint=False)
    wave = (volume * __import__('numpy').sin(2 * 3.14159265 * freq * t) * 32767).astype(__import__('numpy').int16)
    # pygame mixer may need stereo
    try:
        wave2d = __import__('numpy').column_stack([wave, wave])
        return pygame.sndarray.make_sound(wave2d)
    except Exception:
        return pygame.sndarray.make_sound(wave)


# Pre-build pygame sounds once (silently ignored if not available)
_pygame_sounds = {}
if _pygame_ok:
    _pygame_sounds = {
        "good":    _make_beep_pygame(freq=880, duration=0.12, volume=0.4),
        "warning": _make_beep_pygame(freq=300, duration=0.35, volume=0.6),
        "rep":     _make_beep_pygame(freq=660, duration=0.1,  volume=0.5),
    }


# ── Public API ───────────────────────────────────────────────────
def play(sound_key: str):
    """
    Play audio alert by key: 'good', 'warning', 'rep'.
    Non-blocking. Falls back through pygame → winsound → silent.
    """
    if not _can_play(sound_key):
        return

    if _pygame_ok and sound_key in _pygame_sounds:
        snd = _pygame_sounds[sound_key]
        if snd and not pygame.mixer.get_busy():
            snd.play()
        return

    if _winsound_ok:
        # Windows built-in beeper — runs in separate thread to be non-blocking
        freq_map = {"good": 1000, "warning": 400, "rep": 750}
        dur_map  = {"good": 120,  "warning": 350, "rep": 80}
        freq = freq_map.get(sound_key, 600)
        dur  = dur_map.get(sound_key, 200)
        threading.Thread(
            target=winsound.Beep, args=(freq, dur), daemon=True
        ).start()
        return
    # Silent fallback — do nothing
