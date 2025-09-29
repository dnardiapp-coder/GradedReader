"""Utility helpers for the Graded Reader app."""
from __future__ import annotations

import re
from typing import Iterable

from .models import LevelProfile


SCRIPT_PATTERNS = {
    "chinese": r"[\u4e00-\u9fff]",
    "japanese": r"[\u3040-\u30ff]",
    "korean": r"[\uac00-\ud7af]",
    "arabic": r"[\u0600-\u06ff]",
    "cyrillic": r"[\u0400-\u04ff]",
    "devanagari": r"[\u0900-\u097f]",
}


def detect_script(text: str) -> str:
    """Return the dominant script present in *text*."""

    for script, pattern in SCRIPT_PATTERNS.items():
        if re.search(pattern, text):
            return script
    return "latin"


def estimate_reading_time(text: str, level: LevelProfile) -> int:
    """Estimate reading time in minutes based on CEFR level."""

    word_count = len(text.split())
    speeds = {
        "A1": 60,
        "A1+": 70,
        "A2": 90,
        "A2+": 110,
        "B1": 130,
        "B1+": 150,
        "B2": 170,
        "C1": 200,
    }
    wpm = speeds.get(level.cefr_level, 120)
    return max(1, round(word_count / wpm))


def clamp(value: float, lower: float, upper: float) -> float:
    """Clamp *value* to the inclusive interval [lower, upper]."""

    return max(lower, min(upper, value))


def safe_join(parts: Iterable[str], delimiter: str = "\n") -> str:
    """Join an iterable of strings filtering out falsy values."""

    return delimiter.join(part for part in parts if part)
