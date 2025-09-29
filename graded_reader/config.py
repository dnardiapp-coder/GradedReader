"""Configuration constants and enumerations for the Graded Reader app."""
from __future__ import annotations

import os
from enum import Enum
from pathlib import Path

DEFAULT_TTS_MODEL = os.getenv("OPENAI_TTS_MODEL", "tts-1")
DEFAULT_TTS_VOICE = os.getenv("OPENAI_TTS_VOICE", "alloy")
DEFAULT_TEXT_MODEL = os.getenv("OPENAI_TEXT_MODEL", "gpt-4o-mini")

FONT_DIR = Path(os.getenv("FONT_DIR", "fonts"))
FONT_LATIN_PATH = Path(os.getenv("FONT_PATH_LATIN", FONT_DIR / "DejaVuSans.ttf"))
FONT_CJK_PATH = Path(os.getenv("FONT_PATH_CJK", FONT_DIR / "NotoSansSC-Regular.ttf"))
FONT_ARABIC_PATH = Path(os.getenv("FONT_PATH_ARABIC", FONT_DIR / "NotoSansArabic-Regular.ttf"))
FONT_JAPANESE_PATH = Path(os.getenv("FONT_PATH_JAPANESE", FONT_DIR / "NotoSansJP-Regular.ttf"))

PDF_FONT_NAME = "AppSans"
PDF_FONT_FALLBACK = "AppSansFallback"


class Language(Enum):
    """Supported target languages for the graded readers."""

    CHINESE = "Chinese (Simplified)"
    SPANISH = "Spanish"
    FRENCH = "French"
    GERMAN = "German"
    ITALIAN = "Italian"
    PORTUGUESE = "Portuguese"
    RUSSIAN = "Russian"
    JAPANESE = "Japanese"
    KOREAN = "Korean"
    ARABIC = "Arabic"
    HINDI = "Hindi"
    DUTCH = "Dutch"


class ProficiencyLevel(Enum):
    """Internal representation of proficiency bands."""

    BEGINNER_1 = "Beginner 1"
    BEGINNER_2 = "Beginner 2"
    ELEMENTARY_1 = "Elementary 1"
    ELEMENTARY_2 = "Elementary 2"
    INTERMEDIATE_1 = "Intermediate 1"
    INTERMEDIATE_2 = "Intermediate 2"
    UPPER_INTERMEDIATE = "Upper Intermediate"
    ADVANCED = "Advanced"


class StoryLength(Enum):
    """Story length buckets exposed to users."""

    SHORT = (400, "Short (~400 words)")
    MEDIUM = (800, "Medium (~800 words)")
    LONG = (1200, "Long (~1200 words)")

    @property
    def target_words(self) -> int:
        return self.value[0]

    @property
    def label(self) -> str:
        return self.value[1]
