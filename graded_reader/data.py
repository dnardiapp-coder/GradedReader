"""Static data describing languages, levels, and structures."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List

from . import config
from .config import Language, ProficiencyLevel
from .models import LanguageConfig, LevelProfile


@dataclass(frozen=True)
class StoryStructure:
    """Narrative scaffolds that guide story generation."""

    name: str
    description: str
    components: List[str]
    suitable_levels: List[ProficiencyLevel]


LANGUAGE_CONFIGS: Dict[Language, LanguageConfig] = {
    Language.CHINESE: LanguageConfig(
        "Chinese (Simplified)",
        "zh-CN",
        True,
        "Pinyin",
        "logographic",
        "ltr",
        str(config.FONT_CJK_PATH),
    ),
    Language.SPANISH: LanguageConfig(
        "Spanish", "es", False, "", "latin", "ltr", str(config.FONT_LATIN_PATH)
    ),
    Language.FRENCH: LanguageConfig(
        "French", "fr", False, "", "latin", "ltr", str(config.FONT_LATIN_PATH)
    ),
    Language.GERMAN: LanguageConfig(
        "German", "de", False, "", "latin", "ltr", str(config.FONT_LATIN_PATH)
    ),
    Language.ITALIAN: LanguageConfig(
        "Italian", "it", False, "", "latin", "ltr", str(config.FONT_LATIN_PATH)
    ),
    Language.PORTUGUESE: LanguageConfig(
        "Portuguese", "pt", False, "", "latin", "ltr", str(config.FONT_LATIN_PATH)
    ),
    Language.RUSSIAN: LanguageConfig(
        "Russian", "ru", True, "Transliteration", "cyrillic", "ltr", str(config.FONT_LATIN_PATH)
    ),
    Language.JAPANESE: LanguageConfig(
        "Japanese", "ja", True, "Romaji/Furigana", "mixed", "ltr", str(config.FONT_JAPANESE_PATH)
    ),
    Language.KOREAN: LanguageConfig(
        "Korean", "ko", True, "Romanization", "hangul", "ltr", str(config.FONT_CJK_PATH)
    ),
    Language.ARABIC: LanguageConfig(
        "Arabic", "ar", True, "Transliteration", "arabic", "rtl", str(config.FONT_ARABIC_PATH)
    ),
    Language.HINDI: LanguageConfig(
        "Hindi", "hi", True, "Romanization", "devanagari", "ltr", str(config.FONT_LATIN_PATH)
    ),
    Language.DUTCH: LanguageConfig(
        "Dutch", "nl", False, "", "latin", "ltr", str(config.FONT_LATIN_PATH)
    ),
}


LEVEL_PROFILES: Dict[ProficiencyLevel, LevelProfile] = {
    ProficiencyLevel.BEGINNER_1: LevelProfile(
        "Beginner 1",
        "A1",
        "Basic phrases and simple sentences",
        (5, 12),
        (3, 5),
        (0.02, 0.05),
        500,
        "Present tense, basic word order, simple questions",
        ["daily life", "family", "food", "numbers", "greetings", "colors"],
    ),
    ProficiencyLevel.BEGINNER_2: LevelProfile(
        "Beginner 2",
        "A1+",
        "Simple descriptions and daily routines",
        (8, 15),
        (4, 6),
        (0.03, 0.06),
        800,
        "Past tense (simple), basic conjunctions, possessives",
        ["school", "hobbies", "weather", "shopping", "time", "directions"],
    ),
    ProficiencyLevel.ELEMENTARY_1: LevelProfile(
        "Elementary 1",
        "A2",
        "Connected sentences and simple narratives",
        (10, 18),
        (5, 8),
        (0.04, 0.07),
        1200,
        "Future tense, comparatives, modal verbs",
        ["travel", "work", "health", "celebrations", "sports", "technology"],
    ),
    ProficiencyLevel.ELEMENTARY_2: LevelProfile(
        "Elementary 2",
        "A2+",
        "Short stories with clear plot",
        (12, 20),
        (6, 10),
        (0.05, 0.08),
        1800,
        "Conditional (basic), passive voice (intro), relative clauses",
        ["culture", "environment", "media", "relationships", "education"],
    ),
    ProficiencyLevel.INTERMEDIATE_1: LevelProfile(
        "Intermediate 1",
        "B1",
        "Detailed narratives with subplots",
        (15, 25),
        (8, 12),
        (0.06, 0.10),
        2500,
        "Perfect tenses, reported speech, complex conjunctions",
        ["society", "history", "science", "arts", "business", "psychology"],
    ),
    ProficiencyLevel.INTERMEDIATE_2: LevelProfile(
        "Intermediate 2",
        "B1+",
        "Complex stories with multiple viewpoints",
        (18, 30),
        (10, 15),
        (0.08, 0.12),
        3500,
        "Subjunctive mood, advanced passive, phrasal verbs",
        ["politics", "philosophy", "literature", "economics", "global issues"],
    ),
    ProficiencyLevel.UPPER_INTERMEDIATE: LevelProfile(
        "Upper Intermediate",
        "B2",
        "Sophisticated narratives",
        (20, 35),
        (12, 18),
        (0.10, 0.15),
        5000,
        "All tenses, idiomatic expressions, complex syntax",
        ["abstract concepts", "critical thinking", "cultural analysis", "debates"],
    ),
    ProficiencyLevel.ADVANCED: LevelProfile(
        "Advanced",
        "C1",
        "Native-like complexity",
        (25, 40),
        (15, 25),
        (0.12, 0.20),
        8000,
        "Full grammatical range, nuanced expression",
        ["academic topics", "professional discourse", "literary analysis"],
    ),
}


STORY_STRUCTURES: Dict[str, StoryStructure] = {
    "simple_narrative": StoryStructure(
        "Simple Narrative",
        "Basic linear story with clear beginning, middle, end",
        ["introduction", "event", "conclusion"],
        [ProficiencyLevel.BEGINNER_1, ProficiencyLevel.BEGINNER_2],
    ),
    "problem_solution": StoryStructure(
        "Problem-Solution",
        "Character faces a problem and finds a solution",
        ["setup", "problem_introduction", "attempts", "solution", "reflection"],
        [ProficiencyLevel.ELEMENTARY_1, ProficiencyLevel.ELEMENTARY_2],
    ),
    "hero_journey": StoryStructure(
        "Hero's Journey",
        "Classic adventure structure",
        [
            "ordinary_world",
            "call_to_adventure",
            "challenges",
            "transformation",
            "return",
        ],
        [ProficiencyLevel.INTERMEDIATE_1, ProficiencyLevel.INTERMEDIATE_2],
    ),
    "multiple_perspectives": StoryStructure(
        "Multiple Perspectives",
        "Same events from different viewpoints",
        ["event_perspective_1", "event_perspective_2", "revelation", "resolution"],
        [ProficiencyLevel.UPPER_INTERMEDIATE, ProficiencyLevel.ADVANCED],
    ),
}
