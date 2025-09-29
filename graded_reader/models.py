"""Data models used by the Graded Reader application."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional

from .config import Language, ProficiencyLevel


@dataclass(frozen=True)
class LanguageConfig:
    """Metadata describing a supported language."""

    name: str
    code: str
    uses_romanization: bool
    romanization_name: str
    script_type: str
    direction: str = "ltr"
    font_path: Optional[str] = None


@dataclass(frozen=True)
class LevelProfile:
    """Pedagogical profile for a proficiency level."""

    name: str
    cefr_level: str
    description: str
    sentence_len: tuple[int, int]
    paragraph_len: tuple[int, int]
    new_word_pct: tuple[float, float]
    vocab_size: int
    grammar_complexity: str
    themes: List[str] = field(default_factory=list)


@dataclass
class Paragraph:
    """A story paragraph with optional annotations."""

    paragraph_id: int
    text: str
    romanization: Optional[str] = None
    translation: Optional[str] = None
    audio_cues: Optional[str] = None


@dataclass
class VocabularyEntry:
    """Vocabulary item appearing in the graded reader."""

    term: str
    translation: str
    part_of_speech: Optional[str] = None
    romanization: Optional[str] = None
    example: Optional[str] = None
    frequency_rank: Optional[int] = None


@dataclass
class GrammarPoint:
    """Grammar focus section."""

    structure: str
    explanation: str
    examples: List[str] = field(default_factory=list)
    practice: Optional[str] = None


@dataclass
class ComprehensionQuestion:
    """Comprehension questions for the reader."""

    question: str
    question_english: Optional[str] = None
    type: str = "open_ended"
    options: Optional[List[str]] = None
    answer: Optional[str] = None
    explanation: Optional[str] = None


@dataclass
class StoryPackage:
    """Container for the generated story and pedagogy."""

    language: Language
    level: ProficiencyLevel
    title: str
    title_translated: Optional[str]
    summary: Optional[str]
    estimated_reading_time: int
    difficulty_score: float
    story: List[Paragraph]
    vocabulary: List[VocabularyEntry]
    grammar_points: List[GrammarPoint]
    comprehension_questions: List[ComprehensionQuestion] = field(default_factory=list)
    cultural_notes: List[Dict[str, str]] = field(default_factory=list)
    discussion_prompts: List[str] = field(default_factory=list)
    writing_tasks: List[Dict[str, str]] = field(default_factory=list)

    @property
    def primary_text(self) -> str:
        """Return the full story text for downstream tasks."""

        return "\n\n".join(paragraph.text for paragraph in self.story)


@dataclass
class PedagogicalFeatures:
    """Optional pedagogical extras that can be toggled per reader."""

    comprehension_questions: bool = True
    vocabulary_preview: bool = True
    grammar_focus: bool = True
    cultural_notes: bool = False
    discussion_prompts: bool = False
    writing_exercises: bool = False
    listening_tasks: bool = True
