"""Text generation pipeline for the Graded Reader app."""
from __future__ import annotations

import json
from typing import Dict, Iterable, Optional

from openai import OpenAI

from . import config
from .data import LANGUAGE_CONFIGS, LEVEL_PROFILES, STORY_STRUCTURES
from .models import (
    ComprehensionQuestion,
    GrammarPoint,
    Paragraph,
    PedagogicalFeatures,
    StoryPackage,
    VocabularyEntry,
)
from .utils import clamp, estimate_reading_time, safe_join


class StoryGenerator:
    """Generate pedagogically sound graded readers via OpenAI models."""

    def __init__(self, client: Optional[OpenAI], model: str = config.DEFAULT_TEXT_MODEL):
        self.client = client
        self.model = model

    def generate_story(
        self,
        language,
        level,
        topic: str,
        subtopics: Iterable[str],
        structure_key: str,
        story_length: int,
        features: PedagogicalFeatures,
        temperature: float = 0.7,
    ) -> StoryPackage:
        """Return a :class:`StoryPackage`.

        If an OpenAI client is unavailable or generation fails the function falls
        back to deterministic sample content so that the UI remains interactive.
        """

        lang_config = LANGUAGE_CONFIGS[language]
        level_profile = LEVEL_PROFILES[level]

        if not self.client:
            return self._fallback_story(language, level, topic, story_length)

        prompt = self._build_prompt(
            language,
            level,
            topic,
            subtopics,
            structure_key,
            story_length,
            features,
        )

        try:
            response = self.client.chat.completions.create(
                model=self.model,
                temperature=temperature,
                response_format={"type": "json_object"},
                messages=[
                    {"role": "system", "content": prompt},
                    {
                        "role": "user",
                        "content": "Produce the complete graded reader package as specified.",
                    },
                ],
                max_tokens=6000,
            )
            raw_payload = response.choices[0].message.content
            payload = json.loads(raw_payload)
            return self._parse_story_payload(payload, language, level)
        except Exception:
            return self._fallback_story(language, level, topic, story_length)

    # ------------------------------------------------------------------
    def _build_prompt(
        self,
        language,
        level,
        topic: str,
        subtopics: Iterable[str],
        structure_key: str,
        story_length: int,
        features: PedagogicalFeatures,
    ) -> str:
        lang_config = LANGUAGE_CONFIGS[language]
        level_profile = LEVEL_PROFILES[level]
        structure = STORY_STRUCTURES.get(structure_key, next(iter(STORY_STRUCTURES.values())))

        subtopic_text = ", ".join(subtopics) if subtopics else "Author's choice"
        optional_sections = [
            "- Provide a curated vocabulary list with part of speech and examples"
            if features.vocabulary_preview
            else "",
            "- Highlight two or three grammar points that align with the level"
            if features.grammar_focus
            else "",
            "- Write five comprehension questions mixing open and multiple choice"
            if features.comprehension_questions
            else "",
            "- Add cultural notes that enrich understanding without stereotyping"
            if features.cultural_notes
            else "",
            "- Suggest discussion prompts that encourage personal connection"
            if features.discussion_prompts
            else "",
            "- Propose a short writing task reinforcing the focus grammar"
            if features.writing_exercises
            else "",
        ]

        prompt = f"""
You are a seasoned editor of graded readers for language learners. Craft a pedagogically sound
story in {lang_config.name} for learners at the {level_profile.name} ({level_profile.cefr_level}) level.

Story design requirements:
- Target word count: {story_length} ± 15%
- Topic: {topic}
- Sub-topics to weave in: {subtopic_text}
- Narrative structure: {structure.name} — {structure.description} (components: {', '.join(structure.components)})
- Maintain sentence length between {level_profile.sentence_len[0]} and {level_profile.sentence_len[1]} words.
- Limit new vocabulary to {int(level_profile.new_word_pct[0]*100)}–{int(level_profile.new_word_pct[1]*100)}% of the story.
- Focus grammar: {level_profile.grammar_complexity}
- Recycle high-frequency words and include accessible dialogue.
- Ensure content is culturally respectful and engaging.
{safe_join(optional_sections)}

Output JSON with the following structure:
{{
  "title": "Title in target language",
  "title_translated": "English title",
  "summary": "English summary (2 sentences)",
  "difficulty_score": 0.0-1.0,
  "story": [
    {{
      "paragraph_id": 1,
      "text": "Paragraph text",
      "romanization": "If applicable",
      "translation": "English translation",
      "audio_cues": "Performance notes"
    }}
  ],
  "vocabulary": [
    {{
      "term": "Word from story",
      "romanization": "If applicable",
      "translation": "English meaning",
      "part_of_speech": "noun/verb/etc",
      "example": "Short example sentence",
      "frequency_rank": 1-{level_profile.vocab_size}
    }}
  ],
  "grammar_points": [
    {{
      "structure": "Grammar focus",
      "explanation": "Plain-language explanation",
      "examples": ["Example using the story context"],
      "practice": "Simple controlled practice task"
    }}
  ],
  "comprehension_questions": [
    {{
      "question": "Question in target language",
      "question_english": "English support",
      "type": "multiple_choice|true_false|open_ended",
      "options": ["A", "B", "C"],
      "answer": "Correct answer (text or option)",
      "explanation": "Why the answer is correct"
    }}
  ],
  "cultural_notes": [{{"topic": "", "explanation": ""}}],
  "discussion_prompts": ["Prompt"],
  "writing_tasks": [{{"prompt": "", "word_limit": "", "focus": ""}}]
}}

Respond only with valid JSON.
"""
        return prompt

    # ------------------------------------------------------------------
    def _parse_story_payload(self, payload: Dict, language, level) -> StoryPackage:
        paragraphs = [
            Paragraph(
                paragraph_id=item.get("paragraph_id", idx + 1),
                text=item.get("text", ""),
                romanization=item.get("romanization"),
                translation=item.get("translation"),
                audio_cues=item.get("audio_cues"),
            )
            for idx, item in enumerate(payload.get("story", []))
            if item.get("text")
        ]

        vocabulary = [
            VocabularyEntry(
                term=item.get("term", ""),
                translation=item.get("translation", ""),
                part_of_speech=item.get("part_of_speech"),
                romanization=item.get("romanization"),
                example=item.get("example"),
                frequency_rank=item.get("frequency_rank"),
            )
            for item in payload.get("vocabulary", [])
            if item.get("term")
        ]

        grammar_points = [
            GrammarPoint(
                structure=item.get("structure", ""),
                explanation=item.get("explanation", ""),
                examples=item.get("examples", []) or [],
                practice=item.get("practice"),
            )
            for item in payload.get("grammar_points", [])
            if item.get("structure")
        ]

        questions = [
            ComprehensionQuestion(
                question=item.get("question", ""),
                question_english=item.get("question_english"),
                type=item.get("type", "open_ended"),
                options=item.get("options"),
                answer=item.get("answer"),
                explanation=item.get("explanation"),
            )
            for item in payload.get("comprehension_questions", [])
            if item.get("question")
        ]

        level_profile = LEVEL_PROFILES[level]
        text_body = "\n\n".join(paragraph.text for paragraph in paragraphs)
        reading_minutes = payload.get(
            "estimated_reading_time",
            estimate_reading_time(text_body, level_profile),
        )

        difficulty = clamp(float(payload.get("difficulty_score", 0.5)), 0.0, 1.0)

        return StoryPackage(
            language=language,
            level=level,
            title=payload.get("title", "Untitled Reader"),
            title_translated=payload.get("title_translated"),
            summary=payload.get("summary"),
            estimated_reading_time=int(reading_minutes),
            difficulty_score=difficulty,
            story=paragraphs,
            vocabulary=vocabulary,
            grammar_points=grammar_points,
            comprehension_questions=questions,
            cultural_notes=payload.get("cultural_notes", []),
            discussion_prompts=payload.get("discussion_prompts", []),
            writing_tasks=payload.get("writing_tasks", []),
        )

    # ------------------------------------------------------------------
    def _fallback_story(self, language, level, topic: str, story_length: int) -> StoryPackage:
        level_profile = LEVEL_PROFILES[level]
        paragraphs = [
            Paragraph(
                paragraph_id=1,
                text=(
                    f"This is a placeholder {level_profile.cefr_level} story about {topic}. "
                    "Set the OPENAI_API_KEY to unlock rich AI-generated content."
                ),
                translation="Placeholder translation.",
            )
        ]

        vocabulary = [
            VocabularyEntry(term="placeholder", translation="example"),
        ]

        grammar_points = [
            GrammarPoint(
                structure="Example grammar",
                explanation="Demonstrates how grammar points will appear in the real output.",
            )
        ]

        return StoryPackage(
            language=language,
            level=level,
            title=f"Sample Reader about {topic}",
            title_translated="Sample reader",
            summary="Fallback content because no OpenAI credentials were provided.",
            estimated_reading_time=max(1, story_length // 120),
            difficulty_score=0.2,
            story=paragraphs,
            vocabulary=vocabulary,
            grammar_points=grammar_points,
            comprehension_questions=[],
        )
