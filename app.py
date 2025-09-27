import os
import io
import re
import json
import base64
import zipfile
import random
from dataclasses import dataclass, field
from typing import List, Dict, Tuple, Optional, Any
from enum import Enum

import streamlit as st
from fpdf import FPDF
from openai import OpenAI

# =========================
# ======= CONFIG ==========
# =========================

DEFAULT_TTS_MODEL = os.getenv("OPENAI_TTS_MODEL", "tts-1")
DEFAULT_TTS_VOICE = os.getenv("OPENAI_TTS_VOICE", "alloy")
DEFAULT_TEXT_MODEL = os.getenv("OPENAI_TEXT_MODEL", "gpt-4o-mini")

FONT_DIR = os.getenv("FONT_DIR", "fonts")
FONT_LATIN_PATH = os.getenv("FONT_PATH_LATIN", os.path.join(FONT_DIR, "DejaVuSans.ttf"))
FONT_CJK_PATH = os.getenv("FONT_PATH_CJK", os.path.join(FONT_DIR, "NotoSansSC-Regular.ttf"))
FONT_ARABIC_PATH = os.getenv("FONT_PATH_ARABIC", os.path.join(FONT_DIR, "NotoSansArabic-Regular.ttf"))
FONT_JAPANESE_PATH = os.getenv("FONT_PATH_JAPANESE", os.path.join(FONT_DIR, "NotoSansJP-Regular.ttf"))

if not os.path.exists(FONT_CJK_PATH):
    FONT_CJK_PATH = os.path.join(FONT_DIR, "NotoSansSC-Regular.otf")

PDF_FONT_NAME = "AppSans"
MAX_STORIES = 20

# =========================
# ===== LANGUAGES =========
# =========================

class Language(Enum):
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

@dataclass
class LanguageConfig:
    name: str
    code: str
    uses_romanization: bool
    romanization_name: str
    script_type: str
    direction: str = "ltr"
    font_path: Optional[str] = None
    
LANGUAGE_CONFIGS = {
    Language.CHINESE: LanguageConfig(
        "Chinese (Simplified)", "zh-CN", True, "Pinyin", "logographic", "ltr", FONT_CJK_PATH
    ),
    Language.SPANISH: LanguageConfig(
        "Spanish", "es", False, "", "latin", "ltr", FONT_LATIN_PATH
    ),
    Language.FRENCH: LanguageConfig(
        "French", "fr", False, "", "latin", "ltr", FONT_LATIN_PATH
    ),
    Language.GERMAN: LanguageConfig(
        "German", "de", False, "", "latin", "ltr", FONT_LATIN_PATH
    ),
    Language.ITALIAN: LanguageConfig(
        "Italian", "it", False, "", "latin", "ltr", FONT_LATIN_PATH
    ),
    Language.PORTUGUESE: LanguageConfig(
        "Portuguese", "pt", False, "", "latin", "ltr", FONT_LATIN_PATH
    ),
    Language.RUSSIAN: LanguageConfig(
        "Russian", "ru", True, "Transliteration", "cyrillic", "ltr", FONT_LATIN_PATH
    ),
    Language.JAPANESE: LanguageConfig(
        "Japanese", "ja", True, "Romaji/Furigana", "mixed", "ltr", FONT_JAPANESE_PATH
    ),
    Language.KOREAN: LanguageConfig(
        "Korean", "ko", True, "Romanization", "hangul", "ltr", FONT_CJK_PATH
    ),
    Language.ARABIC: LanguageConfig(
        "Arabic", "ar", True, "Transliteration", "arabic", "rtl", FONT_ARABIC_PATH
    ),
    Language.HINDI: LanguageConfig(
        "Hindi", "hi", True, "Romanization", "devanagari", "ltr", FONT_LATIN_PATH
    ),
    Language.DUTCH: LanguageConfig(
        "Dutch", "nl", False, "", "latin", "ltr", FONT_LATIN_PATH
    ),
}

# =========================
# ======= LEVELS ==========
# =========================

@dataclass
class LevelProfile:
    name: str
    cefr_level: str
    description: str
    sentence_len: Tuple[int, int]
    paragraph_len: Tuple[int, int]
    new_word_pct: Tuple[float, float]
    vocab_size: int
    grammar_complexity: str
    themes: List[str] = field(default_factory=list)
    
class ProficiencyLevel(Enum):
    BEGINNER_1 = "Beginner 1"
    BEGINNER_2 = "Beginner 2"
    ELEMENTARY_1 = "Elementary 1"
    ELEMENTARY_2 = "Elementary 2"
    INTERMEDIATE_1 = "Intermediate 1"
    INTERMEDIATE_2 = "Intermediate 2"
    UPPER_INTERMEDIATE = "Upper Intermediate"
    ADVANCED = "Advanced"

LEVEL_PROFILES = {
    ProficiencyLevel.BEGINNER_1: LevelProfile(
        "Beginner 1", "A1", "Basic phrases and simple sentences",
        (5, 12), (3, 5), (0.02, 0.05), 500,
        "Present tense, basic word order, simple questions",
        ["daily life", "family", "food", "numbers", "greetings", "colors"]
    ),
    ProficiencyLevel.BEGINNER_2: LevelProfile(
        "Beginner 2", "A1+", "Simple descriptions and daily routines",
        (8, 15), (4, 6), (0.03, 0.06), 800,
        "Past tense (simple), basic conjunctions, possessives",
        ["school", "hobbies", "weather", "shopping", "time", "directions"]
    ),
    ProficiencyLevel.ELEMENTARY_1: LevelProfile(
        "Elementary 1", "A2", "Connected sentences and simple narratives",
        (10, 18), (5, 8), (0.04, 0.07), 1200,
        "Future tense, comparatives, modal verbs",
        ["travel", "work", "health", "celebrations", "sports", "technology"]
    ),
    ProficiencyLevel.ELEMENTARY_2: LevelProfile(
        "Elementary 2", "A2+", "Short stories with clear plot",
        (12, 20), (6, 10), (0.05, 0.08), 1800,
        "Conditional (basic), passive voice (intro), relative clauses",
        ["culture", "environment", "media", "relationships", "education"]
    ),
    ProficiencyLevel.INTERMEDIATE_1: LevelProfile(
        "Intermediate 1", "B1", "Detailed narratives with subplots",
        (15, 25), (8, 12), (0.06, 0.10), 2500,
        "Perfect tenses, reported speech, complex conjunctions",
        ["society", "history", "science", "arts", "business", "psychology"]
    ),
    ProficiencyLevel.INTERMEDIATE_2: LevelProfile(
        "Intermediate 2", "B1+", "Complex stories with multiple viewpoints",
        (18, 30), (10, 15), (0.08, 0.12), 3500,
        "Subjunctive mood, advanced passive, phrasal verbs",
        ["politics", "philosophy", "literature", "economics", "global issues"]
    ),
    ProficiencyLevel.UPPER_INTERMEDIATE: LevelProfile(
        "Upper Intermediate", "B2", "Sophisticated narratives",
        (20, 35), (12, 18), (0.10, 0.15), 5000,
        "All tenses, idiomatic expressions, complex syntax",
        ["abstract concepts", "critical thinking", "cultural analysis", "debates"]
    ),
    ProficiencyLevel.ADVANCED: LevelProfile(
        "Advanced", "C1", "Native-like complexity",
        (25, 40), (15, 25), (0.12, 0.20), 8000,
        "Full grammatical range, nuanced expression",
        ["academic topics", "professional discourse", "literary analysis"]
    ),
}

# =========================
# === STORY STRUCTURES ====
# =========================

@dataclass
class StoryStructure:
    name: str
    description: str
    components: List[str]
    suitable_levels: List[ProficiencyLevel]

STORY_STRUCTURES = {
    "simple_narrative": StoryStructure(
        "Simple Narrative",
        "Basic linear story with clear beginning, middle, end",
        ["introduction", "event", "conclusion"],
        [ProficiencyLevel.BEGINNER_1, ProficiencyLevel.BEGINNER_2]
    ),
    "problem_solution": StoryStructure(
        "Problem-Solution",
        "Character faces a problem and finds a solution",
        ["setup", "problem_introduction", "attempts", "solution", "reflection"],
        [ProficiencyLevel.ELEMENTARY_1, ProficiencyLevel.ELEMENTARY_2]
    ),
    "hero_journey": StoryStructure(
        "Hero's Journey",
        "Classic adventure structure",
        ["ordinary_world", "call_to_adventure", "challenges", "transformation", "return"],
        [ProficiencyLevel.INTERMEDIATE_1, ProficiencyLevel.INTERMEDIATE_2]
    ),
    "multiple_perspectives": StoryStructure(
        "Multiple Perspectives",
        "Same events from different viewpoints",
        ["event_perspective_1", "event_perspective_2", "revelation", "resolution"],
        [ProficiencyLevel.UPPER_INTERMEDIATE, ProficiencyLevel.ADVANCED]
    ),
}

# =========================
# === PEDAGOGICAL FEATURES =
# =========================

@dataclass
class PedagogicalFeatures:
    comprehension_questions: bool = True
    vocabulary_preview: bool = True
    grammar_focus: bool = True
    cultural_notes: bool = False
    discussion_prompts: bool = False
    writing_exercises: bool = False
    listening_tasks: bool = True
    
# =========================
# ====== UTILITIES ========
# =========================

def detect_script(text: str) -> str:
    """Detect the primary script used in the text"""
    if re.search(r'[\u4e00-\u9fff]', text):
        return "chinese"
    elif re.search(r'[\u3040-\u309f\u30a0-\u30ff]', text):
        return "japanese"
    elif re.search(r'[\uac00-\ud7af]', text):
        return "korean"
    elif re.search(r'[\u0600-\u06ff]', text):
        return "arabic"
    elif re.search(r'[\u0400-\u04ff]', text):
        return "cyrillic"
    elif re.search(r'[\u0900-\u097f]', text):
        return "devanagari"
    return "latin"

def estimate_reading_time(text: str, level: LevelProfile) -> int:
    """Estimate reading time in minutes based on text and level"""
    word_count = len(text.split())
    # Reading speed varies by level (words per minute)
    speeds = {
        "A1": 50, "A1+": 60, "A2": 80, "A2+": 100,
        "B1": 120, "B1+": 140, "B2": 160, "C1": 200
    }
    wpm = speeds.get(level.cefr_level, 100)
    return max(1, round(word_count / wpm))

# =========================
# ====== OPENAI GEN =======
# =========================

class StoryGenerator:
    def __init__(self, client: Optional[OpenAI], model: str = DEFAULT_TEXT_MODEL):
        self.client = client
        self.model = model
        
    def generate_story(
        self,
        language: Language,
        level: ProficiencyLevel,
        topic: str,
        subtopics: List[str],
        structure: str,
        target_length: int,
        romanization: bool,
        features: PedagogicalFeatures,
        temperature: float = 0.7
    ) -> Dict[str, Any]:
        """Generate a complete story with all pedagogical components"""
        
        if not self.client:
            return self._fallback_story(language, level, topic, target_length)
            
        prompt = self._build_comprehensive_prompt(
            language, level, topic, subtopics, structure,
            target_length, romanization, features
        )
        
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": prompt},
                    {"role": "user", "content": "Generate the complete story package as specified."}
                ],
                response_format={"type": "json_object"},
                temperature=temperature,
                max_tokens=4000
            )
            
            story_data = json.loads(response.choices[0].message.content)
            return self._validate_and_enhance(story_data, language, level, romanization)
            
        except Exception as e:
            st.warning(f"AI generation failed: {e}. Using fallback.")
            return self._fallback_story(language, level, topic, target_length)
    
    def _build_comprehensive_prompt(
        self,
        language: Language,
        level: ProficiencyLevel,
        topic: str,
        subtopics: List[str],
        structure: str,
        target_length: int,
        romanization: bool,
        features: PedagogicalFeatures
    ) -> str:
        """Build a comprehensive prompt for story generation"""
        
        lang_config = LANGUAGE_CONFIGS[language]
        level_profile = LEVEL_PROFILES[level]
        story_struct = STORY_STRUCTURES.get(structure, STORY_STRUCTURES["simple_narrative"])
        
        prompt = f"""You are an expert language learning material developer specializing in graded readers.
Create a compelling story in {lang_config.name} at {level_profile.cefr_level} level.

SPECIFICATIONS:
- Language: {lang_config.name}
- Level: {level_profile.name} ({level_profile.cefr_level})
- Description: {level_profile.description}
- Target length: {target_length} words (±15%)
- Sentence length: {level_profile.sentence_len[0]}-{level_profile.sentence_len[1]} words
- Vocabulary size limit: {level_profile.vocab_size} most common words
- New vocabulary rate: {level_profile.new_word_pct[0]*100:.0f}-{level_profile.new_word_pct[1]*100:.0f}%
- Grammar focus: {level_profile.grammar_complexity}

STORY REQUIREMENTS:
- Topic: {topic}
- Subtopics to incorporate: {', '.join(subtopics) if subtopics else 'freestyle'}
- Structure: {story_struct.name} - {story_struct.description}
- Components to include: {', '.join(story_struct.components)}

PEDAGOGICAL FEATURES:
{self._format_features(features, romanization, lang_config)}

OUTPUT FORMAT (JSON):
{{
    "title": "Engaging title in target language",
    "title_translated": "Title in English",
    "summary": "Brief summary in English (2-3 sentences)",
    "estimated_reading_time": {target_length // 100},
    "difficulty_score": 0.0-1.0 based on actual complexity,
    "story": [
        {{
            "paragraph_id": 1,
            "text": "Paragraph in {lang_config.name}",
            {"'romanization': 'Romanization if applicable'," if romanization and lang_config.uses_romanization else ""}
            "translation": "English translation",
            "audio_cues": "Description for expressive reading"
        }}
    ],
    "vocabulary": [
        {{
            "term": "Word/phrase from story",
            {"'romanization': 'Romanization'," if romanization and lang_config.uses_romanization else ""}
            "translation": "English meaning",
            "part_of_speech": "noun/verb/adj/etc",
            "example": "Example sentence from story",
            "frequency_rank": 1-{level_profile.vocab_size}
        }}
    ],
    "grammar_points": [
        {{
            "structure": "Grammar pattern",
            "explanation": "Clear explanation",
            "examples": ["Example 1 from story", "Example 2"],
            "practice": "Simple exercise"
        }}
    ],
    {"'comprehension_questions': [" if features.comprehension_questions else ""}
        {'''{{
            "question": "Question in target language",
            "question_english": "English translation",
            "type": "multiple_choice/true_false/open_ended",
            "options": ["A", "B", "C", "D"] if multiple choice,
            "answer": "Correct answer",
            "explanation": "Why this answer"
        }}''' if features.comprehension_questions else ""}
    {"]," if features.comprehension_questions else ""}
    {"'cultural_notes': [" if features.cultural_notes else ""}
        {'''{{
            "topic": "Cultural element",
            "explanation": "Context and significance"
        }}''' if features.cultural_notes else ""}
    {"]," if features.cultural_notes else ""}
    {"'discussion_prompts': [" if features.discussion_prompts else ""}
        {'''"Thought-provoking question 1",
        "Personal connection prompt 2"''' if features.discussion_prompts else ""}
    {"]," if features.discussion_prompts else ""}
    {"'writing_tasks': [" if features.writing_exercises else ""}
        {'''{{
            "prompt": "Writing prompt",
            "word_limit": 50-100,
            "focus": "Grammar/vocabulary focus"
        }}''' if features.writing_exercises else ""}
    {"]" if features.writing_exercises else ""}
}}

QUALITY GUIDELINES:
1. Use natural, authentic language appropriate for the level
2. Incorporate repetition and recycling of key vocabulary
3. Build complexity gradually within the story
4. Include dialogue to practice conversational patterns
5. Ensure cultural sensitivity and age-appropriate content
6. Create engaging plot with clear character development
7. Use high-frequency vocabulary from standard frequency lists
8. Introduce new vocabulary in context with clear meaning
9. Focus on one or two grammar patterns, reinforced throughout
10. End with a satisfying conclusion that reinforces learning

Generate a complete, pedagogically sound story that learners will enjoy reading."""

        return prompt
    
    def _format_features(self, features: PedagogicalFeatures, romanization: bool, lang_config: LanguageConfig) -> str:
        """Format pedagogical features for the prompt"""
        feature_list = []
        
        if features.vocabulary_preview:
            feature_list.append("- Include vocabulary list with 10-15 key terms")
        if features.grammar_focus:
            feature_list.append("- Highlight 2-3 grammar patterns with examples")
        if features.comprehension_questions:
            feature_list.append("- Create 5-8 comprehension questions (mix of types)")
        if features.cultural_notes:
            feature_list.append("- Add 2-3 cultural insights relevant to the story")
        if features.discussion_prompts:
            feature_list.append("- Provide 3-4 discussion questions for speaking practice")
        if features.writing_exercises:
            feature_list.append("- Include 2 guided writing tasks")
        if romanization and lang_config.uses_romanization:
            feature_list.append(f"- Add {lang_config.romanization_name} for all text")
            
        return "\n".join(feature_list)
    
    def _validate_and_enhance(self, story_data: Dict, language: Language, level: ProficiencyLevel, romanization: bool) -> Dict:
        """Validate and enhance the generated story data"""
        
        # Ensure all required fields exist
        required_fields = ["title", "story", "vocabulary"]
        for field in required_fields:
            if field not in story_data:
                story_data[field] = self._get_default_field(field)
        
        # Add metadata
        story_data["metadata"] = {
            "language": language.value,
            "level": level.value,
            "cefr": LEVEL_PROFILES[level].cefr_level,
            "generated_at": str(st.session_state.get("generation_time", "unknown")),
            "word_count": sum(len(p.get("text", "").split()) for p in story_data.get("story", [])),
            "romanization_included": romanization
        }
        
        # Ensure vocabulary has proper structure
        if "vocabulary" in story_data:
            for vocab in story_data["vocabulary"]:
                if "frequency_rank" not in vocab:
                    vocab["frequency_rank"] = random.randint(1, LEVEL_PROFILES[level].vocab_size)
        
        return story_data
    
    def _fallback_story(self, language: Language, level: ProficiencyLevel, topic: str, target_length: int) -> Dict:
        """Generate a simple fallback story when AI generation fails"""
        
        # This would be expanded with actual fallback content per language
        return {
            "title": f"{topic} Story",
            "title_translated": f"{topic} Story",
            "summary": "A simple story for language learning.",
            "story": [
                {
                    "paragraph_id": 1,
                    "text": "This is a fallback story. Please check your API settings.",
                    "translation": "This is a fallback story. Please check your API settings."
                }
            ],
            "vocabulary": [],
            "grammar_points": [],
            "metadata": {
                "language": language.value,
                "level": level.value,
                "fallback": True
            }
        }
    
    def _get_default_field(self, field: str) -> Any:
        """Get default value for a missing field"""
        defaults = {
            "title": "Untitled Story",
            "story": [{"paragraph_id": 1, "text": "Story content missing.", "translation": "Story content missing."}],
            "vocabulary": [],
            "grammar_points": [],
            "comprehension_questions": []
        }
        return defaults.get(field, [])

# =========================
# ====== TTS / AUDIO ======
# =========================

class AudioGenerator:
    def __init__(self, client: Optional[OpenAI], model: str = DEFAULT_TTS_MODEL, voice: str = DEFAULT_TTS_VOICE):
        self.client = client
        self.model = model
        self.voice = voice
        
    def generate_story_audio(self, story_data: Dict, language: Language, speed: float = 1.0) -> Dict[str, bytes]:
        """Generate audio files for the story"""
        
        if not self.client:
            return {}
            
        audio_files = {}
        lang_config = LANGUAGE_CONFIGS[language]
        
        # Combine story text
        full_text = " ".join([
            para.get("text", "") for para in story_data.get("story", [])
        ])
        
        if not full_text:
            return {}
            
        try:
            # Generate normal speed audio
            normal_audio = self._synthesize_audio(full_text, speed=1.0)
            audio_files["story_normal.mp3"] = normal_audio
            
            # Generate slow speed audio if requested
            if speed < 1.0:
                slow_audio = self._synthesize_audio(full_text, speed=speed)
                audio_files["story_slow.mp3"] = slow_audio
                
            # Generate paragraph-by-paragraph audio for practice
            for i, para in enumerate(story_data.get("story", []), 1):
                if para.get("text"):
                    para_audio = self._synthesize_audio(para["text"], speed=1.0)
                    audio_files[f"paragraph_{i:02d}.mp3"] = para_audio
                    
            # Generate vocabulary audio
            for i, vocab in enumerate(story_data.get("vocabulary", []), 1):
                if vocab.get("term"):
                    vocab_audio = self._synthesize_audio(vocab["term"], speed=0.8)
                    audio_files[f"vocab_{i:02d}_{vocab['term']}.mp3"] = vocab_audio
                    
        except Exception as e:
            st.warning(f"Audio generation error: {e}")
            
        return audio_files
    
    def _synthesize_audio(self, text: str, speed: float = 1.0) -> bytes:
        """Synthesize audio from text"""
        
        try:
            response = self.client.audio.speech.create(
                model=self.model,
                voice=self.voice,
                input=text,
                speed=speed
            )
            
            # Handle different response formats
            if hasattr(response, 'content'):
                return response.content
            elif hasattr(response, 'read'):
                return response.read()
            elif isinstance(response, bytes):
                return response
            else:
                # Try to extract from response
                return bytes(response)
                
        except Exception as e:
            raise Exception(f"TTS synthesis failed: {e}")

# =========================
# ======== PDF ============
# =========================

class EnhancedPDF(FPDF):
    def __init__(self, title: str = "", subtitle: str = ""):
        super().__init__()
        self.title = self._clean_text(title)
        self.subtitle = self._clean_text(subtitle)
        self.add_font('DejaVu', '', FONT_LATIN_PATH if os.path.exists(FONT_LATIN_PATH) else '', uni=True) if os.path.exists(FONT_LATIN_PATH) else None
        
    def _clean_text(self, text: str) -> str:
        """Clean text to remove unsupported characters"""
        if not text:
            return ""
        # Replace special characters with ASCII equivalents
        replacements = {
            '•': '-',
            '▸': '>',
            '—': '-',
            '"': '"',
            '"': '"',
            ''': "'",
            ''': "'",
            '…': '...',
            '–': '-',
            '×': 'x',
            '÷': '/',
            '≈': '~',
            '≤': '<=',
            '≥': '>=',
            '≠': '!=',
            '±': '+/-',
            '°': ' degrees',
            '™': 'TM',
            '®': '(R)',
            '©': '(C)',
            '€': 'EUR',
            '£': 'GBP',
            '¥': 'JPY',
            '₹': 'INR',
        }
        for old, new in replacements.items():
            text = text.replace(old, new)
        # Remove any remaining non-ASCII characters for safety
        return ''.join(char if ord(char) < 128 else '?' for char in text)
        
    def header(self):
        if self.page_no() > 1:  # Skip header on title page
            self.set_font('Arial', 'I', 9)
            self.cell(0, 10, self.title, 0, 1, 'C')
            self.ln(5)
            
    def footer(self):
        self.set_y(-15)
        self.set_font('Arial', 'I', 8)
        self.cell(0, 10, f'Page {self.page_no()}', 0, 0, 'C')
        
    def chapter_title(self, num: int, title: str):
        self.set_font('Arial', 'B', 14)
        self.set_text_color(0, 51, 102)
        self.cell(0, 10, self._clean_text(f'Chapter {num}: {title}'), 0, 1, 'L')
        self.set_text_color(0, 0, 0)
        self.ln(4)
        
    def story_paragraph(self, text: str, translation: str = "", romanization: str = ""):
        # Main text
        self.set_font('Arial', '', 11)
        self.multi_cell(0, 6, self._clean_text(text), 0, 'J')
        
        # Romanization (if provided)
        if romanization:
            self.set_font('Arial', 'I', 9)
            self.set_text_color(100, 100, 100)
            self.multi_cell(0, 5, self._clean_text(romanization), 0, 'J')
            self.set_text_color(0, 0, 0)
            
        # Translation (if provided)
        if translation:
            self.set_font('Arial', 'I', 9)
            self.set_text_color(80, 80, 80)
            self.multi_cell(0, 5, self._clean_text(translation), 0, 'J')
            self.set_text_color(0, 0, 0)
            
        self.ln(2)
        
    def vocabulary_section(self, vocab_list: List[Dict]):
        self.set_font('Arial', 'B', 12)
        self.cell(0, 10, 'Vocabulary', 0, 1, 'L')
        self.set_font('Arial', '', 10)
        
        for vocab in vocab_list:
            term = vocab.get("term", "")
            translation = vocab.get("translation", "")
            pos = vocab.get("part_of_speech", "")
            romanization = vocab.get("romanization", "")
            
            line = f"- {term}"
            if romanization:
                line += f" [{romanization}]"
            if pos:
                line += f" ({pos})"
            if translation:
                line += f": {translation}"
                
            self.multi_cell(0, 5, self._clean_text(line), 0, 'L')
            
        self.ln(3)
        
    def grammar_section(self, grammar_points: List[Dict]):
        self.set_font('Arial', 'B', 12)
        self.cell(0, 10, 'Grammar Points', 0, 1, 'L')
        
        for point in grammar_points:
            self.set_font('Arial', 'B', 10)
            self.multi_cell(0, 5, self._clean_text(f"> {point.get('structure', '')}"), 0, 'L')
            
            self.set_font('Arial', '', 9)
            self.multi_cell(0, 5, self._clean_text(point.get('explanation', '')), 0, 'L')
            
            if point.get('examples'):
                for ex in point['examples']:
                    self.cell(10, 5, '', 0, 0)  # Indent
                    self.multi_cell(0, 5, self._clean_text(f"- {ex}"), 0, 'L')
                    
            self.ln(2)
            
    def questions_section(self, questions: List[Dict]):
        self.set_font('Arial', 'B', 12)
        self.cell(0, 10, 'Comprehension Questions', 0, 1, 'L')
        self.set_font('Arial', '', 10)
        
        for i, q in enumerate(questions, 1):
            question = q.get('question', '')
            q_english = q.get('question_english', '')
            
            self.multi_cell(0, 5, self._clean_text(f"{i}. {question}"), 0, 'L')
            if q_english:
                self.set_font('Arial', 'I', 9)
                self.set_text_color(100, 100, 100)
                self.multi_cell(0, 5, self._clean_text(f"   ({q_english})"), 0, 'L')
                self.set_text_color(0, 0, 0)
                self.set_font('Arial', '', 10)
                
            if q.get('type') == 'multiple_choice' and q.get('options'):
                for opt in q['options']:
                    self.cell(10, 5, '', 0, 0)  # Indent
                    self.cell(0, 5, self._clean_text(f"   {opt}"), 0, 1)
                    
            self.ln(2)

def create_enhanced_pdf(
    book_title: str,
    stories: List[Dict],
    language: Language,
    level: ProficiencyLevel,
    include_translations: bool = True,
    include_romanization: bool = True
) -> bytes:
    """Create an enhanced PDF with all learning materials"""
    
    pdf = EnhancedPDF(book_title, f"{language.value} - {level.value}")
    pdf.add_page()
    
    # Title page
    pdf.set_font('Arial', 'B', 24)
    pdf.cell(0, 20, pdf._clean_text(book_title), 0, 1, 'C')
    pdf.set_font('Arial', '', 14)
    pdf.cell(0, 10, pdf._clean_text(f"{language.value} - {level.value}"), 0, 1, 'C')
    pdf.cell(0, 10, f"CEFR Level: {LEVEL_PROFILES[level].cefr_level}", 0, 1, 'C')
    pdf.ln(20)
    
    # Introduction page
    pdf.add_page()
    pdf.set_font('Arial', 'B', 16)
    pdf.cell(0, 10, 'How to Use This Book', 0, 1, 'L')
    pdf.set_font('Arial', '', 11)
    
    intro_text = """This graded reader has been carefully designed to match your language learning level. 
    
Each story includes:
- Main text in the target language
- Optional romanization for pronunciation support
- English translations for comprehension
- Key vocabulary with definitions
- Grammar explanations with examples
- Comprehension questions to test understanding
- Audio files for listening practice

Learning Tips:
1. First, listen to the audio without reading
2. Read the story without looking at translations
3. Check vocabulary and grammar notes
4. Re-read with full understanding
5. Answer comprehension questions
6. Practice reading aloud with the audio

Remember: The goal is 95% comprehension. If you understand less, this level might be too advanced. 
If you understand everything easily, try the next level up!"""
    
    pdf.multi_cell(0, 5, intro_text, 0, 'J')
    
    # Table of Contents
    pdf.add_page()
    pdf.set_font('Arial', 'B', 16)
    pdf.cell(0, 10, 'Table of Contents', 0, 1, 'L')
    pdf.ln(5)
    
    toc_entries = []
    for i, story in enumerate(stories, 1):
        title = story.get('title', f'Story {i}')
        title_trans = story.get('title_translated', '')
        toc_entries.append((i, title, title_trans, pdf.page_no() + i))
    
    pdf.set_font('Arial', '', 11)
    for num, title, trans, page in toc_entries:
        pdf.cell(0, 8, pdf._clean_text(f"{num}. {title}"), 0, 1)
        if trans:
            pdf.set_font('Arial', 'I', 9)
            pdf.set_text_color(100, 100, 100)
            pdf.cell(0, 6, pdf._clean_text(f"    {trans}"), 0, 1)
            pdf.set_text_color(0, 0, 0)
            pdf.set_font('Arial', '', 11)
    
    # Stories
    for i, story in enumerate(stories, 1):
        pdf.add_page()
        
        # Chapter title
        pdf.chapter_title(i, story.get('title', f'Story {i}'))
        
        # Story metadata
        if story.get('summary'):
            pdf.set_font('Arial', 'I', 10)
            pdf.multi_cell(0, 5, pdf._clean_text(story['summary']), 0, 'J')
            pdf.ln(3)
            
        # Story content
        for para in story.get('story', []):
            pdf.story_paragraph(
                para.get('text', ''),
                para.get('translation', '') if include_translations else '',
                para.get('romanization', '') if include_romanization else ''
            )
        
        # Vocabulary section
        if story.get('vocabulary'):
            pdf.vocabulary_section(story['vocabulary'])
        
        # Grammar section
        if story.get('grammar_points'):
            pdf.grammar_section(story['grammar_points'])
        
        # Comprehension questions
        if story.get('comprehension_questions'):
            pdf.questions_section(story['comprehension_questions'])
        
        # Cultural notes
        if story.get('cultural_notes'):
            pdf.set_font('Arial', 'B', 12)
            pdf.cell(0, 10, 'Cultural Notes', 0, 1, 'L')
            pdf.set_font('Arial', '', 10)
            for note in story['cultural_notes']:
                pdf.set_font('Arial', 'B', 10)
                pdf.cell(0, 5, pdf._clean_text(f"- {note.get('topic', '')}"), 0, 1)
                pdf.set_font('Arial', '', 9)
                pdf.multi_cell(0, 5, pdf._clean_text(note.get('explanation', '')), 0, 'J')
                pdf.ln(2)
        
        # Discussion prompts
        if story.get('discussion_prompts'):
            pdf.set_font('Arial', 'B', 12)
            pdf.cell(0, 10, 'Discussion Questions', 0, 1, 'L')
            pdf.set_font('Arial', '', 10)
            for j, prompt in enumerate(story['discussion_prompts'], 1):
                pdf.multi_cell(0, 5, pdf._clean_text(f"{j}. {prompt}"), 0, 'L')
                pdf.ln(1)
    
    # Answer key (if available)
    pdf.add_page()
    pdf.set_font('Arial', 'B', 16)
    pdf.cell(0, 10, 'Answer Key', 0, 1, 'L')
    pdf.ln(3)
    
    for i, story in enumerate(stories, 1):
        if story.get('comprehension_questions'):
            pdf.set_font('Arial', 'B', 12)
            pdf.cell(0, 8, pdf._clean_text(f"Chapter {i}: {story.get('title', '')}"), 0, 1)
            pdf.set_font('Arial', '', 10)
            
            for j, q in enumerate(story['comprehension_questions'], 1):
                answer = q.get('answer', 'N/A')
                explanation = q.get('explanation', '')
                pdf.cell(0, 5, pdf._clean_text(f"{j}. {answer}"), 0, 1)
                if explanation:
                    pdf.set_font('Arial', 'I', 9)
                    pdf.cell(10, 5, '', 0, 0)  # Indent
                    pdf.multi_cell(0, 4, pdf._clean_text(explanation), 0, 'L')
                    pdf.set_font('Arial', '', 10)
            pdf.ln(2)
    
    return pdf.output(dest='S').encode('latin-1', 'ignore')

# =========================
# ===== STREAMLIT UI ======
# =========================

st.set_page_config(
    page_title="Enhanced Graded Reader Builder",
    page_icon="📚",
    layout="wide"
)

st.title("📚 Enhanced Graded Reader Builder")
st.caption("Create pedagogically-sound language learning materials with AI assistance")

# Initialize session state
if "generated_stories" not in st.session_state:
    st.session_state.generated_stories = None
if "pdf_data" not in st.session_state:
    st.session_state.pdf_data = None
if "audio_files" not in st.session_state:
    st.session_state.audio_files = None

# Sidebar configuration
with st.sidebar:
    st.header("📖 Book Configuration")
    
    # Language selection
    language = st.selectbox(
        "Target Language",
        options=list(Language),
        format_func=lambda x: x.value
    )
    
    lang_config = LANGUAGE_CONFIGS[language]
    
    # Level selection
    level = st.selectbox(
        "Proficiency Level",
        options=list(ProficiencyLevel),
        format_func=lambda x: f"{x.value} ({LEVEL_PROFILES[x].cefr_level})"
    )
    
    level_profile = LEVEL_PROFILES[level]
    
    # Display level info
    with st.expander("Level Details"):
        st.write(f"**CEFR:** {level_profile.cefr_level}")
        st.write(f"**Description:** {level_profile.description}")
        st.write(f"**Vocabulary Size:** ~{level_profile.vocab_size} words")
        st.write(f"**Themes:** {', '.join(level_profile.themes[:5])}")
    
    st.divider()
    
    # Content settings
    st.subheader("📝 Content Settings")
    
    topic = st.text_input(
        "Main Topic",
        value=random.choice(level_profile.themes)
    )
    
    subtopics = st.text_area(
        "Subtopics (one per line)",
        value="\n".join(random.sample(level_profile.themes, min(3, len(level_profile.themes))))
    )
    
    num_stories = st.slider(
        "Number of Stories",
        min_value=1,
        max_value=10,
        value=3
    )
    
    story_structure = st.selectbox(
        "Story Structure",
        options=[s for s, struct in STORY_STRUCTURES.items() 
                if level in struct.suitable_levels],
        format_func=lambda x: STORY_STRUCTURES[x].name
    )
    
    st.divider()
    
    # Pedagogical features
    st.subheader("📚 Learning Features")
    
    features = PedagogicalFeatures(
        vocabulary_preview=st.checkbox("Vocabulary Lists", value=True),
        grammar_focus=st.checkbox("Grammar Explanations", value=True),
        comprehension_questions=st.checkbox("Comprehension Questions", value=True),
        cultural_notes=st.checkbox("Cultural Notes", value=level != ProficiencyLevel.BEGINNER_1),
        discussion_prompts=st.checkbox("Discussion Questions", value=level not in [ProficiencyLevel.BEGINNER_1, ProficiencyLevel.BEGINNER_2]),
        writing_exercises=st.checkbox("Writing Tasks", value=level not in [ProficiencyLevel.BEGINNER_1, ProficiencyLevel.BEGINNER_2]),
        listening_tasks=st.checkbox("Audio Materials", value=True)
    )
    
    include_romanization = False
    if lang_config.uses_romanization:
        include_romanization = st.checkbox(
            f"Include {lang_config.romanization_name}",
            value=level in [ProficiencyLevel.BEGINNER_1, ProficiencyLevel.BEGINNER_2]
        )
    
    include_translations = st.checkbox(
        "Include English Translations",
        value=level in [ProficiencyLevel.BEGINNER_1, ProficiencyLevel.BEGINNER_2, ProficiencyLevel.ELEMENTARY_1]
    )
    
    st.divider()
    
    # Generation settings
    st.subheader("⚙️ Generation Settings")
    
    use_ai = st.checkbox("Use AI Generation", value=True)
    
    if use_ai:
        model = st.selectbox(
            "AI Model",
            options=["gpt-4o-mini", "gpt-4o", "gpt-3.5-turbo"],
            index=0
        )
        
        temperature = st.slider(
            "Creativity",
            min_value=0.0,
            max_value=1.0,
            value=0.7,
            step=0.1,
            help="Higher values make the output more creative but less predictable"
        )
    else:
        model = None
        temperature = 0.7
    
    # Audio settings
    if features.listening_tasks:
        st.subheader("🎧 Audio Settings")
        
        tts_model = st.selectbox(
            "TTS Model",
            options=["tts-1", "tts-1-hd"],
            index=0
        )
        
        tts_voice = st.selectbox(
            "Voice",
            options=["alloy", "echo", "fable", "onyx", "nova", "shimmer"],
            index=0
        )
        
        audio_speed = st.slider(
            "Slow Audio Speed",
            min_value=0.5,
            max_value=1.0,
            value=0.75,
            step=0.05,
            help="Speed for slow practice audio"
        )
    else:
        tts_model = DEFAULT_TTS_MODEL
        tts_voice = DEFAULT_TTS_VOICE
        audio_speed = 0.75
    
    st.divider()
    
    # API Key
    api_key = st.text_input(
        "OpenAI API Key",
        type="password",
        help="Required for AI generation and audio"
    )
    
    # Generate button
    generate_button = st.button(
        "🚀 Generate Reader",
        type="primary",
        use_container_width=True
    )

# Main content area
col1, col2 = st.columns([2, 1])

with col1:
    st.header("📖 Generated Content")
    
    if generate_button:
        if not api_key and use_ai:
            st.error("Please provide an OpenAI API key for AI generation")
        else:
            with st.spinner("Generating your graded reader..."):
                try:
                    # Initialize generators
                    client = OpenAI(api_key=api_key) if api_key else None
                    story_gen = StoryGenerator(client, model if use_ai else None)
                    audio_gen = AudioGenerator(client, tts_model, tts_voice) if features.listening_tasks else None
                    
                    # Generate stories
                    stories = []
                    progress = st.progress(0)
                    
                    for i in range(num_stories):
                        progress.progress((i + 1) / (num_stories + 2))
                        
                        # Calculate target length based on level
                        min_words = level_profile.paragraph_len[0] * level_profile.sentence_len[0]
                        max_words = level_profile.paragraph_len[1] * level_profile.sentence_len[1]
                        target_length = random.randint(min_words, max_words)
                        
                        # Generate story
                        story = story_gen.generate_story(
                            language=language,
                            level=level,
                            topic=topic,
                            subtopics=subtopics.split('\n') if subtopics else [],
                            structure=story_structure,
                            target_length=target_length,
                            romanization=include_romanization,
                            features=features,
                            temperature=temperature
                        )
                        
                        stories.append(story)
                    
                    # Generate PDF
                    progress.progress((num_stories + 1) / (num_stories + 2))
                    pdf_data = create_enhanced_pdf(
                        book_title=f"{topic} - {language.value} Reader",
                        stories=stories,
                        language=language,
                        level=level,
                        include_translations=include_translations,
                        include_romanization=include_romanization
                    )
                    
                    # Generate audio if enabled
                    audio_files = {}
                    if audio_gen and features.listening_tasks:
                        progress.progress(1.0)
                        for i, story in enumerate(stories):
                            story_audio = audio_gen.generate_story_audio(
                                story,
                                language,
                                audio_speed
                            )
                            for filename, audio_data in story_audio.items():
                                audio_files[f"story_{i+1:02d}_{filename}"] = audio_data
                    
                    # Store in session state
                    st.session_state.generated_stories = stories
                    st.session_state.pdf_data = pdf_data
                    st.session_state.audio_files = audio_files
                    st.session_state.generation_time = pd.Timestamp.now()
                    
                    st.success("✅ Graded reader generated successfully!")
                    
                except Exception as e:
                    st.error(f"Generation failed: {str(e)}")
    
    # Display generated content
    if st.session_state.generated_stories:
        stories = st.session_state.generated_stories
        
        # Story tabs
        story_tabs = st.tabs([f"Story {i+1}" for i in range(len(stories))])
        
        for i, (tab, story) in enumerate(zip(story_tabs, stories)):
            with tab:
                # Title and metadata
                st.subheader(story.get('title', f'Story {i+1}'))
                if story.get('title_translated'):
                    st.caption(story['title_translated'])
                
                if story.get('summary'):
                    st.info(story['summary'])
                
                # Story content
                st.markdown("### Story Text")
                for para in story.get('story', []):
                    with st.expander(f"Paragraph {para.get('paragraph_id', '?')}", expanded=True):
                        st.write(para.get('text', ''))
                        if include_romanization and para.get('romanization'):
                            st.caption(para['romanization'])
                        if include_translations and para.get('translation'):
                            st.write(f"*{para['translation']}*")
                
                # Vocabulary
                if story.get('vocabulary'):
                    st.markdown("### Vocabulary")
                    vocab_cols = st.columns(2)
                    for j, vocab in enumerate(story['vocabulary']):
                        with vocab_cols[j % 2]:
                            term = vocab.get('term', '')
                            if vocab.get('romanization'):
                                term += f" [{vocab['romanization']}]"
                            st.write(f"**{term}** - {vocab.get('translation', '')}")
                
                # Grammar points
                if story.get('grammar_points'):
                    st.markdown("### Grammar Points")
                    for point in story['grammar_points']:
                        with st.expander(point.get('structure', '')):
                            st.write(point.get('explanation', ''))
                            if point.get('examples'):
                                st.write("**Examples:**")
                                for ex in point['examples']:
                                    st.write(f"- {ex}")
                
                # Comprehension questions
                if story.get('comprehension_questions'):
                    st.markdown("### Comprehension Questions")
                    for q in story['comprehension_questions']:
                        st.write(f"**{q.get('question', '')}**")
                        if q.get('question_english'):
                            st.caption(q['question_english'])
                        
                        if q.get('type') == 'multiple_choice' and q.get('options'):
                            correct = st.radio(
                                "Select answer:",
                                options=q['options'],
                                key=f"q_{i}_{q['question'][:20]}"
                            )
                            if st.button(f"Check Answer", key=f"check_{i}_{q['question'][:20]}"):
                                if correct == q.get('answer'):
                                    st.success("Correct! " + q.get('explanation', ''))
                                else:
                                    st.error(f"Try again. Hint: {q.get('explanation', '')}")

with col2:
    st.header("📥 Downloads")
    
    if st.session_state.pdf_data:
        st.download_button(
            label="📕 Download PDF",
            data=st.session_state.pdf_data,
            file_name=f"{topic.replace(' ', '_').lower()}_reader.pdf",
            mime="application/pdf",
            use_container_width=True
        )
    
    if st.session_state.audio_files:
        # Create zip file with all audio
        import io
        import zipfile
        
        zip_buffer = io.BytesIO()
        with zipfile.ZipFile(zip_buffer, 'w', zipfile.ZIP_DEFLATED) as zip_file:
            for filename, audio_data in st.session_state.audio_files.items():
                zip_file.writestr(filename, audio_data)
        
        st.download_button(
            label="🎧 Download Audio (ZIP)",
            data=zip_buffer.getvalue(),
            file_name=f"{topic.replace(' ', '_').lower()}_audio.zip",
            mime="application/zip",
            use_container_width=True
        )
    
    if st.session_state.generated_stories:
        # Export as JSON for further processing
        stories_json = json.dumps(st.session_state.generated_stories, ensure_ascii=False, indent=2)
        st.download_button(
            label="📄 Download JSON Data",
            data=stories_json,
            file_name=f"{topic.replace(' ', '_').lower()}_data.json",
            mime="application/json",
            use_container_width=True
        )
    
    # Statistics
    if st.session_state.generated_stories:
        st.divider()
        st.subheader("📊 Statistics")
        
        total_words = sum(
            sum(len(p.get('text', '').split()) for p in story.get('story', []))
            for story in st.session_state.generated_stories
        )
        total_vocab = sum(
            len(story.get('vocabulary', []))
            for story in st.session_state.generated_stories
        )
        
        st.metric("Total Words", f"{total_words:,}")
        st.metric("Vocabulary Items", total_vocab)
        st.metric("Stories", len(st.session_state.generated_stories))
        
        if hasattr(st.session_state, 'generation_time'):
            st.caption(f"Generated: {st.session_state.generation_time.strftime('%Y-%m-%d %H:%M')}")

# Footer
st.divider()
st.caption("""
**Enhanced Graded Reader Builder** - Create comprehensive language learning materials with:
- 🌍 Support for 12+ languages
- 📊 8 proficiency levels (A1-C1)
- 📝 Pedagogically-sound content structure
- 🎯 Level-appropriate vocabulary and grammar
- 🎧 Audio generation for listening practice
- 📚 Complete learning package with exercises
""")

# Add import for pandas if using timestamps
import pandas as pd
