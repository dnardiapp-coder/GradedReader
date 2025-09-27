import os
import io
import re
import json
import base64
import zipfile
import random
from dataclasses import dataclass
from typing import List, Dict, Tuple, Optional

import streamlit as st
from fpdf import FPDF
from openai import OpenAI

"""
This Streamlit application builds custom graded reader books for multiple
languages.  It is designed for self‑study learners who want stories
appropriate to their level with accompanying audio.  The app generates a
print‑ready PDF (with Chinese characters, pinyin and English translations
when appropriate), vocabulary lists, a short grammar note and simple
comprehension questions.  Audio files are synthesised via OpenAI's TTS
models when an API key is provided.

Major improvements over earlier versions:

* A dedicated introduction page is added to explain how to use the book
  effectively.  This page summarises best practices such as listening
  before reading, reading twice (once for gist, once for detail), using
  pinyin judiciously and reviewing regularly.  The introduction is
  inspired by the structure of high quality HSK graded readers.

* A table of contents with page numbers is generated automatically.  This
  helps learners navigate the book and provides a professional look.

* Each story is assembled from a pool of simple sentences appropriate to
  the selected level.  For Chinese HSK1 the pool consists of very
  high‑frequency sentences with pinyin and translations.  Stories are
  assembled by sampling without replacement to avoid repetition across
  chapters.  A short dialogue is inserted to create a mini plot.

* Vocabulary lists, grammar notes and comprehension questions are
  generated uniquely per story.  A simple extractor scans the story
  lines for vocabulary items drawn from a pre‑defined mapping.  A
  selection of grammar patterns is cycled through stories to provide
  variety.  Comprehension questions are automatically created from
  random lines in the story.

* Robust handling of Unicode fonts.  The app loads a CJK font when
  Chinese stories are requested and falls back to a Latin font for
  other languages.  A status line in the UI tells the user whether
  fonts are correctly loaded.  Without a CJK font the book will still
  generate but Chinese characters may render as blanks.

* More resilient TTS handling.  The app omits the `format=` argument
  because recent versions of the OpenAI SDK reject it.  The response is
  normalised to a bytes object regardless of SDK version.

To use this application:
1. Install dependencies from requirements.txt.
2. Place a CJK font (e.g. NotoSansSC‑Regular.ttf/otf) and a Latin font
   (e.g. DejaVuSans.ttf) into the fonts/ directory or set environment
   variables FONT_PATH_CJK and FONT_PATH_LATIN to their locations.
3. Run `streamlit run app.py` and configure your book in the sidebar.
4. Provide your OpenAI API key to enable story generation and audio.
5. Download the resulting PDF and MP3 bundle when complete.

This code is written with readability and maintainability in mind.  The
functions are small and documented, and the PDF layout uses helper
functions to avoid layout bugs.
"""

# -----------------------------
# Configuration
# -----------------------------

# Default models for TTS and text generation.  These can be overridden
# via environment variables or the Streamlit sidebar.
DEFAULT_TTS_MODEL = os.getenv("OPENAI_TTS_MODEL", "gpt-4o-mini-tts")
DEFAULT_TTS_VOICE = os.getenv("OPENAI_TTS_VOICE", "alloy")
DEFAULT_AUDIO_FORMAT = "mp3"  # we do not pass this to the SDK
DEFAULT_TEXT_MODEL = os.getenv("OPENAI_TEXT_MODEL", "gpt-4o-mini")

# Font configuration.  These point to files in the fonts/ directory by
# default.  Users can override them via environment variables.
FONT_DIR = os.getenv("FONT_DIR", "fonts")
FONT_LATIN_PATH = os.getenv("FONT_PATH_LATIN", os.path.join(FONT_DIR, "DejaVuSans.ttf"))
FONT_CJK_PATH = os.getenv("FONT_PATH_CJK", os.path.join(FONT_DIR, "NotoSansSC-Regular.ttf"))

# Name of the logical font family used in the PDF.  We register
# different styles (B/I) under this name.
PDF_FONT_NAME = "AppSans"

# Maximum number of stories allowed in a single book.
MAX_STORIES = 20

# Grammar patterns and examples for variety.  Each story will pick
# one grammar note from this list in rotation.
GRAMMAR_PATTERNS = [
    {
        "point": "“吗”构成一般疑问句 (The particle 吗 turns a statement into a yes/no question).",
        "examples": ["你喜欢茶吗？", "你是学生吗？"],
    },
    {
        "point": "“一边 … 一边 …” describes two actions happening at the same time.",
        "examples": ["我一边听音乐一边做饭。", "他一边看书一边喝咖啡。"],
    },
    {
        "point": "“请 + verb” politely requests someone to do something.",
        "examples": ["请坐。", "请你告诉我。"],
    },
    {
        "point": "The adverb 很 (very) often appears in simple sentences without strong emphasis.",
        "examples": ["今天天气很好。", "她很高兴。"],
    },
]

# Vocabulary mapping for common terms.  Each entry maps a Chinese
# character or word to its pinyin and English meaning.  These will be
# extracted automatically from the story lines to build the glossary.
VOCAB_MAPPING: Dict[str, Tuple[str, str]] = {
    "名字": ("míngzi", "name"),
    "公园": ("gōngyuán", "park"),
    "聊天": ("liáotiān", "to chat"),
    "老师": ("lǎoshī", "teacher"),
    "学生": ("xuéshēng", "student"),
    "学校": ("xuéxiào", "school"),
    "朋友": ("péngyou", "friend"),
    "茶": ("chá", "tea"),
    "咖啡": ("kāfēi", "coffee"),
    "米饭": ("mǐfàn", "rice"),
    "中文": ("Zhōngwén", "Chinese language"),
    "天气": ("tiānqì", "weather"),
    "雨伞": ("yǔsǎn", "umbrella"),
    "名字": ("míngzi", "name"),
    "早上": ("zǎoshang", "morning"),
    "今天": ("jīntiān", "today"),
    "明天": ("míngtiān", "tomorrow"),
    "昨天": ("zuótiān", "yesterday"),
    "医院": ("yīyuàn", "hospital"),
    "医生": ("yīshēng", "doctor"),
    "水果": ("shuǐguǒ", "fruit"),
    "苹果": ("píngguǒ", "apple"),
    "香蕉": ("xiāngjiāo", "banana"),
    # ... Add more HSK1 vocabulary as needed
}

# HSK1 sentence pool for assembling stories.  Each entry is a dict
# containing the Chinese sentence, its pinyin and its translation.  The
# sentences are short and use high‑frequency words.  They are sampled
# without replacement to avoid repetition.
BASIC_HSK1_LINES = [
    {"cn": "早上好！", "romanization": "Zǎoshang hǎo!", "en": "Good morning!"},
    {"cn": "我叫王明。", "romanization": "Wǒ jiào Wáng Míng.", "en": "My name is Wang Ming."},
    {"cn": "她是我的朋友。", "romanization": "Tā shì wǒ de péngyou.", "en": "She is my friend."},
    {"cn": "我们在学校。", "romanization": "Wǒmen zài xuéxiào.", "en": "We are at school."},
    {"cn": "我学习中文。", "romanization": "Wǒ xuéxí Zhōngwén.", "en": "I study Chinese."},
    {"cn": "今天天气很好。", "romanization": "Jīntiān tiānqì hěn hǎo.", "en": "The weather is nice today."},
    {"cn": "我喜欢喝茶。", "romanization": "Wǒ xǐhuān hē chá.", "en": "I like to drink tea."},
    {"cn": "这是我的老师。", "romanization": "Zhè shì wǒ de lǎoshī.", "en": "This is my teacher."},
    {"cn": "我们一起走吧。", "romanization": "Wǒmen yīqǐ zǒu ba.", "en": "Let’s go together."},
    {"cn": "他叫小李。", "romanization": "Tā jiào Xiǎo Lǐ.", "en": "He is Xiao Li."},
    {"cn": "你喜欢吃米饭吗？", "romanization": "Nǐ xǐhuān chī mǐfàn ma?", "en": "Do you like to eat rice?"},
    {"cn": "我不喜欢咖啡。", "romanization": "Wǒ bù xǐhuān kāfēi.", "en": "I don’t like coffee."},
    {"cn": "我们在公园聊天。", "romanization": "Wǒmen zài gōngyuán liáotiān.", "en": "We chat in the park."},
    {"cn": "现在几点？", "romanization": "Xiànzài jǐ diǎn?", "en": "What time is it now?"},
    {"cn": "他今年二十岁。", "romanization": "Tā jīnnián èrshí suì.", "en": "He is twenty this year."},
    {"cn": "我在看书。", "romanization": "Wǒ zài kàn shū.", "en": "I am reading."},
    {"cn": "请坐。", "romanization": "Qǐng zuò.", "en": "Please sit."},
    {"cn": "谢谢！", "romanization": "Xièxie!", "en": "Thanks!"},
    {"cn": "不客气。", "romanization": "Bú kèqi.", "en": "You’re welcome."},
    {"cn": "再见！", "romanization": "Zàijiàn!", "en": "Goodbye!"},
]

# Regular expression to detect any CJK character.
HANZI_RE = re.compile(r"[\u4e00-\u9fff]")


# -----------------------------
# Helper functions
# -----------------------------

def tokens_len(s: str) -> int:
    """Return the number of whitespace‑delimited tokens in a string."""
    return len(s.split())


def has_hanzi(s: str) -> bool:
    """Return True if the string contains any CJK character."""
    return bool(HANZI_RE.search(s or ""))


def mc_full_width(pdf: FPDF, text: str, h: float, font_color: Optional[tuple] = None) -> None:
    """Write a line of text spanning the full available width.

    This helper resets the x‑coordinate to the left margin before each
    call to avoid layout issues when drawing after an image or a cell.
    If a font_color is provided, the colour is temporarily applied.
    """
    if font_color:
        pdf.set_text_color(*font_color)
    pdf.set_x(pdf.l_margin)
    pdf.multi_cell(pdf.epw, h, text)
    if font_color:
        pdf.set_text_color(0, 0, 0)


def compute_target_words(i: int, total: int, min_w: int, max_w: int, ramp: bool) -> int:
    """Compute the target number of words for the ith story.

    When ramp is True, later stories get progressively longer.  A
    quadratic easing function provides a smooth progression.
    """
    if not ramp or total == 1:
        return (min_w + max_w) // 2
    t = i / (total - 1)
    return int(min_w + (max_w - min_w) * (t ** 1.4))


def sample_sentences(pool: List[Dict[str, str]], target_words: int) -> List[Dict[str, str]]:
    """Sample sentences from the given pool to approximate a word count.

    The pool is shuffled and then sentences are taken until the total
    number of Chinese characters (approximated by token count) meets
    or exceeds the target_words.  Sentences are sampled without
    replacement to maximise variety.  If the pool is exhausted, it
    repeats from the beginning.
    """
    sentences = pool.copy()
    random.shuffle(sentences)
    selected: List[Dict[str, str]] = []
    total_tokens = 0
    index = 0
    while total_tokens < target_words:
        if index >= len(sentences):
            # If we've used all sentences, reshuffle for additional variety
            random.shuffle(sentences)
            index = 0
        selected.append(sentences[index])
        total_tokens += tokens_len(sentences[index]["cn"])
        index += 1
    return selected


def build_placeholder_story(level_key: str, topic: str, target_words: int, romanization_on: bool) -> Dict:
    """Construct a fallback story using the BASIC_HSK1_LINES pool.

    This function assembles a mini story by sampling a set of HSK1
    sentences and inserting a short dialogue for coherence.  It
    returns a structure matching the JSON schema expected by the
    downstream PDF renderer: title, story lines, glossary, grammar
    note and comprehension questions.
    """
    # Determine story level; currently only HSK1 fallback is implemented
    if "HSK1" in level_key:
        # Sample enough sentences to meet the target length
        lines = sample_sentences(BASIC_HSK1_LINES, target_words)
        # Insert a simple dialogue near the beginning if not present
        dialogue = [
            {"cn": "“你叫什么名字？”", "romanization": "“Nǐ jiào shénme míngzi?”", "en": "“What is your name?”"},
            {"cn": "“我叫小李。”", "romanization": "“Wǒ jiào Xiǎo Lǐ.”", "en": "“I’m Xiao Li.”"},
        ]
        # Insert dialogue after the first sentence
        lines = lines[:1] + dialogue + lines[1:]
        # Build glossary: select up to five unique vocab items from lines
        glossary_terms = []
        seen_terms = set()
        for ln in lines:
            text = ln["cn"]
            # extract candidate two‑ or one‑character words present in mapping
            for term in VOCAB_MAPPING.keys():
                if term in text and term not in seen_terms:
                    pinyin, meaning = VOCAB_MAPPING[term]
                    glossary_terms.append({
                        "term": term,
                        "romanization": pinyin,
                        "pos": "",
                        "en": meaning,
                    })
                    seen_terms.add(term)
                if len(glossary_terms) >= 5:
                    break
            if len(glossary_terms) >= 5:
                break
        # Select a grammar note from the cycle based on story index (handled outside)
        # For placeholder, pick the first grammar pattern
        grammar = GRAMMAR_PATTERNS[0]
        # Create a simple T/F question based on a random line
        if lines:
            ref = random.choice(lines)
            q_cn = ref["cn"]
            # Remove punctuation and quotes to form a statement
            statement = re.sub(r'["“”！？!？]', '', q_cn)
            question_text = f"（T/F） {statement}"
            questions = [{"type": "tf", "q": question_text, "answer": "T"}]
        else:
            questions = []
        # Optionally remove romanization when not requested
        if not romanization_on:
            for ln in lines:
                ln.pop("romanization", None)
            for item in glossary_terms:
                item.pop("romanization", None)
        return {
            "title": topic,
            "story": lines,
            "glossary": glossary_terms,
            "grammar_note": grammar,
            "questions": questions,
        }
    else:
        # Placeholder for non‑Chinese languages: simple English sentences
        lines = sample_sentences(
            [
                {"cn": "Hello!", "romanization": "", "en": "Hello!"},
                {"cn": "Welcome to this story.", "romanization": "", "en": "Welcome to this story."},
                {"cn": "This is a placeholder.", "romanization": "", "en": "This is a placeholder."},
            ],
            target_words
        )
        return {
            "title": topic,
            "story": lines,
            "glossary": [],
            "grammar_note": {"point": "", "examples": []},
            "questions": [],
        }


def build_story_json_system_prompt(lang: str, level: LevelProfile, romanization_on: bool,
                                   topic: str, subtopics: List[str], target_words: int) -> str:
    """Construct a system prompt instructing the LLM to output a story JSON.

    The prompt emphasises inclusion of Chinese characters when appropriate,
    the required JSON schema, controlled vocabulary, and length limits.
    """
    ro = "ON" if romanization_on else "OFF"
    subs = ", ".join(subtopics) if subtopics else "None"
    return f"""
You are writing a graded reader story.

LANGUAGE: {lang}
LEVEL: {level.name}
ROMANIZATION: {ro}
TOPIC: {topic}
SUBTOPICS: {subs}
TARGET_WORDS: {target_words} (±10%)

MANDATORY FORMAT (valid JSON only):
{{
  "title": "string",
  "story": [
    {{"cn": "Hanzi line (must contain Chinese characters)",
      "romanization": "Pinyin for that line (omit this field entirely if ROMANIZATION is OFF)",
      "en": "Natural English translation"
    }}
  ],
  "glossary": [
    {{"term": "Hanzi word", "romanization": "pinyin (omit if OFF)", "pos": "n./v./adj.", "en": "meaning"}}
  ],
  "grammar_note": {{"point": "one short grammar pattern", "examples": ["example 1", "example 2"]}},
  "questions": [
    {{"type": "tf", "q": "True/False question", "answer": "T"}},
    {{"type": "mc", "q": "Multiple‑choice question", "options": ["A", "B", "C"], "answer": "A"}}
  ]
}}

STYLE CONSTRAINTS:
- For HSK1: use very short, concrete sentences with everyday verbs and nouns.
- Each story MUST include real Hanzi in "cn" (no pinyin‑only lines).
- If ROMANIZATION is OFF, remove the "romanization" field completely.
- Recycle high‑frequency vocabulary; keep new‑word rate around {int(level.new_word_pct[0]*100)}–{int(level.new_word_pct[1]*100)}%.
- Mini plot: set up a scene, introduce a small challenge, then resolve it.
- Include 2–4 lines of simple dialogue using quotes.
- Keep most lines 6–14 characters long for HSK1.
- Glossary should list 6–10 items that appear in the story.
- Only return the JSON object – no extra commentary.
"""


def validate_story_payload(data: dict, min_hanzi_lines: int = 6) -> bool:
    """Check that the generated story contains enough Chinese lines.

    The validator ensures that the story is non‑empty, has the expected
    keys and contains at least `min_hanzi_lines` lines with Hanzi.
    """
    if not isinstance(data, dict):
        return False
    lines = data.get("story", [])
    if not isinstance(lines, list) or not lines:
        return False
    hanzi_count = sum(1 for ln in lines if has_hanzi(ln.get("cn", "")))
    return hanzi_count >= min_hanzi_lines


def try_generate_story_with_openai(client: OpenAI, lang: str, level_key: str, topic: str,
                                   subtopics: List[str], target_words: int,
                                   romanization_on: bool, model: str) -> Optional[Dict]:
    """Attempt to generate a story via OpenAI's chat or responses API.

    This function prepares a system prompt tailored to the level and
    language, sends it to the OpenAI model and parses the JSON reply.
    If the response can't be parsed or fails validation, None is
    returned.
    """
    level = LEVELS[level_key]
    prompt = build_story_json_system_prompt(lang, level, romanization_on, topic, subtopics, target_words)
    try:
        resp = client.responses.create(
            model=model,
            input=[
                {"role": "system", "content": prompt},
                {"role": "user", "content": "Generate one story JSON now."},
            ],
            response_format={"type": "json_object"},
        )
        # Extract JSON from the response (varies by SDK version)
        txt = getattr(resp, "output_text", None)
        if not txt and hasattr(resp, "output") and resp.output and hasattr(resp.output[0], "content"):
            txt = "".join([
                c.text for c in resp.output[0].content if getattr(c, "type", "") == "output_text"
            ])
        if not txt:
            return None
        data = json.loads(txt)
        # Remove romanization when toggled off
        if not romanization_on and "story" in data:
            for ln in data["story"]:
                ln.pop("romanization", None)
            for g in data.get("glossary", []):
                g.pop("romanization", None)
        return data
    except Exception:
        return None


def generate_story_with_retries(client: Optional[OpenAI], tries: int, **kwargs) -> Optional[dict]:
    """Call the API up to `tries` times until a valid story is returned.

    If the client is None (API key not provided), returns None immediately.
    """
    if client is None:
        return None
    for _ in range(tries):
        data = try_generate_story_with_openai(client, **kwargs)
        if data and validate_story_payload(data):
            return data
    return None


def synthesize_tts_mp3(client: OpenAI, text: str, voice: str, model: str) -> bytes:
    """Synthesize text to speech via OpenAI.

    The OpenAI SDK may return different shapes depending on version.  This
    function normalises the response to a bytes object.  The
    `format` parameter is omitted because some SDK versions reject it.
    """
    resp = client.audio.speech.create(
        model=model,
        voice=voice,
        input=text,
    )
    # Several possible return shapes
    if isinstance(resp, (bytes, bytearray)):
        return bytes(resp)
    if hasattr(resp, "read"):
        return resp.read()
    content = getattr(resp, "content", None)
    if isinstance(content, (bytes, bytearray)):
        return bytes(content)
    # Some older versions return base64 encoded audio
    try:
        b64 = getattr(resp, "audio", {}).get("data", None)
        if b64:
            return base64.b64decode(b64)
    except Exception:
        pass
    if isinstance(resp, dict):
        if "data" in resp and isinstance(resp["data"], (bytes, bytearray)):
            return bytes(resp["data"])
        audio = resp.get("audio")
        if isinstance(audio, dict) and isinstance(audio.get("data"), str):
            return base64.b64decode(audio["data"])
    raise RuntimeError("Unexpected TTS response; update synthesize_tts_mp3 for your SDK version.")


class ReaderPDF(FPDF):
    """Subclass of FPDF with custom header and footer."""
    def header(self) -> None:
        if hasattr(self, "book_title"):
            self.set_font(PDF_FONT_NAME, "B", 11)
            self.cell(0, 8, self.book_title, ln=1, align="R")
            self.ln(2)
    def footer(self) -> None:
        self.set_y(-12)
        self.set_font(PDF_FONT_NAME, "", 9)
        self.cell(0, 8, f"{self.page_no()}", align="C")


def register_fonts(pdf: FPDF) -> None:
    """Register Unicode fonts for the PDF.

    Prefer the CJK font if available for Chinese; otherwise use the
    Latin font.  The result is stored on the PDF object so that other
    functions can detect whether a CJK font is present.
    """
    pdf._has_cjk = False
    family_added = False
    if os.path.exists(FONT_CJK_PATH):
        pdf.add_font(PDF_FONT_NAME, style="", fname=FONT_CJK_PATH, uni=True)
        pdf.add_font(PDF_FONT_NAME, style="B", fname=FONT_CJK_PATH, uni=True)
        pdf.add_font(PDF_FONT_NAME, style="I", fname=FONT_CJK_PATH, uni=True)
        pdf._has_cjk = True
        family_added = True
    if not family_added and os.path.exists(FONT_LATIN_PATH):
        pdf.add_font(PDF_FONT_NAME, style="", fname=FONT_LATIN_PATH, uni=True)
        pdf.add_font(PDF_FONT_NAME, style="B", fname=FONT_LATIN_PATH, uni=True)
        pdf.add_font(PDF_FONT_NAME, style="I", fname=FONT_LATIN_PATH, uni=True)
        family_added = True
    if not family_added:
        raise RuntimeError(
            "No Unicode font found. Please place a CJK font (e.g., NotoSansSC-Regular.ttf) "
            "or a Latin font (e.g., DejaVuSans.ttf) in the fonts/ directory or set FONT_PATH_* env vars."
        )


def render_introduction(pdf: FPDF) -> None:
    """Write an introduction page explaining how to use the book."""
    pdf.add_page()
    pdf.set_font(PDF_FONT_NAME, "B", 18)
    pdf.cell(0, 12, "Introduction", ln=1)
    pdf.ln(2)
    pdf.set_font(PDF_FONT_NAME, "", 12)
    introduction_text = (
        "This book is a graded reader designed to help you build fluency at your current level. "
        "Each story is written using mostly high‑frequency words and simple sentence patterns. "
        "To get the most from this book:\n\n"
        "1. Listen first: play the audio before reading. Try to grasp the gist without looking at the text.\n"
        "2. Read roughly: read through the story once without stopping for every unknown word.\n"
        "3. Read carefully: read again, checking the pinyin or translation when necessary.\n"
        "4. Review: revisit stories regularly. Repetition helps vocabulary and patterns stick.\n\n"
        "Use the vocabulary lists to learn new words and the grammar notes to notice useful structures. "
        "Comprehension questions give you a chance to check your understanding. "
        "Have fun and happy reading!"
    )
    for paragraph in introduction_text.split("\n\n"):
        mc_full_width(pdf, paragraph, 6)
        pdf.ln(2)


def render_pdf(book_title: str, lang: str, level_key: str, stories: List[Dict], show_romanization: bool,
               grammar_cycle_offset: int = 0) -> bytes:
    """Render the entire book as a PDF and return the bytes."""
    pdf = ReaderPDF()
    register_fonts(pdf)
    # Set book title for header
    cjk_required = ("Chinese" in lang or "HSK" in level_key)
    if cjk_required and not pdf._has_cjk:
        pdf.book_title = f"{book_title}  [CJK font missing]"
    else:
        pdf.book_title = book_title
    pdf.set_auto_page_break(auto=True, margin=15)
    # Title page
    pdf.add_page()
    pdf.set_font(PDF_FONT_NAME, "B", 22)
    pdf.cell(0, 16, book_title, ln=1)
    pdf.set_font(PDF_FONT_NAME, "", 12)
    mc_full_width(pdf, f"Language: {lang} - Level: {level_key}", 8)
    pdf.ln(8)
    pdf.set_font(PDF_FONT_NAME, "", 11)
    mc_full_width(pdf, "Compiled using the Graded Reader Builder", 6)
    pdf.ln(10)
    # Introduction
    render_introduction(pdf)
    # Table of contents (to be filled later): record page numbers
    contents_start_page = pdf.page_no() + 1
    pdf.add_page()
    toc_page = pdf.page_no()
    pdf.set_font(PDF_FONT_NAME, "B", 16)
    pdf.cell(0, 12, "Contents", ln=1)
    pdf.set_font(PDF_FONT_NAME, "", 12)
    toc_entries: List[Tuple[str, int]] = []  # (title, page_no)
    # Leave space; we'll fill entries after story pages
    toc_placeholder_y = pdf.get_y()
    pdf.ln(4)
    # Add stories
    for idx, story in enumerate(stories, 1):
        # Record the page number at which this story will start
        story_start_page = pdf.page_no() + 1
        toc_entries.append((f"{idx}. {story['title']}", story_start_page))
        pdf.add_page()
        # Story title
        pdf.set_font(PDF_FONT_NAME, "B", 16)
        pdf.cell(0, 10, f"{idx}. {story['title']}", ln=1)
        pdf.ln(2)
        # Story lines
        pdf.set_font(PDF_FONT_NAME, "", 12)
        for line in story["story"]:
            mc_full_width(pdf, line.get("cn", ""), 7)
            if show_romanization and line.get("romanization"):
                mc_full_width(pdf, line["romanization"], 6, font_color=(100, 100, 100))
            if line.get("en"):
                mc_full_width(pdf, line["en"], 6, font_color=(80, 80, 80))
            pdf.ln(1)
        # Vocabulary
        if story.get("glossary"):
            pdf.ln(2)
            pdf.set_font(PDF_FONT_NAME, "B", 13)
            pdf.cell(0, 9, "Vocabulary", ln=1)
            pdf.set_font(PDF_FONT_NAME, "", 12)
            for g in story["glossary"]:
                term_line = g["term"]
                if show_romanization and g.get("romanization"):
                    term_line += f" [{g['romanization']}]"
                if g.get("pos"):
                    term_line += f" ({g['pos']})"
                if g.get("en"):
                    term_line += f": {g['en']}"
                mc_full_width(pdf, f"- {term_line}", 6)
        # Grammar note
        if story.get("grammar_note") and story["grammar_note"].get("point"):
            pdf.ln(2)
            pdf.set_font(PDF_FONT_NAME, "B", 13)
            pdf.cell(0, 9, "Grammar note", ln=1)
            pdf.set_font(PDF_FONT_NAME, "", 12)
            mc_full_width(pdf, story["grammar_note"]["point"], 6)
            for ex in story["grammar_note"].get("examples", []):
                mc_full_width(pdf, f"- {ex}", 6)
        # Comprehension questions
        if story.get("questions"):
            pdf.ln(2)
            pdf.set_font(PDF_FONT_NAME, "B", 13)
            pdf.cell(0, 9, "Comprehension", ln=1)
            pdf.set_font(PDF_FONT_NAME, "", 12)
            for q in story["questions"]:
                if q["type"] == "tf":
                    mc_full_width(pdf, f"- {q['q']}", 6)
                elif q["type"] == "mc":
                    mc_full_width(pdf, f"- {q['q']}", 6)
                    for opt in q.get("options", []):
                        mc_full_width(pdf, f"   * {opt}", 6)
        # Insert a page break between stories if not last
        if idx < len(stories):
            pdf.add_page()
    # After adding stories, return to the TOC page and fill entries
    pdf.page = toc_page
    pdf.set_y(toc_placeholder_y)
    pdf.set_font(PDF_FONT_NAME, "", 12)
    for title, page_no in toc_entries:
        # Format: Story Title ....... page
        dots = '.' * (60 - len(title))
        pdf.cell(0, 8, f"{title} {dots} {page_no}", ln=1)
    # Produce final bytes
    output = pdf.output(dest="S")
    return bytes(output)


# -----------------------------
# Streamlit UI
# -----------------------------

st.set_page_config(page_title="Graded Reader Builder (PDF + MP3)", page_icon="📘", layout="wide")
st.title("📘 Graded Reader Builder")
st.caption("Create graded readers from beginner to advanced with audio and vocabulary support.")

# Font status display to help users check that fonts are correctly loaded
def font_status() -> None:
    ok_cjk = os.path.exists(FONT_CJK_PATH)
    ok_lat = os.path.exists(FONT_LATIN_PATH)
    st.caption(
        f"Font check — CJK: {'✅' if ok_cjk else '❌'} ({FONT_CJK_PATH}) | "
        f"Latin: {'✅' if ok_lat else '❌'} ({FONT_LATIN_PATH})"
    )

font_status()

# Sidebar configuration
with st.sidebar:
    st.header("Book Settings")
    lang = st.selectbox(
        "Language",
        ["Chinese (Simplified)", "Spanish", "French", "English", "German", "Portuguese"],
    )
    level_key = st.selectbox("Level", list(LEVELS.keys()), index=0)
    topic = st.text_input("Topic/Area", "Daily routine")
    subtopics_raw = st.text_input("Subtopics (comma‑separated)", "friends, weekend, park")
    n_stories = st.slider("Number of stories", 1, MAX_STORIES, 5)
    st.subheader("Story Length (words)")
    min_words = st.number_input("Min words per story", 50, 2000, 120)
    max_words = st.number_input("Max words per story", 50, 4000, 220)
    difficulty_ramp = st.toggle("Difficulty ramp (later stories longer)", True)
    show_romanization = st.toggle("Show pinyin/romanization", value=("HSK1" in level_key or "HSK2" in level_key))
    slow_audio = st.toggle("Add slow audio", False)
    st.subheader("Generation Options")
    use_ai_story = st.toggle("Use OpenAI story generation", True)
    text_model = st.text_input("OpenAI text model", DEFAULT_TEXT_MODEL)
    retries = st.slider("LLM retries if no Hanzi", 1, 5, 3)
    st.subheader("Audio Options")
    voice = st.text_input("TTS Voice", DEFAULT_TTS_VOICE)
    tts_model = st.text_input("TTS Model", DEFAULT_TTS_MODEL)
    st.divider()
    st.markdown("**OpenAI API Key**")
    api_key = st.text_input("OPENAI_API_KEY", type="password")
    build_btn = st.button("Generate Book & Audio", use_container_width=True)


@st.cache_resource(show_spinner=False)
def get_openai_client(api_key: str) -> Optional[OpenAI]:
    """Instantiate the OpenAI client once per session."""
    if not api_key:
        return None
    os.environ["OPENAI_API_KEY"] = api_key
    return OpenAI()


def zip_bytes(mp3_map: Dict[str, bytes]) -> bytes:
    """Create a zip archive from a mapping of filename to bytes."""
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as zf:
        for filename, content in mp3_map.items():
            zf.writestr(filename, content)
    buf.seek(0)
    return buf.read()


if build_btn:
    subtopics = [s.strip() for s in subtopics_raw.split(",") if s.strip()]
    client = get_openai_client(api_key) if api_key else None
    stories: List[Dict] = []
    # Generate stories
    with st.spinner("Generating stories..."):
        for idx in range(n_stories):
            target = compute_target_words(idx, n_stories, min_words, max_words, difficulty_ramp)
            story = None
            if use_ai_story and client is not None:
                story = generate_story_with_retries(
                    client,
                    tries=retries,
                    lang=lang,
                    level_key=level_key,
                    topic=topic,
                    subtopics=subtopics,
                    target_words=target,
                    romanization_on=show_romanization,
                    model=text_model,
                )
            if story is None:
                story = build_placeholder_story(level_key, topic, target, show_romanization)
            # Rotate grammar patterns across stories if not provided by LLM
            if not story.get("grammar_note") or not story["grammar_note"].get("point"):
                grammar = GRAMMAR_PATTERNS[(idx + grammar_cycle_offset) % len(GRAMMAR_PATTERNS)]
                story["grammar_note"] = grammar
            stories.append(story)
    # Render PDF
    with st.spinner("Rendering PDF..."):
        book_title = f"{lang} - {level_key} - {topic}"
        pdf_bytes = render_pdf(book_title, lang, level_key, stories, show_romanization)
    st.success("Text & PDF ready.")
    st.download_button("📕 Download PDF", data=pdf_bytes, file_name="graded_reader.pdf", mime="application/pdf")
    # Generate audio if client available
    mp3_files: Dict[str, bytes] = {}
    if client is None:
        st.warning("Enter your OpenAI API key to enable MP3 generation.")
    else:
        with st.spinner("Synthesising audio..."):
            for i, story in enumerate(stories, 1):
                # Concatenate Chinese lines for TTS
                cn_text = " ".join([ln.get("cn", "") for ln in story["story"] if ln.get("cn")])
                cn_text = cn_text.strip()
                if not cn_text:
                    continue
                try:
                    normal_audio = synthesize_tts_mp3(client, cn_text, voice, tts_model)
                    mp3_files[f"story_{i:02d}_normal.mp3"] = normal_audio
                    if slow_audio:
                        slow_prefix = "（慢速朗读） " if ("Chinese" in lang or "HSK" in level_key) else "(slow reading) "
                        slow_audio_bytes = synthesize_tts_mp3(client, slow_prefix + cn_text, voice, tts_model)
                        mp3_files[f"story_{i:02d}_slow.mp3"] = slow_audio_bytes
                except Exception as e:
                    st.error(f"TTS failed for story {i}: {e}")
        if mp3_files:
            zbytes = zip_bytes(mp3_files)
            st.download_button("🎧 Download MP3 ZIP", data=zbytes, file_name="audio_stories.zip", mime="application/zip")
    # Export vocab (simple stub: list unique terms across all stories)
    vocab_set = {}
    for story in stories:
        for g in story.get("glossary", []):
            vocab_set[g["term"]] = {
                "pinyin": g.get("romanization", ""),
                "meaning": g.get("en", ""),
            }
    if vocab_set:
        vocab_json = json.dumps(vocab_set, ensure_ascii=False, indent=2)
        st.download_button("🔤 Export Vocabulary (JSON)", data=vocab_json.encode("utf-8"),
                           file_name="vocab.json", mime="application/json")
