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
Graded Reader Builder (PDF + MP3)
- Intro page with usage tips
- Auto Table of Contents with page numbers
- HSK1-focused placeholder generator (varied, no obvious repetition)
- Optional OpenAI JSON story generation (strong prompt + validator)
- Unicode font handling (CJK + Latin)
- OpenAI TTS with robust response normalization (no format= kwarg)
"""

# =========================
# ====== CONFIG ===========
# =========================

DEFAULT_TTS_MODEL = os.getenv("OPENAI_TTS_MODEL", "gpt-4o-mini-tts")  # or "tts-1"
DEFAULT_TTS_VOICE = os.getenv("OPENAI_TTS_VOICE", "alloy")
DEFAULT_AUDIO_FORMAT = "mp3"  # not passed to SDK

DEFAULT_TEXT_MODEL = os.getenv("OPENAI_TEXT_MODEL", "gpt-4o-mini")

FONT_DIR = os.getenv("FONT_DIR", "fonts")
FONT_LATIN_PATH = os.getenv("FONT_PATH_LATIN", os.path.join(FONT_DIR, "DejaVuSans.ttf"))
# try .ttf then .otf for Simplified Chinese
FONT_CJK_PATH = os.getenv("FONT_PATH_CJK", os.path.join(FONT_DIR, "NotoSansSC-Regular.ttf"))
if not os.path.exists(FONT_CJK_PATH):
    FONT_CJK_PATH = os.getenv("FONT_PATH_CJK", os.path.join(FONT_DIR, "NotoSansSC-Regular.otf"))

PDF_FONT_NAME = "AppSans"
MAX_STORIES = 20

# Grammar patterns rotated across stories when placeholders are used
GRAMMAR_PATTERNS = [
    {
        "point": "“吗”构成一般疑问句（Yes/No 问句）。",
        "examples": ["你喜欢茶吗？", "你是学生吗？"],
    },
    {
        "point": "“一边 … 一边 …”表示两个动作同时进行。",
        "examples": ["我一边听音乐一边做饭。", "他一边看书一边喝咖啡。"],
    },
    {
        "point": "“请 + 动词”用于礼貌请求。",
        "examples": ["请坐。", "请你告诉我。"],
    },
    {
        "point": "副词“很”常见于初级句子中，不一定表示“非常”。",
        "examples": ["今天天气很好。", "她很高兴。"],
    },
]

# Basic HSK1 vocabulary mapping (expand as needed)
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
    "早上": ("zǎoshang", "morning"),
    "今天": ("jīntiān", "today"),
    "明天": ("míngtiān", "tomorrow"),
    "昨天": ("zuótiān", "yesterday"),
    "水果": ("shuǐguǒ", "fruit"),
    "苹果": ("píngguǒ", "apple"),
    "香蕉": ("xiāngjiāo", "banana"),
}

# HSK1 sentence pool (expanded; add more over time)
BASIC_HSK1_LINES = [
    {"cn":"早上好！","romanization":"Zǎoshang hǎo!","en":"Good morning!"},
    {"cn":"我叫王明。","romanization":"Wǒ jiào Wáng Míng.","en":"My name is Wang Ming."},
    {"cn":"她是我的朋友。","romanization":"Tā shì wǒ de péngyou.","en":"She is my friend."},
    {"cn":"我们在学校。","romanization":"Wǒmen zài xuéxiào.","en":"We are at school."},
    {"cn":"我学习中文。","romanization":"Wǒ xuéxí Zhōngwén.","en":"I study Chinese."},
    {"cn":"今天天气很好。","romanization":"Jīntiān tiānqì hěn hǎo.","en":"The weather is nice today."},
    {"cn":"我喜欢喝茶。","romanization":"Wǒ xǐhuān hē chá.","en":"I like to drink tea."},
    {"cn":"这是我的老师。","romanization":"Zhè shì wǒ de lǎoshī.","en":"This is my teacher."},
    {"cn":"我们一起走吧。","romanization":"Wǒmen yīqǐ zǒu ba.","en":"Let’s go together."},
    {"cn":"他叫小李。","romanization":"Tā jiào Xiǎo Lǐ.","en":"He is Xiao Li."},
    {"cn":"你喜欢吃米饭吗？","romanization":"Nǐ xǐhuān chī mǐfàn ma?","en":"Do you like to eat rice?"},
    {"cn":"我不喜欢咖啡。","romanization":"Wǒ bù xǐhuān kāfēi.","en":"I don’t like coffee."},
    {"cn":"我们在公园聊天。","romanization":"Wǒmen zài gōngyuán liáotiān.","en":"We chat in the park."},
    {"cn":"现在几点？","romanization":"Xiànzài jǐ diǎn?","en":"What time is it now?"},
    {"cn":"他今年二十岁。","romanization":"Tā jīnnián èrshí suì.","en":"He is twenty this year."},
    {"cn":"我在看书。","romanization":"Wǒ zài kàn shū.","en":"I am reading."},
    {"cn":"请坐。","romanization":"Qǐng zuò.","en":"Please sit."},
    {"cn":"谢谢！","romanization":"Xièxie!","en":"Thanks!"},
    {"cn":"不客气。","romanization":"Bú kèqi.","en":"You’re welcome."},
    {"cn":"再见！","romanization":"Zàijiàn!","en":"Goodbye!"},
    {"cn":"我们明天见。","romanization":"Wǒmen míngtiān jiàn.","en":"See you tomorrow."},
    {"cn":"他在喝水。","romanization":"Tā zài hē shuǐ.","en":"He is drinking water."},
    {"cn":"我有一个苹果。","romanization":"Wǒ yǒu yí gè píngguǒ.","en":"I have an apple."},
    {"cn":"她很忙。","romanization":"Tā hěn máng.","en":"She is busy."},
    {"cn":"今天下雨，我们带雨伞。","romanization":"Jīntiān xià yǔ, wǒmen dài yǔsǎn.","en":"It’s raining today; we bring umbrellas."},
]

HANZI_RE = re.compile(r"[\u4e00-\u9fff]")

# =========================
# ====== LEVELS ===========
# =========================

@dataclass
class LevelProfile:
    name: str
    sentence_len: Tuple[int, int]
    new_word_pct: Tuple[float, float]
    romanization: str
    script: str

LEVELS: Dict[str, LevelProfile] = {
    "HSK1 (A1)":  LevelProfile("HSK1 (A1)",  (6, 14),  (0.02, 0.05), "pinyin", "hanzi"),
    "HSK2 (A1+)": LevelProfile("HSK2 (A1+)", (8, 18),  (0.03, 0.06), "pinyin", "hanzi"),
    "HSK3 (A2)":  LevelProfile("HSK3 (A2)",  (10, 22), (0.03, 0.06), "pinyin", "hanzi"),
    "HSK4 (B1)":  LevelProfile("HSK4 (B1)",  (12, 26), (0.03, 0.07), "none",   "hanzi"),
    "HSK5 (B2)":  LevelProfile("HSK5 (B2)",  (14, 30), (0.03, 0.08), "none",   "hanzi"),
    "HSK6 (C1)":  LevelProfile("HSK6 (C1)",  (16, 34), (0.03, 0.10), "none",   "hanzi"),
}

# =========================
# ===== UTILITIES =========
# =========================

def has_hanzi(s: str) -> bool:
    return bool(HANZI_RE.search(s or ""))

def char_count_cn(s: str) -> int:
    """Rough measure for Chinese length (count CJK chars if present; else words)."""
    if has_hanzi(s):
        return sum(1 for ch in s if HANZI_RE.match(ch))
    return len(s.split())

def mc_full_width(pdf: FPDF, text: str, h: float, font_color: Optional[tuple] = None):
    if font_color:
        pdf.set_text_color(*font_color)
    pdf.set_x(pdf.l_margin)
    pdf.multi_cell(pdf.epw, h, text)
    if font_color:
        pdf.set_text_color(0, 0, 0)

def compute_target_words(i: int, total: int, min_w: int, max_w: int, ramp: bool) -> int:
    if not ramp or total == 1:
        return (min_w + max_w) // 2
    t = i / (total - 1)
    return int(min_w + (max_w - min_w) * (t ** 1.4))

def sample_sentences(pool: List[Dict[str, str]], target: int) -> List[Dict[str, str]]:
    """Sample without replacement until ~target characters/words."""
    lines = pool.copy()
    random.shuffle(lines)
    out, total = [], 0
    i = 0
    while total < target:
        if i >= len(lines):
            random.shuffle(lines)
            i = 0
        out.append(lines[i])
        total += char_count_cn(lines[i]["cn"])
        i += 1
    return out

# =========================
# === PLACEHOLDER GEN =====
# =========================

def placeholder_story(level_key: str, topic: str, target_words: int, romanization_on: bool) -> Dict:
    if "HSK" in level_key:
        lines = sample_sentences(BASIC_HSK1_LINES, target_words)
        # tiny dialogue near the start for a plot shape
        dialog = [
            {"cn":"“你叫什么名字？”","romanization":"“Nǐ jiào shénme míngzi?”","en":"“What is your name?”"},
            {"cn":"“我叫小李。”","romanization":"“Wǒ jiào Xiǎo Lǐ.”","en":"“I’m Xiao Li.”"},
        ]
        lines = lines[:1] + dialog + lines[1:]

        # glossary: up to 7 unique mapped items found in story
        seen, glossary = set(), []
        for ln in lines:
            for term, (py, en) in VOCAB_MAPPING.items():
                if term in ln["cn"] and term not in seen:
                    glossary.append({"term": term, "romanization": py, "pos": "", "en": en})
                    seen.add(term)
                    if len(glossary) >= 7:
                        break
            if len(glossary) >= 7:
                break

        # default grammar + one T/F question from a line
        grammar = random.choice(GRAMMAR_PATTERNS)
        ref = random.choice(lines)
        statement = re.sub(r'[“”"！？!?，,。．.]', '', ref["cn"])
        questions = [{"type":"tf","q":f"（T/F）{statement}", "answer":"T"}]

        if not romanization_on:
            for ln in lines:
                ln.pop("romanization", None)
            for g in glossary:
                g.pop("romanization", None)

        return {
            "title": f"{topic}",
            "story": lines,
            "glossary": glossary,
            "grammar_note": grammar,
            "questions": questions,
        }

    # Non-Chinese fallback (simple)
    lines = sample_sentences(
        [
            {"cn":"Hello!","romanization":"","en":"Hello!"},
            {"cn":"Welcome to this story.","romanization":"","en":"Welcome to this story."},
            {"cn":"This is a placeholder.","romanization":"","en":"This is a placeholder."},
        ],
        target_words
    )
    return {
        "title": topic,
        "story": lines,
        "glossary": [],
        "grammar_note": {"point":"", "examples":[]},
        "questions": [],
    }

# =========================
# ==== AI GENERATION ======
# =========================

def build_story_json_system_prompt(lang: str, level: LevelProfile, romanization_on: bool,
                                   topic: str, subtopics: List[str], target_words: int) -> str:
    ro = "ON" if romanization_on else "OFF"
    subs = ", ".join(subtopics) if subtopics else "None"
    return f"""
You are writing an HSK-style graded reader story.

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
      "romanization": "Pinyin for that line (omit the field entirely if ROMANIZATION is OFF)",
      "en": "Natural English translation"
    }}
  ],
  "glossary": [
    {{"term": "Hanzi word", "romanization": "pinyin (omit if OFF)", "pos": "n./v./adj.", "en": "meaning"}}
  ],
  "grammar_note": {{"point": "one short HSK1 pattern", "examples": ["example 1", "example 2"]}},
  "questions": [
    {{"type": "tf", "q": "True/False question", "answer": "T"}},
    {{"type": "mc", "q": "Multiple-choice question", "options": ["A", "B", "C"], "answer": "A"}}
  ]
}}

STYLE CONSTRAINTS:
- HSK1 only: very short, concrete sentences; everyday verbs and nouns.
- Each story MUST include real Hanzi in "cn" (no pinyin-only lines).
- If ROMANIZATION is OFF, do NOT include the "romanization" field at all.
- Recycle high-frequency vocabulary; new-word rate about {int(level.new_word_pct[0]*100)}–{int(level.new_word_pct[1]*100)}%.
- Mini-plot: setup → small challenge → resolution.
- 2–4 lines of simple dialogue using quotes.
- Keep most lines ~6–14 Hanzi.
- Glossary ~6–10 items; only words used in the story.
- Return JSON ONLY; no commentary.
"""

def validate_story_payload(data: dict, min_hanzi_lines: int = 6) -> bool:
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
        txt = getattr(resp, "output_text", None)
        if not txt and hasattr(resp, "output") and resp.output and hasattr(resp.output[0], "content"):
            txt = "".join([c.text for c in resp.output[0].content if getattr(c, "type", "") == "output_text"])
        if not txt:
            return None
        data = json.loads(txt)
        if not romanization_on and "story" in data:
            for ln in data["story"]:
                ln.pop("romanization", None)
            for g in data.get("glossary", []):
                g.pop("romanization", None)
        return data
    except Exception:
        return None

def generate_story_with_retries(client: Optional[OpenAI], tries: int, **kwargs) -> Optional[dict]:
    if client is None:
        return None
    for _ in range(tries):
        data = try_generate_story_with_openai(client, **kwargs)
        if data and validate_story_payload(data):
            return data
    return None

# =========================
# ====== TTS (OpenAI) =====
# =========================

def synthesize_tts_mp3(client: OpenAI, text: str, voice: str, model: str) -> bytes:
    # omit format=; MP3 is default in newer SDKs
    resp = client.audio.speech.create(model=model, voice=voice, input=text)
    if isinstance(resp, (bytes, bytearray)):
        return bytes(resp)
    if hasattr(resp, "read"):
        return resp.read()
    content = getattr(resp, "content", None)
    if isinstance(content, (bytes, bytearray)):
        return bytes(content)
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

# =========================
# ====== PDF / FONTS ======
# =========================

class ReaderPDF(FPDF):
    def header(self):
        if hasattr(self, "book_title"):
            self.set_font(PDF_FONT_NAME, "B", 11)
            self.cell(0, 8, self.book_title, ln=1, align="R")
            self.ln(2)
    def footer(self):
        self.set_y(-12)
        self.set_font(PDF_FONT_NAME, "", 9)
        self.cell(0, 8, f"{self.page_no()}", align="C")

def register_fonts(pdf: FPDF):
    pdf._has_cjk = False
    added = False
    if os.path.exists(FONT_CJK_PATH):
        pdf.add_font(PDF_FONT_NAME, style="", fname=FONT_CJK_PATH, uni=True)
        pdf.add_font(PDF_FONT_NAME, style="B", fname=FONT_CJK_PATH, uni=True)
        pdf.add_font(PDF_FONT_NAME, style="I", fname=FONT_CJK_PATH, uni=True)
        pdf._has_cjk, added = True, True
    if not added and os.path.exists(FONT_LATIN_PATH):
        pdf.add_font(PDF_FONT_NAME, style="", fname=FONT_LATIN_PATH, uni=True)
        pdf.add_font(PDF_FONT_NAME, style="B", fname=FONT_LATIN_PATH, uni=True)
        pdf.add_font(PDF_FONT_NAME, style="I", fname=FONT_LATIN_PATH, uni=True)
        added = True
    if not added:
        raise RuntimeError(
            "No Unicode font found. Put NotoSansSC-Regular.ttf/otf and DejaVuSans.ttf under fonts/ "
            "or set FONT_PATH_CJK / FONT_PATH_LATIN."
        )

def render_introduction(pdf: FPDF):
    pdf.add_page()
    pdf.set_font(PDF_FONT_NAME, "B", 18)
    pdf.cell(0, 12, "Introduction", ln=1)
    pdf.ln(2)
    pdf.set_font(PDF_FONT_NAME, "", 12)
    intro = (
        "This graded reader uses high-frequency words and simple patterns.\n\n"
        "How to use it:\n"
        "1) Listen first without reading. Try to get the gist.\n"
        "2) Read once quickly; don’t stop for every new word.\n"
        "3) Read again carefully. Use pinyin and translation only when needed.\n"
        "4) Review often. Repetition builds fluency.\n\n"
        "Vocabulary lists highlight key words. Grammar notes point out useful forms. "
        "Answer the questions to check your understanding."
    )
    for p in intro.split("\n\n"):
        mc_full_width(pdf, p, 6)
        pdf.ln(2)

def render_pdf(book_title: str, lang: str, level_key: str, stories: List[Dict], show_romanization: bool) -> bytes:
    pdf = ReaderPDF()
    register_fonts(pdf)
    needs_cjk = ("Chinese" in lang or "HSK" in level_key)
    pdf.book_title = f"{book_title}{'' if (not needs_cjk or pdf._has_cjk) else '  [CJK font missing: hanzi may be blank]'}"
    pdf.set_auto_page_break(auto=True, margin=15)

    # Title
    pdf.add_page()
    pdf.set_font(PDF_FONT_NAME, "B", 22)
    pdf.cell(0, 16, book_title, ln=1)
    pdf.set_font(PDF_FONT_NAME, "", 12)
    mc_full_width(pdf, f"Language: {lang} - Level: {level_key}", 8)
    pdf.ln(8)
    pdf.set_font(PDF_FONT_NAME, "", 11)
    mc_full_width(pdf, "Compiled with Graded Reader Builder", 6)
    pdf.ln(6)

    # Intro
    render_introduction(pdf)

    # Table of contents
    pdf.add_page()
    toc_page = pdf.page_no()
    pdf.set_font(PDF_FONT_NAME, "B", 16)
    pdf.cell(0, 12, "Contents", ln=1)
    pdf.set_font(PDF_FONT_NAME, "", 12)
    toc_y = pdf.get_y()
    pdf.ln(2)
    toc_entries: List[Tuple[str, int]] = []

    # Stories
    for i, story in enumerate(stories, 1):
        toc_entries.append((f"{i}. {story['title']}", pdf.page_no() + 1))
        pdf.add_page()
        pdf.set_font(PDF_FONT_NAME, "B", 16)
        pdf.cell(0, 10, f"{i}. {story['title']}", ln=1)
        pdf.ln(2)

        pdf.set_font(PDF_FONT_NAME, "", 12)
        for line in story["story"]:
            mc_full_width(pdf, line.get("cn", ""), 7)
            if show_romanization and line.get("romanization"):
                mc_full_width(pdf, line["romanization"], 6, font_color=(100, 100, 100))
            if line.get("en"):
                mc_full_width(pdf, line["en"], 6, font_color=(80, 80, 80))
            pdf.ln(1)

        if story.get("glossary"):
            pdf.ln(2)
            pdf.set_font(PDF_FONT_NAME, "B", 13)
            pdf.cell(0, 9, "Vocabulary", ln=1)
            pdf.set_font(PDF_FONT_NAME, "", 12)
            for g in story["glossary"]:
                s = g["term"]
                if show_romanization and g.get("romanization"):
                    s += f" [{g['romanization']}]"
                if g.get("pos"):
                    s += f" ({g['pos']})"
                if g.get("en"):
                    s += f": {g['en']}"
                mc_full_width(pdf, "- " + s, 6)

        if story.get("grammar_note") and story["grammar_note"].get("point"):
            pdf.ln(2)
            pdf.set_font(PDF_FONT_NAME, "B", 13)
            pdf.cell(0, 9, "Grammar note", ln=1)
            pdf.set_font(PDF_FONT_NAME, "", 12)
            mc_full_width(pdf, story["grammar_note"]["point"], 6)
            for ex in story["grammar_note"].get("examples", []):
                mc_full_width(pdf, "- " + ex, 6)

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

    # Fill TOC
    pdf.page = toc_page
    pdf.set_y(toc_y)
    pdf.set_font(PDF_FONT_NAME, "", 12)
    for title, page_number in toc_entries:
        dots = "." * max(4, 64 - len(title))
        pdf.cell(0, 8, f"{title} {dots} {page_number}", ln=1)

    out = pdf.output(dest="S")
    return bytes(out)

# =========================
# ===== STREAMLIT UI ======
# =========================

st.set_page_config(page_title="Graded Reader Builder (PDF + MP3)", page_icon="📘", layout="wide")
st.title("📘 Graded Reader Builder")
st.caption("Create graded readers from beginner to advanced with audio and vocabulary support.")

def font_status():
    ok_cjk = os.path.exists(FONT_CJK_PATH)
    ok_lat = os.path.exists(FONT_LATIN_PATH)
    st.caption(
        f"Font check — CJK: {'✅' if ok_cjk else '❌'} ({FONT_CJK_PATH}) | "
        f"Latin: {'✅' if ok_lat else '❌'} ({FONT_LATIN_PATH})"
    )
font_status()

with st.sidebar:
    st.header("Book Settings")
    lang = st.selectbox("Language", ["Chinese (Simplified)", "Spanish", "French", "English", "German", "Portuguese"])
    level_key = st.selectbox("Level", list(LEVELS.keys()), index=0)
    topic = st.text_input("Topic/Area", "Daily routine")
    subtopics_raw = st.text_input("Subtopics (comma-separated)", "friends, weekend, park")
    n_stories = st.slider("Number of stories", 1, MAX_STORIES, 5)

    st.subheader("Story Length (approx chars/words)")
    min_words = st.number_input("Min per story", 50, 2000, 120)
    max_words = st.number_input("Max per story", 50, 4000, 220)
    difficulty_ramp = st.toggle("Difficulty ramp", True)

    show_romanization = st.toggle("Show pinyin/romanization", value=("HSK1" in level_key or "HSK2" in level_key))
    slow_audio = st.toggle("Add slow audio", False)

    st.subheader("Generation")
    use_ai_story = st.toggle("Use OpenAI story generation (JSON)", True)
    text_model = st.text_input("Text model", DEFAULT_TEXT_MODEL)
    retries = st.slider("LLM retries if no Hanzi", 1, 5, 3)

    st.subheader("Audio (TTS)")
    voice = st.text_input("TTS Voice", DEFAULT_TTS_VOICE)
    tts_model = st.text_input("TTS Model", DEFAULT_TTS_MODEL)

    st.divider()
    api_key = st.text_input("OPENAI_API_KEY", type="password")
    build_btn = st.button("Generate Book & Audio", use_container_width=True)

@st.cache_resource(show_spinner=False)
def get_openai_client(api_key: str):
    if not api_key:
        return None
    os.environ["OPENAI_API_KEY"] = api_key
    return OpenAI()

def zip_bytes(mp3_map: Dict[str, bytes]) -> bytes:
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as z:
        for fn, data in mp3_map.items():
            z.writestr(fn, data)
    buf.seek(0)
    return buf.read()

if build_btn:
    subtopics = [s.strip() for s in subtopics_raw.split(",") if s.strip()]
    client = get_openai_client(api_key) if api_key else None

    stories: List[Dict] = []
    with st.spinner("Generating stories…"):
        for idx in range(n_stories):
            target = compute_target_words(idx, n_stories, min_words, max_words, difficulty_ramp)
            story = None
            if use_ai_story and client is not None:
                story = generate_story_with_retries(
                    client, tries=retries,
                    lang=lang, level_key=level_key, topic=topic, subtopics=subtopics,
                    target_words=target, romanization_on=show_romanization, model=text_model
                )
            if story is None:
                story = placeholder_story(level_key, topic, target, show_romanization)
            # rotate grammar if LLM didn’t provide it
            if not story.get("grammar_note") or not story["grammar_note"].get("point"):
                story["grammar_note"] = GRAMMAR_PATTERNS[idx % len(GRAMMAR_PATTERNS)]
            story["title"] = f"{story['title']} ({idx+1})"
            stories.append(story)

    with st.spinner("Rendering PDF…"):
        book_title = f"{lang} - {level_key} - {topic}"
        pdf_bytes = render_pdf(book_title, lang, level_key, stories, show_romanization)
    st.success("Text & PDF ready.")
    st.download_button("📕 Download PDF", data=pdf_bytes, file_name="graded_reader.pdf", mime="application/pdf")

    mp3_files: Dict[str, bytes] = {}
    if client is None:
        st.warning("Enter OPENAI_API_KEY to enable MP3 generation.")
    else:
        with st.spinner("Synthesizing audio…"):
            for i, s in enumerate(stories, 1):
                cn_text = " ".join([ln.get("cn","") for ln in s["story"] if ln.get("cn")]).strip()
                if not cn_text:
                    continue
                try:
                    mp3_files[f"story_{i:02d}_normal.mp3"] = synthesize_tts_mp3(client, cn_text, voice, tts_model)
                    if slow_audio:
                        prefix = "（慢速朗读） " if ("Chinese" in lang or "HSK" in level_key) else "(slow reading) "
                        mp3_files[f"story_{i:02d}_slow.mp3"] = synthesize_tts_mp3(client, prefix + cn_text, voice, tts_model)
                except Exception as e:
                    st.error(f"TTS failed for story {i}: {e}")
        if mp3_files:
            st.download_button("🎧 Download MP3 ZIP", data=zip_bytes(mp3_files),
                               file_name="audio_stories.zip", mime="application/zip")

    # Simple vocab export (unique across all stories)
    vocab = {}
    for s in stories:
        for g in s.get("glossary", []):
            vocab[g["term"]] = {"pinyin": g.get("romanization",""), "meaning": g.get("en","")}
    if vocab:
        st.download_button(
            "🔤 Export Vocabulary (JSON)",
            data=json.dumps(vocab, ensure_ascii=False, indent=2).encode("utf-8"),
            file_name="vocab.json",
            mime="application/json",
        )
