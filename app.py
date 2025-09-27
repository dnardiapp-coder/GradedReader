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

# =========================
# ====== CONFIG ===========
# =========================

DEFAULT_TTS_MODEL = os.getenv("OPENAI_TTS_MODEL", "gpt-4o-mini-tts")  # or "tts-1"
DEFAULT_TTS_VOICE = os.getenv("OPENAI_TTS_VOICE", "alloy")
DEFAULT_AUDIO_FORMAT = "mp3"  # not passed to SDK (some versions reject the kwarg)

DEFAULT_TEXT_MODEL = os.getenv("OPENAI_TEXT_MODEL", "gpt-4o-mini")    # story gen

# --- Font paths (override via env or keep the defaults and place files in fonts/) ---
FONT_DIR = os.getenv("FONT_DIR", "fonts")
FONT_LATIN_PATH = os.getenv("FONT_PATH_LATIN", os.path.join(FONT_DIR, "DejaVuSans.ttf"))
FONT_CJK_PATH   = os.getenv("FONT_PATH_CJK",   os.path.join(FONT_DIR, "NotoSansSC-Regular.ttf"))  # try .ttf first
if not os.path.exists(FONT_CJK_PATH):
    # fallback to .otf if .ttf not present
    FONT_CJK_PATH = os.getenv("FONT_PATH_CJK", os.path.join(FONT_DIR, "NotoSansSC-Regular.otf"))

PDF_FONT_NAME   = "AppSans"   # single logical family we’ll use everywhere

MAX_STORIES = 20

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
    "HSK2 (A1+)": LevelProfile("HSK2 (A1+)", (8, 18), (0.03, 0.06), "pinyin", "hanzi"),
    "HSK3 (A2)":  LevelProfile("HSK3 (A2)",  (10, 22), (0.03, 0.06), "pinyin", "hanzi"),
    "HSK4 (B1)":  LevelProfile("HSK4 (B1)",  (12, 26), (0.03, 0.07), "none",   "hanzi"),
    "HSK5 (B2)":  LevelProfile("HSK5 (B2)",  (14, 30), (0.03, 0.08), "none",   "hanzi"),
    "HSK6 (C1)":  LevelProfile("HSK6 (C1)",  (16, 34), (0.03, 0.10), "none",   "hanzi"),
    "CEFR A1":    LevelProfile("CEFR A1",    (8, 18),  (0.02, 0.05), "none", "alpha"),
    "CEFR A2":    LevelProfile("CEFR A2",    (10, 22), (0.03, 0.06), "none", "alpha"),
    "CEFR B1":    LevelProfile("CEFR B1",    (12, 26), (0.03, 0.07), "none", "alpha"),
    "CEFR B2":    LevelProfile("CEFR B2",    (14, 30), (0.03, 0.08), "none", "alpha"),
    "CEFR C1":    LevelProfile("CEFR C1",    (16, 34), (0.03, 0.10), "none", "alpha"),
}

# =========================
# ===== UTILITIES =========
# =========================

def tokens_len(s: str) -> int:
    return len(s.split())

def compute_target_words(i: int, total: int, min_w: int, max_w: int, ramp: bool) -> int:
    if not ramp or total == 1:
        return (min_w + max_w) // 2
    t = i / (total - 1)
    val = min_w + (max_w - min_w) * (t ** 1.4)
    return int(val)

# Safe multi_cell: always full effective width; reset X to left margin
def mc_full_width(pdf: FPDF, text: str, h: float, font_color: Optional[tuple] = None):
    if font_color:
        pdf.set_text_color(*font_color)
    pdf.set_x(pdf.l_margin)
    pdf.multi_cell(pdf.epw, h, text)
    if font_color:
        pdf.set_text_color(0, 0, 0)

# =========================
# === PLACEHOLDER GEN =====
# (Better HSK1-ish fallback so it doesn't look repetitive)
# =========================

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
]

def placeholder_story(lang: str, level_key: str, topic: str, target_words: int, romanization_on: bool) -> Dict:
    if "Chinese" in lang or "HSK" in level_key:
        k = min(len(BASIC_HSK1_LINES), max(8, target_words // 8))
        lines = random.sample(BASIC_HSK1_LINES, k=k)
        # Add short dialogue to create a tiny arc
        lines.insert(2, {"cn":"“你叫什么名字？”","romanization":"“Nǐ jiào shénme míngzi?”","en":"“What is your name?”"})
        lines.insert(3, {"cn":"“我叫小李。”","romanization":"“Wǒ jiào Xiǎo Lǐ.”","en":"“I’m Xiao Li.”"})
        gloss = [
            {"term":"名字","romanization":"míngzi","pos":"n.","en":"name"},
            {"term":"公园","romanization":"gōngyuán","pos":"n.","en":"park"},
            {"term":"聊天","romanization":"liáotiān","pos":"v.","en":"to chat"},
        ]
        note = {"point":"“吗”构成一般疑问句（Yes/No）。","examples":["你喜欢茶吗？","你是学生吗？"]}
        qs = [{"type":"tf","q":"他们在公园聊天。","answer":"T"}]
        if not romanization_on:
            for l in lines:
                l["romanization"] = ""
            for g in gloss:
                g["romanization"] = ""
        return {
            "title": f"{topic} ({level_key})",
            "story": lines,
            "glossary": gloss,
            "grammar_note": note,
            "questions": qs
        }
    # Non-Chinese fallback
    lines = [
        {"cn":"Hello! This is a placeholder line.","romanization":"","en":"Replace with generator."},
        {"cn":"We go to the park and talk.","romanization":"","en":"Replace with generator."},
    ]
    if not romanization_on:
        for l in lines:
            l["romanization"] = ""
    return {"title": f"{topic} ({level_key})", "story": lines, "glossary": [], "grammar_note": {"point":"", "examples":[]}, "questions": []}

# =========================
# ==== AI GENERATION ======
# =========================

def build_story_json_system_prompt(lang: str, level: LevelProfile, romanization_on: bool, topic: str, subtopics: List[str], target_words: int) -> str:
    ro = "ON" if romanization_on else "OFF"
    subs = ", ".join(subtopics) if subtopics else "None"
    return f"""
You are writing an HSK-style graded reader story.

LANGUAGE: {lang}
LEVEL: {level.name}  (HSK1 ≈ 150 words; very simple sentences)
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
    {{"type": "mc", "q": "One multiple-choice question", "options": ["A", "B", "C"], "answer": "A"}}
  ]
}}

STYLE CONSTRAINTS:
- HSK1 only: very short, concrete sentences; everyday verbs and nouns.
- Each story MUST include real Hanzi in "cn" (no pinyin-only lines).
- If ROMANIZATION is OFF, do NOT include the "romanization" field at all.
- Recycle HSK1-range words; new-word rate about {int(level.new_word_pct[0]*100)}–{int(level.new_word_pct[1]*100)}%.
- Mini-plot: setup → small challenge → resolution.
- 2–4 lines of simple dialogue using quotes.
- Keep most lines ~6–14 Hanzi.
- Glossary ~6–10 items; only words used in the story.
- Return JSON ONLY; no commentary.
"""

HANZI_RE = re.compile(r"[\u4e00-\u9fff]")

def has_hanzi(s: str) -> bool:
    return bool(HANZI_RE.search(s or ""))

def validate_story_payload(data: dict, min_hanzi_lines: int = 6) -> bool:
    if not isinstance(data, dict):
        return False
    lines = data.get("story", [])
    if not isinstance(lines, list) or not lines:
        return False
    hanzi_count = sum(1 for ln in lines if has_hanzi(ln.get("cn", "")))
    return hanzi_count >= min_hanzi_lines

def try_generate_story_with_openai(client: OpenAI, lang: str, level_key: str, topic: str, subtopics: List[str],
                                   target_words: int, romanization_on: bool, model: str) -> Optional[Dict]:
    level = LEVELS[level_key]
    system_prompt = build_story_json_system_prompt(lang, level, romanization_on, topic, subtopics, target_words)
    try:
        resp = client.responses.create(
            model=model,
            input=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": "Generate one story JSON now."}
            ],
            response_format={"type": "json_object"}
        )
        txt = getattr(resp, "output_text", None)
        if not txt and hasattr(resp, "output") and resp.output and hasattr(resp.output[0], "content"):
            txt = "".join([c.text for c in resp.output[0].content if getattr(c, "type", "") == "output_text"])
        data = json.loads(txt or "{}")
        # If romanization is OFF, remove the field entirely to keep PDF clean
        if not romanization_on and "story" in data:
            for ln in data["story"]:
                ln.pop("romanization", None)
        return data
    except Exception:
        return None

def generate_story_with_retries(client: Optional[OpenAI], *, tries=3, **kw) -> Optional[dict]:
    for _ in range(tries):
        data = try_generate_story_with_openai(client, **kw) if client else None
        if data and validate_story_payload(data):
            return data
    return None

# =========================
# ====== TTS (OpenAI) =====
# =========================

def synthesize_tts_mp3(client: OpenAI, text: str, voice: str, model: str, fmt: str) -> bytes:
    """
    Some SDK versions reject a `format=` kwarg; MP3 is default. We omit it.
    """
    resp = client.audio.speech.create(
        model=model,
        voice=voice,
        input=text,
    )
    # Common return shapes across versions
    if isinstance(resp, (bytes, bytearray)):
        return bytes(resp)
    if hasattr(resp, "read"):
        return resp.read()
    data = getattr(resp, "content", None)
    if isinstance(data, (bytes, bytearray)):
        return bytes(data)
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

def register_unicode_font(pdf: FPDF):
    """
    Register a Unicode-capable font family. Prefer CJK font if available.
    Sets pdf._has_cjk to True/False so we can guard against missing Hanzi glyphs.
    """
    pdf._has_cjk = False
    family_added = False
    if os.path.exists(FONT_CJK_PATH):
        pdf.add_font(PDF_FONT_NAME, style="", fname=FONT_CJK_PATH, uni=True)
        pdf.add_font(PDF_FONT_NAME, style="B", fname=FONT_CJK_PATH, uni=True)
        pdf.add_font(PDF_FONT_NAME, style="I", fname=FONT_CJK_PATH, uni=True)
        family_added = True
        pdf._has_cjk = True
    if not family_added and os.path.exists(FONT_LATIN_PATH):
        pdf.add_font(PDF_FONT_NAME, style="", fname=FONT_LATIN_PATH, uni=True)
        pdf.add_font(PDF_FONT_NAME, style="B", fname=FONT_LATIN_PATH, uni=True)
        pdf.add_font(PDF_FONT_NAME, style="I", fname=FONT_LATIN_PATH, uni=True)
        family_added = True
    if not family_added:
        raise RuntimeError(
            "No Unicode font found. Add a CJK font (e.g., NotoSansSC-Regular.ttf/otf) "
            "or a Unicode Latin font (e.g., DejaVuSans.ttf) to fonts/ or set FONT_PATH_* env vars."
        )

def render_pdf(book_title: str, lang: str, level_key: str, stories: List[Dict], show_romanization: bool) -> bytes:
    pdf = ReaderPDF()
    register_unicode_font(pdf)  # ensure Unicode font before any text ops

    # Soft fallback title if Chinese selected but no CJK font found
    cjk_required = ("Chinese" in lang or "HSK" in level_key)
    cjk_ok = getattr(pdf, "_has_cjk", False)
    if cjk_required and not cjk_ok:
        pdf.book_title = f"{book_title}  [CJK font missing: hanzi may be blank]"
    else:
        pdf.book_title = book_title

    pdf.set_auto_page_break(auto=True, margin=15)

    # Title
    pdf.add_page()
    pdf.set_font(PDF_FONT_NAME, "B", 20)
    pdf.cell(0, 12, pdf.book_title, ln=1)
    pdf.set_font(PDF_FONT_NAME, "", 12)
    mc_full_width(pdf, f"Language: {lang} - Level: {level_key}", 8)
    pdf.ln(4)

    # TOC
    pdf.set_font(PDF_FONT_NAME, "B", 14)
    pdf.cell(0, 10, "Contents", ln=1)
    pdf.set_font(PDF_FONT_NAME, "", 12)
    pdf.set_x(pdf.l_margin)
    for i, s in enumerate(stories, 1):
        pdf.cell(0, 8, f"{i}. {s['title']}", ln=1)

    # Stories
    for i, story in enumerate(stories, 1):
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

        if story["glossary"]:
            pdf.ln(2)
            pdf.set_font(PDF_FONT_NAME, "B", 13)
            pdf.cell(0, 9, "Vocabulary", ln=1)
            pdf.set_font(PDF_FONT_NAME, "", 12)
            for g in story["glossary"]:
                line = g.get("term", "")
                if show_romanization and g.get("romanization"):
                    line += f" [{g['romanization']}]"
                if g.get("pos"):
                    line += f" ({g['pos']})"
                if g.get("en"):
                    line += f": {g['en']}"
                mc_full_width(pdf, "- " + line, 6)

        if story["grammar_note"].get("point"):
            pdf.ln(2)
            pdf.set_font(PDF_FONT_NAME, "B", 13)
            pdf.cell(0, 9, "Grammar note", ln=1)
            pdf.set_font(PDF_FONT_NAME, "", 12)
            mc_full_width(pdf, story["grammar_note"]["point"], 6)
            for ex in story["grammar_note"].get("examples", []):
                mc_full_width(pdf, f"- {ex}", 6)

        if story["questions"]:
            pdf.ln(2)
            pdf.set_font(PDF_FONT_NAME, "B", 13)
            pdf.cell(0, 9, "Comprehension", ln=1)
            pdf.set_font(PDF_FONT_NAME, "", 12)
            for q in story["questions"]:
                if q["type"] == "tf":
                    mc_full_width(pdf, f"- (T/F) {q['q']}", 6)
                elif q["type"] == "mc":
                    mc_full_width(pdf, f"- {q['q']}", 6)
                    for opt in q.get("options", []):
                        mc_full_width(pdf, f"   * {opt}", 6)

    out = pdf.output(dest="S")
    return bytes(out)  # normalize bytes/bytearray across fpdf2 versions

# =========================
# ===== STREAMLIT UI ======
# =========================

st.set_page_config(page_title="Graded Reader Builder (PDF + MP3)", page_icon="📘", layout="wide")
st.title("📘 Graded Reader Builder")

# ---- Quick font status line (helps debug paths case-sensitively on Linux)
def _font_status():
    ok_cjk = os.path.exists(FONT_CJK_PATH)
    ok_lat = os.path.exists(FONT_LATIN_PATH)
    st.caption(
        f"Font check — CJK: {'✅' if ok_cjk else '❌'} ({FONT_CJK_PATH}) | "
        f"Latin: {'✅' if ok_lat else '❌'} ({FONT_LATIN_PATH})"
    )

_font_status()
st.caption("Create graded readers from beginner to advanced with audio.")

with st.sidebar:
    st.header("Settings")
    lang = st.selectbox(
        "Language",
        ["Chinese (Simplified)", "Spanish", "French", "English", "German", "Portuguese"]
    )
    level_key = st.selectbox("Level", list(LEVELS.keys()), index=0)

    topic = st.text_input("Topic/Area", "Daily routine")
    subtopics_raw = st.text_input("Subtopics (comma-separated)", "friends, weekend, park")
    n_stories = st.slider("Number of stories", 1, MAX_STORIES, 5)

    st.subheader("Story Length")
    min_words = st.number_input("Min words per story", 50, 2000, 120)
    max_words = st.number_input("Max words per story", 50, 4000, 220)
    difficulty_ramp = st.toggle("Apply difficulty ramp (later stories longer)", value=True)

    show_romanization = st.toggle("Show pinyin/romanization", value=("HSK1" in level_key or "HSK2" in level_key))
    slow_audio = st.toggle("Add slow audio version", value=False)

    st.subheader("Generation")
    use_ai_story = st.toggle("Use OpenAI to generate stories (JSON)", value=True)
    text_model = st.text_input("Text model for stories", DEFAULT_TEXT_MODEL)
    retries = st.slider("LLM retries if no Hanzi", 1, 5, 3)

    st.subheader("Audio (TTS)")
    voice = st.text_input("TTS Voice", DEFAULT_TTS_VOICE)
    tts_model = st.text_input("TTS Model", DEFAULT_TTS_MODEL)

    st.divider()
    st.markdown("**OpenAI API Key**")
    api_key = st.text_input("OPENAI_API_KEY", type="password")

    build_btn = st.button("Generate Book & Audio", type="primary", use_container_width=True)

@st.cache_resource(show_spinner=False)
def get_openai_client(api_key: str):
    if not api_key:
        return None
    os.environ["OPENAI_API_KEY"] = api_key
    return OpenAI()

def zip_bytes(mp3_map: Dict[str, bytes]) -> bytes:
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as z:
        for filename, content in mp3_map.items():
            z.writestr(filename, content)
    buf.seek(0)
    return buf.read()

if build_btn:
    subtopics = [s.strip() for s in subtopics_raw.split(",") if s.strip()]

    # 1) Build stories
    stories: List[Dict] = []
    client: Optional[OpenAI] = get_openai_client(api_key) if api_key else None

    with st.spinner("Generating stories…"):
        for idx in range(n_stories):
            target_words = compute_target_words(idx, n_stories, min_words, max_words, difficulty_ramp)
            story = None
            if use_ai_story and client:
                story = generate_story_with_retries(
                    client,
                    tries=retries,
                    lang=lang,
                    level_key=level_key,
                    topic=topic,
                    subtopics=subtopics,
                    target_words=target_words,
                    romanization_on=show_romanization,
                    model=text_model
                )
            if story is None:
                story = placeholder_story(lang, level_key, topic, target_words, show_romanization)
            story["title"] = f"{story['title']} ({idx+1})"
            stories.append(story)

    # 2) Render PDF (no images)
    with st.spinner("Rendering PDF…"):
        book_title = f"{lang} - {level_key} - {topic}"
        pdf_bytes = render_pdf(book_title, lang, level_key, stories, show_romanization)
    st.success("Text & PDF ready.")
    st.download_button("📕 Download PDF", data=pdf_bytes, file_name="graded_reader.pdf", mime="application/pdf")

    # 3) TTS (MP3)
    mp3_files: Dict[str, bytes] = {}
    if not client:
        st.warning("Enter your OPENAI_API_KEY to enable MP3 generation.")
    else:
        with st.spinner("Synthesizing audio…"):
            for i, s in enumerate(stories, 1):
                cn_text = " ".join([ln.get("cn","") for ln in s["story"] if ln.get("cn")]).strip()
                if not cn_text:
                    continue
                try:
                    audio_bytes = synthesize_tts_mp3(client, cn_text, voice=voice, model=tts_model, fmt=DEFAULT_AUDIO_FORMAT)
                    mp3_files[f"story_{i:02d}_normal.mp3"] = audio_bytes
                    if slow_audio:
                        slow_prefix = "（慢速朗读） " if ("Chinese" in lang or "HSK" in level_key) else "(slow reading) "
                        audio_bytes_slow = synthesize_tts_mp3(client, slow_prefix + cn_text, voice=voice, model=tts_model, fmt=DEFAULT_AUDIO_FORMAT)
                        mp3_files[f"story_{i:02d}_slow.mp3"] = audio_bytes_slow
                except Exception as e:
                    st.error(f"TTS failed for story {i}: {e}")

        if mp3_files:
            zbytes = zip_bytes(mp3_files)
            st.download_button("🎧 Download MP3 ZIP", data=zbytes, file_name="audio_stories.zip", mime="application/zip")

    # 4) Vocab export (stub)
    vocab_payload = {"note": "Plug in real vocab extraction if using a lexicon.", "stories": len(stories)}
    st.download_button(
        "🔤 Export Vocabulary (JSON)",
        data=json.dumps(vocab_payload, ensure_ascii=False, indent=2).encode("utf-8"),
        file_name="vocab.json",
        mime="application/json",
    )
