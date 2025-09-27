import os
import io
import json
import base64
import zipfile
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

DEFAULT_TEXT_MODEL = os.getenv("OPENAI_TEXT_MODEL", "gpt-4o-mini")    # optional story gen

# --- Font paths (override via env or keep the defaults and place files in fonts/) ---
FONT_DIR = os.getenv("FONT_DIR", "fonts")
FONT_LATIN_PATH = os.getenv("FONT_PATH_LATIN", os.path.join(FONT_DIR, "DejaVuSans.ttf"))
FONT_CJK_PATH   = os.getenv("FONT_PATH_CJK",   os.path.join(FONT_DIR, "NotoSansSC-Regular.otf"))
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
    "HSK1 (A1)":  LevelProfile("HSK1 (A1)",  (8, 24),  (0.02, 0.05), "pinyin", "hanzi"),
    "HSK2 (A1+)": LevelProfile("HSK2 (A1+)", (10, 28), (0.03, 0.06), "pinyin", "hanzi"),
    "HSK3 (A2)":  LevelProfile("HSK3 (A2)",  (12, 30), (0.03, 0.06), "pinyin", "hanzi"),
    "HSK4 (B1)":  LevelProfile("HSK4 (B1)",  (14, 34), (0.03, 0.07), "none",   "hanzi"),
    "HSK5 (B2)":  LevelProfile("HSK5 (B2)",  (16, 38), (0.03, 0.08), "none",   "hanzi"),
    "HSK6 (C1)":  LevelProfile("HSK6 (C1)",  (18, 44), (0.03, 0.10), "none",   "hanzi"),
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

# A safe multi_cell that always uses the full effective page width (fpdf2: pdf.epw)
# and resets X to left margin to avoid "Not enough horizontal space" errors.

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
    # gentle curve that increases across stories
    t = i / (total - 1)
    val = min_w + (max_w - min_w) * (t ** 1.4)
    return int(val)

# =========================
# === PLACEHOLDER GEN =====
# =========================

def placeholder_story(lang: str, level_key: str, topic: str, target_words: int, romanization_on: bool) -> Dict:
    if "Chinese" in lang or "HSK" in level_key:
        lines = [
            {"cn":"你好！你叫什么名字？","romanization":"Nǐ hǎo! Nǐ jiào shénme míngzi?","en":"Hello! What’s your name?"},
            {"cn":"我叫李明。你呢？","romanization":"Wǒ jiào Lǐ Míng. Nǐ ne?","en":"I’m Li Ming. And you?"},
            {"cn":"今天我们在公园见面，一边走一边聊天。","romanization":"Jīntiān wǒmen zài gōngyuán jiànmiàn, yībiān zǒu yībiān liáotiān.","en":"We meet in the park and talk while walking."},
            {"cn":"天气很好，可是风不小，我们带了雨伞。","romanization":"Tiānqì hěn hǎo, kěshì fēng bù xiǎo, wǒmen dàile yǔsǎn.","en":"Nice weather, but windy; we brought umbrellas."},
        ]
        gloss = [
            {"term":"公园","romanization":"gōngyuán","pos":"n.","en":"park"},
            {"term":"聊天","romanization":"liáotiān","pos":"v.","en":"to chat"},
        ]
        note = {"point":"一边…一边… (do two actions at once)","examples":["我一边走一边说话。","她一边听音乐一边做饭。"]}
        qs = [{"type":"tf","q":"他们在公园见面。","answer":"T"}]
    else:
        lines = [
            {"cn":"Hello! This is a placeholder line.","romanization":"","en":"Replace with generator."},
            {"cn":"We are going to the park and talking.","romanization":"","en":"Replace with generator."},
        ]
        gloss, note, qs = [], {"point":"","examples":[]}, []

    total = sum(tokens_len(l["cn"]) for l in lines)
    i = 0
    while total < target_words:
        lines.append(lines[i % len(lines)])
        total = sum(tokens_len(l["cn"]) for l in lines)
        i += 1

    if not romanization_on:
        for l in lines:
            l["romanization"] = ""

    return {
        "title": f"{topic} ({level_key})",
        "story": lines,
        "glossary": gloss,
        "grammar_note": note,
        "questions": qs
    }

# =========================
# ==== AI GENERATION ======
# =========================

def build_story_json_system_prompt(lang: str, level: LevelProfile, romanization_on: bool, topic: str, subtopics: List[str], target_words: int) -> str:
    ro = "ON" if romanization_on else "OFF"
    subs = ", ".join(subtopics) if subtopics else "None"
    return f"""You are a graded-reader author.
Language: {lang}. Level: {level.name}. Romanization: {ro}.
Topic: {topic}. Subtopics: {subs}.
Target words per story: {target_words}.
Constraints:
- Keep the story within ±10% of target words.
- Use short, natural sentences appropriate for {level.name}.
- Recycle vocabulary; new-word rate ≈ {int(level.new_word_pct[0]*100)}–{int(level.new_word_pct[1]*100)}%.
- Include brief, realistic dialogue (2–4 lines).
- Include a mini plot (setup → small problem → resolution).
- If Chinese: prefer high-frequency words; if romanization is OFF, omit pinyin.

Return valid JSON only with keys:
{{
 "title": "string",
 "story": [{{"cn":"string","romanization":"string (optional)","en":"string"}}],
 "glossary": [{{"term":"string","romanization":"string (optional)","pos":"string","en":"string"}}],
 "grammar_note": {{"point":"string","examples":["string","string"]}},
 "questions": [{{"type":"tf|mc","q":"string","options":["string","string"],"answer":"string"}}]
}}
    """

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
        if not isinstance(data, dict) or "story" not in data:
            return None
        if not romanization_on:
            for ln in data.get("story", []):
                ln["romanization"] = ""
            for g in data.get("glossary", []):
                g["romanization"] = ""
        return data
    except Exception:
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
            "No Unicode font found. Add a CJK font (e.g., NotoSansSC-Regular.otf/ttf) "
            "or a Unicode Latin font (e.g., DejaVuSans.ttf) to fonts/ or set FONT_PATH_* env vars."
        )

def render_pdf(book_title: str, lang: str, level_key: str, stories: List[Dict], show_romanization: bool) -> bytes:
    pdf = ReaderPDF()
    register_unicode_font(pdf)  # ensure Unicode font before any text ops

    # Soft fallback: if Chinese is selected but no CJK font is loaded, append a warning to the title
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

    st.subheader("Generation Engines")
    use_ai_story = st.toggle("Use OpenAI to generate stories (JSON)", value=False)
    text_model = st.text_input("Text model for stories", DEFAULT_TEXT_MODEL)

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
                story = try_generate_story_with_openai(
                    client, lang, level_key, topic, subtopics,
                    target_words, show_romanization, text_model
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
                cn_text = " ".join([ln.get("cn","") for ln in s["story"] if ln.get("cn")])
                if not cn_text.strip():
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
