import os
import io
import json
import math
import base64
import zipfile
import tempfile
from dataclasses import dataclass
from typing import List, Dict, Tuple, Optional

import streamlit as st
from fpdf import FPDF
from PIL import Image

# --- OpenAI SDK (Text + TTS + Images) ---
# pip install openai>=1.40.0
# Docs:
#  - TTS: https://platform.openai.com/docs/guides/text-to-speech
#  - Images: https://platform.openai.com/docs/guides/images
#  - Responses (JSON mode): https://platform.openai.com/docs/guides/responses
from openai import OpenAI


# =========================
# ====== CONFIG ===========
# =========================

DEFAULT_TTS_MODEL = os.getenv("OPENAI_TTS_MODEL", "gpt-4o-mini-tts")  # or "tts-1"
DEFAULT_TTS_VOICE = os.getenv("OPENAI_TTS_VOICE", "alloy")            # choose any available voice
DEFAULT_AUDIO_FORMAT = "mp3"                                          # mp3 | m4a | wav (depends on SDK version)

DEFAULT_TEXT_MODEL = os.getenv("OPENAI_TEXT_MODEL", "gpt-4o-mini")    # for optional AI story generation
DEFAULT_IMAGE_MODEL = os.getenv("OPENAI_IMAGE_MODEL", "gpt-image-1")  # for auto-generated illustrations

MAX_STORIES = 20
MAX_IMG_WIDTH_PX = 1200

# =========================
# ====== LEVELS ===========
# =========================

@dataclass
class LevelProfile:
    name: str
    sentence_len: Tuple[int, int]      # rough chars per sentence (heuristic for style)
    new_word_pct: Tuple[float, float]  # allowable % of new words
    romanization: str                  # "pinyin"/"romaji"/"none"
    script: str                        # "hanzi"/"alpha" etc.

LEVELS: Dict[str, LevelProfile] = {
    # HSK tracks (Beginner -> Advanced)
    "HSK1 (A1)":  LevelProfile("HSK1 (A1)",  (8, 24),  (0.02, 0.05), "pinyin", "hanzi"),
    "HSK2 (A1+)": LevelProfile("HSK2 (A1+)", (10, 28), (0.03, 0.06), "pinyin", "hanzi"),
    "HSK3 (A2)":  LevelProfile("HSK3 (A2)",  (12, 30), (0.03, 0.06), "pinyin", "hanzi"),
    "HSK4 (B1)":  LevelProfile("HSK4 (B1)",  (14, 34), (0.03, 0.07), "none",   "hanzi"),
    "HSK5 (B2)":  LevelProfile("HSK5 (B2)",  (16, 38), (0.03, 0.08), "none",   "hanzi"),
    "HSK6 (C1)":  LevelProfile("HSK6 (C1)",  (18, 44), (0.03, 0.10), "none",   "hanzi"),
    # CEFR tracks (multilingual)
    "CEFR A1":    LevelProfile("CEFR A1",    (8, 18),  (0.02, 0.05), "none", "alpha"),
    "CEFR A2":    LevelProfile("CEFR A2",    (10, 22), (0.03, 0.06), "none", "alpha"),
    "CEFR B1":    LevelProfile("CEFR B1",    (12, 26), (0.03, 0.07), "none", "alpha"),
    "CEFR B2":    LevelProfile("CEFR B2",    (14, 30), (0.03, 0.08), "none", "alpha"),
    "CEFR C1":    LevelProfile("CEFR C1",    (16, 34), (0.03, 0.10), "none", "alpha"),
}


# =========================
# ====== UTILITIES =========
# =========================

def clamp(n: float, lo: float, hi: float) -> float:
    return max(lo, min(n, hi))

def approx_target_lines(target_words: int, avg_words_per_line: int = 10) -> int:
    return max(3, target_words // avg_words_per_line)

def tokens_len(s: str) -> int:
    return len(s.split())


# =========================
# ====== PLACEHOLDER GEN ===
# =========================

def placeholder_story(lang: str, level_key: str, topic: str, target_words: int, romanization_on: bool) -> Dict:
    """
    Deterministic placeholder story. Replace with LLM generation for production.
    """
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

    # Expand approximately to target_words
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
        "title": f"{topic}（{level_key}）",
        "story": lines,
        "glossary": gloss,
        "grammar_note": note,
        "questions": qs
    }


# =========================
# ====== AI GENERATION =====
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
- If Chinese {('(HSK track)') if level.script=='hanzi' else ''}: prefer high-frequency words; if romanization is OFF, omit pinyin.

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
    """Attempts JSON story generation via Responses API. Falls back to placeholder on error."""
    level = LEVELS[level_key]
    system_prompt = build_story_json_system_prompt(lang, level, romanization_on, topic, subtopics, target_words)

    try:
        resp = client.responses.create(
            model=model,
            input=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": "Generate one story JSON now."}
            ],
            response_format={"type":"json_object"}
        )
        # New SDK returns a top-level object with output_text or JSON in 'output'
        if hasattr(resp, "output") and resp.output and hasattr(resp.output[0], "content"):
            txt = "".join([c.text for c in resp.output[0].content if c.type=="output_text"])
        else:
            # Fallback: many SDKs expose .output_text directly
            txt = getattr(resp, "output_text", None) or json.dumps({})
        data = json.loads(txt)
        # sanity checks
        if not isinstance(data, dict) or "story" not in data:
            return None
        if not romanization_on:
            # strip romanization fields if present
            for ln in data.get("story", []):
                ln["romanization"] = ""
            for g in data.get("glossary", []):
                g["romanization"] = ""
        return data
    except Exception as e:
        # You can log/print for debugging if running locally
        return None


# =========================
# ====== TTS (OpenAI) =====
# =========================

def synthesize_tts_mp3(client: OpenAI, text: str, voice: str, model: str, fmt: str) -> bytes:
    """
    Uses OpenAI TTS to produce audio bytes.
    Depending on SDK version, streaming helpers may exist.
    """
    # Preferred (streaming) pattern in some SDK versions:
    # with client.audio.speech.with_streaming_response.create(
    #     model=model, voice=voice, input=text, format=fmt
    # ) as resp:
    #     return resp.read()

    # Broadly compatible fallback:
    resp = client.audio.speech.create(
        model=model,
        voice=voice,
        input=text,
        format=fmt,  # mp3/m4a/wav based on SDK
    )
    # Try common return shapes:
    if isinstance(resp, (bytes, bytearray)):
        return resp
    if hasattr(resp, "read"):
        return resp.read()
    if isinstance(resp, dict) and "data" in resp:
        return resp["data"]
    # Some SDKs return .audio.data (base64). Try to decode if present.
    try:
        b64 = getattr(resp, "audio", {}).get("data", None)
        if b64:
            return base64.b64decode(b64)
    except Exception:
        pass
    raise RuntimeError("Unexpected TTS response; check SDK version or update synthesize_tts_mp3")


# =========================
# ====== IMAGES ===========
# =========================

def resize_for_pdf(im: Image.Image, max_width_px: int = MAX_IMG_WIDTH_PX) -> Image.Image:
    if im.width <= max_width_px:
        return im
    ratio = max_width_px / im.width
    new_size = (int(im.width * ratio), int(im.height * ratio))
    return im.resize(new_size, Image.LANCZOS)

def save_temp_png(im: Image.Image) -> str:
    f = tempfile.NamedTemporaryFile(delete=False, suffix=".png")
    im.save(f.name, format="PNG", optimize=True)
    return f.name

def generate_image_from_story(client: OpenAI, story_title: str, story_lines: list, style: str, seed_prompt: str, model: str) -> Image.Image:
    # Build a concise, safe prompt
    scene = story_title + " — " + " ".join(
        (ln.get("en") or ln.get("cn","")) for ln in story_lines[:6]
    )
    prompt = f"""{seed_prompt}
Style: {style}.
Scene summary: {scene}
Content guidelines: clear, classroom-friendly, non-violent, inclusive, white/neutral background if illustration.
"""
    resp = client.images.generate(
        model=model,
        prompt=prompt,
        size="1024x1024",
        quality="standard"
    )
    b64 = resp.data[0].b64_json
    img_bytes = base64.b64decode(b64)
    return Image.open(io.BytesIO(img_bytes)).convert("RGB")


# =========================
# ====== PDF RENDER =======
# =========================

class ReaderPDF(FPDF):
    def header(self):
        if hasattr(self, "book_title"):
            self.set_font("Helvetica", "B", 11)
            self.cell(0, 8, self.book_title, 0, 1, "R")
            self.ln(2)

    def footer(self):
        self.set_y(-12)
        self.set_font("Helvetica", "", 9)
        self.cell(0, 8, f"{self.page_no()}", 0, 0, "C")

def embed_image_pdf(pdf: FPDF, img_path: str, placement: str):
    page_w = pdf.w - pdf.l_margin - pdf.r_margin
    if placement.startswith("Full"):
        pdf.image(img_path, x=pdf.l_margin, w=page_w)
        pdf.ln(4)
    else:
        pdf.image(img_path, x=pdf.l_margin, w=page_w/2)
        pdf.ln(2)

def render_pdf(book_title: str, lang: str, level_key: str, stories: List[Dict], show_romanization: bool,
               image_paths: List[Optional[str]], img_placement: str) -> bytes:
    pdf = ReaderPDF()
    pdf.book_title = book_title
    pdf.set_auto_page_break(auto=True, margin=15)

    # Title
    pdf.add_page()
    pdf.set_font("Helvetica", "B", 20)
    pdf.cell(0, 12, book_title, ln=1)
    pdf.set_font("Helvetica", "", 12)
    pdf.multi_cell(0, 8, f"Language: {lang} • Level: {level_key}")

    # TOC
    pdf.ln(4)
    pdf.set_font("Helvetica", "B", 14)
    pdf.cell(0, 10, "Contents", ln=1)
    pdf.set_font("Helvetica", "", 12)
    for i, s in enumerate(stories, 1):
        pdf.cell(0, 8, f"{i}. {s['title']}", ln=1)

    # Stories
    for i, story in enumerate(stories, 1):
        pdf.add_page()
        pdf.set_font("Helvetica", "B", 16)
        pdf.cell(0, 10, f"{i}. {story['title']}", ln=1)
        pdf.ln(2)

        # Image (if any)
        if image_paths and image_paths[i-1]:
            embed_image_pdf(pdf, image_paths[i-1], img_placement)

        pdf.set_font("Helvetica", "", 12)
        for line in story["story"]:
            pdf.multi_cell(0, 7, line.get("cn",""))
            if show_romanization and line.get("romanization"):
                pdf.set_text_color(100, 100, 100)
                pdf.multi_cell(0, 6, line["romanization"])
                pdf.set_text_color(0, 0, 0)
            if line.get("en"):
                pdf.set_text_color(80, 80, 80)
                pdf.multi_cell(0, 6, line["en"])
                pdf.set_text_color(0, 0, 0)
            pdf.ln(1)

        if story["glossary"]:
            pdf.ln(2)
            pdf.set_font("Helvetica", "B", 13)
            pdf.cell(0, 9, "Vocabulary", ln=1)
            pdf.set_font("Helvetica", "", 12)
            for g in story["glossary"]:
                line = g["term"]
                if show_romanization and g.get("romanization"):
                    line += f" [{g['romanization']}]"
                if g.get("pos"):
                    line += f" ({g['pos']})"
                if g.get("en"):
                    line += f": {g['en']}"
                pdf.multi_cell(0, 6, "• " + line)

        if story["grammar_note"].get("point"):
            pdf.ln(2)
            pdf.set_font("Helvetica", "B", 13)
            pdf.cell(0, 9, "Grammar note", ln=1)
            pdf.set_font("Helvetica", "", 12)
            pdf.multi_cell(0, 6, story["grammar_note"]["point"])
            for ex in story["grammar_note"].get("examples", []):
                pdf.multi_cell(0, 6, f"— {ex}")

        if story["questions"]:
            pdf.ln(2)
            pdf.set_font("Helvetica", "B", 13)
            pdf.cell(0, 9, "Comprehension", ln=1)
            pdf.set_font("Helvetica", "", 12)
            for q in story["questions"]:
                if q["type"] == "tf":
                    pdf.multi_cell(0, 6, f"• (T/F) {q['q']}")
                elif q["type"] == "mc":
                    pdf.multi_cell(0, 6, f"• {q['q']}")
                    for opt in q.get("options", []):
                        pdf.multi_cell(0, 6, f"   - {opt}")

    return pdf.output(dest="S").encode("latin1")


# =========================
# ====== STREAMLIT UI ======
# =========================

st.set_page_config(page_title="Graded Reader Builder (PDF + MP3 + Images)", page_icon="📘", layout="wide")
st.title("📘 Graded Reader Builder")
st.caption("Create level-appropriate graded readers with audio (MP3) and optional images. Listening → reading best practice.")

with st.sidebar:
    st.header("Settings")
    lang = st.selectbox("Language", ["Chinese (Simplified)", "Spanish", "French", "English", "German", "Portuguese"])
    level_key = st.selectbox("Level", list(LEVELS.keys()), index=0)

    topic = st.text_input("Topic/Area", "Daily routine")
    subtopics_raw = st.text_input("Subtopics (comma-separated)", "friends, weekend, park")
    n_stories = st.slider("Number of stories", 1, MAX_STORIES, 5)

    # Size & Difficulty
    st.subheader("Story Length")
    min_words = st.number_input("Min words per story", 50, 2000, 120)
    max_words = st.number_input("Max words per story", 50, 4000, 220)
    difficulty_ramp = st.toggle("Apply difficulty ramp (later stories longer)", value=True)

    show_romanization = st.toggle("Show pinyin/romanization", value=("HSK1" in level_key or "HSK2" in level_key))
    slow_audio = st.toggle("Add slow audio version", value=False)

    # AI toggles
    st.subheader("Generation Engines")
    use_ai_story = st.toggle("Use OpenAI to generate stories (JSON)", value=False)
    text_model = st.text_input("Text model for stories", DEFAULT_TEXT_MODEL)

    # TTS
    st.subheader("Audio (TTS)")
    voice = st.text_input("TTS Voice", DEFAULT_TTS_VOICE)
    tts_model = st.text_input("TTS Model", DEFAULT_TTS_MODEL)

    # Images
    st.subheader("Images")
    img_mode = st.selectbox("Add images?", ["None", "Upload", "Auto-generate"], index=0)
    img_placement = st.selectbox("Placement", ["Full width (banner)", "Inset (half width)"], index=0)
    img_style = st.selectbox("Style (auto)", ["flat illustration", "watercolor", "manga", "photo-real", "minimal icon"], index=0)
    img_seed_prompt = st.text_area("Image seed prompt", "An educational, friendly illustration that reinforces the story’s main scene.")
    max_img_width_px = st.slider("Max image width (px)", 600, 1600, 1000)

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

def compute_target_words(i: int, total: int, min_w: int, max_w: int, ramp: bool) -> int:
    if not ramp or total == 1:
        return (min_w + max_w) // 2
    # quadratic ease-in
    t = i / (total - 1)
    val = min_w + (max_w - min_w) * (t**1.4)
    return int(val)

def zip_bytes(mp3_map: Dict[str, bytes]) -> bytes:
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as z:
        for filename, content in mp3_map.items():
            z.writestr(filename, content)
    buf.seek(0)
    return buf.read()

if build_btn:
    subtopics = [s.strip() for s in subtopics_raw.split(",") if s.strip()]
    level = LEVELS[level_key]

    # 1) Build stories
    stories: List[Dict] = []
    client: Optional[OpenAI] = None

    if api_key:
        client = get_openai_client(api_key)

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

    # 2) Images
    image_paths: List[Optional[str]] = [None] * n_stories
    if img_mode == "Auto-generate":
        if not client:
            st.warning("Enter your OPENAI_API_KEY to auto-generate images.")
        else:
            with st.spinner("Generating images…"):
                for i, s in enumerate(stories):
                    try:
                        im = generate_image_from_story(client, s["title"], s["story"], img_style, img_seed_prompt, DEFAULT_IMAGE_MODEL)
                        im = resize_for_pdf(im, max_img_width_px)
                        image_paths[i] = save_temp_png(im)
                    except Exception as e:
                        st.error(f"Image generation failed for story {i+1}: {e}")

    elif img_mode == "Upload":
        st.subheader("Upload images per story")
        for i, s in enumerate(stories, 1):
            up = st.file_uploader(f"Story {i}: {s['title']}", type=["png", "jpg", "jpeg"], key=f"up_{i}")
            if up:
                try:
                    im = Image.open(up).convert("RGB")
                    im = resize_for_pdf(im, max_img_width_px)
                    image_paths[i-1] = save_temp_png(im)
                except Exception as e:
                    st.error(f"Could not process image for story {i}: {e}")

    # 3) Render PDF
    with st.spinner("Rendering PDF…"):
        book_title = f"{lang} • {level_key} • {topic}"
        pdf_bytes = render_pdf(book_title, lang, level_key, stories, show_romanization, image_paths, img_placement)
    st.success("Text & PDF ready.")
    st.download_button("📕 Download PDF", data=pdf_bytes, file_name="graded_reader.pdf", mime="application/pdf")

    # 4) TTS (MP3)
    mp3_files: Dict[str, bytes] = {}
    if not client:
        st.warning("Enter your OPENAI_API_KEY to enable MP3 generation.")
    else:
        with st.spinner("Synthesizing audio…"):
            for i, s in enumerate(stories, 1):
                cn_text = " ".join([ln.get("cn","") for ln in s["story"] if ln.get("cn")])
                try:
                    audio_bytes = synthesize_tts_mp3(client, cn_text, voice=voice, model=DEFAULT_TTS_MODEL if not st.session_state.get("tts_model") else st.session_state["tts_model"], fmt=DEFAULT_AUDIO_FORMAT)
                except Exception:
                    # Use text input box model if provided
                    audio_bytes = synthesize_tts_mp3(client, cn_text, voice=voice, model=tts_model, fmt=DEFAULT_AUDIO_FORMAT)
                mp3_files[f"story_{i:02d}_normal.mp3"] = audio_bytes

                if slow_audio:
                    slow_text = "（慢速朗读） " + cn_text if "Chinese" in lang or "HSK" in level_key else "(slow reading) " + cn_text
                    try:
                        audio_bytes_slow = synthesize_tts_mp3(client, slow_text, voice=voice, model=tts_model, fmt=DEFAULT_AUDIO_FORMAT)
                        mp3_files[f"story_{i:02d}_slow.mp3"] = audio_bytes_slow
                    except Exception as e:
                        st.error(f"TTS slow version failed for story {i}: {e}")

        if mp3_files:
            zbytes = zip_bytes(mp3_files)
            st.download_button("🎧 Download MP3 ZIP", data=zbytes, file_name="audio_stories.zip", mime="application/zip")

    # 5) Vocab export (stub)
    vocab_payload = {"note": "Plug in real vocab extraction if using a lexicon.", "stories": len(stories)}
    st.download_button("🔤 Export Vocabulary (JSON)", data=json.dumps(vocab_payload, ensure_ascii=False, indent=2).encode("utf-8"),
                       file_name="vocab.json", mime="application/json")

st.markdown("""
**Tips**
- Turn on *Use OpenAI to generate stories* to get fresh content per run.
- For Chinese HSK1–2, keep romanization ON; for higher levels, consider turning it OFF.
- Use the difficulty ramp to gradually increase words/story across the book.
""")
