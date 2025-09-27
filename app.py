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
# ======= CONFIG ==========
# =========================

DEFAULT_TTS_MODEL = os.getenv("OPENAI_TTS_MODEL", "gpt-4o-mini-tts")
DEFAULT_TTS_VOICE = os.getenv("OPENAI_TTS_VOICE", "alloy")
DEFAULT_TEXT_MODEL = os.getenv("OPENAI_TEXT_MODEL", "gpt-4o-mini")

FONT_DIR = os.getenv("FONT_DIR", "fonts")
FONT_LATIN_PATH = os.getenv("FONT_PATH_LATIN", os.path.join(FONT_DIR, "DejaVuSans.ttf"))
FONT_CJK_PATH = os.getenv("FONT_PATH_CJK", os.path.join(FONT_DIR, "NotoSansSC-Regular.ttf"))
if not os.path.exists(FONT_CJK_PATH):
    # some users have the OTF
    FONT_CJK_PATH = os.path.join(FONT_DIR, "NotoSansSC-Regular.otf")

PDF_FONT_NAME = "AppSans"
MAX_STORIES = 20

# =========================
# ======= LEVELS ==========
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
}

# =========================
# ======= DATA ============
# =========================

VOCAB_MAPPING: Dict[str, Tuple[str, str]] = {
    "你好": ("nǐ hǎo", "hello"),
    "名字": ("míngzi", "name"),
    "朋友": ("péngyou", "friend"),
    "老师": ("lǎoshī", "teacher"),
    "学生": ("xuéshēng", "student"),
    "学校": ("xuéxiào", "school"),
    "咖啡": ("kāfēi", "coffee"),
    "茶": ("chá", "tea"),
    "米饭": ("mǐfàn", "rice"),
    "苹果": ("píngguǒ", "apple"),
    "公园": ("gōngyuán", "park"),
    "聊天": ("liáotiān", "to chat"),
    "今天": ("jīntiān", "today"),
    "明天": ("míngtiān", "tomorrow"),
    "天气": ("tiānqì", "weather"),
    "雨伞": ("yǔsǎn", "umbrella"),
    "再见": ("zàijiàn", "goodbye"),
    "谢谢": ("xièxie", "thanks"),
    "请坐": ("qǐng zuò", "please sit"),
}

GRAMMAR_PATTERNS = [
    {"point": "“吗”构成一般疑问句。", "examples": ["你喜欢茶吗？", "你是学生吗？"]},
    {"point": "“一边 … 一边 …”表示两个动作同时进行。", "examples": ["我一边听音乐一边做饭。", "他一边看书一边喝咖啡。"]},
    {"point": "“请 + 动词”用于礼貌请求。", "examples": ["请坐。", "请你告诉我。"]},
    {"point": "副词“很”常见于初级句子中。", "examples": ["今天天气很好。", "她很高兴。"]},
]

BASIC_NAMES = [
    ("王明", "Wáng Míng"),
    ("小李", "Xiǎo Lǐ"),
    ("王芳", "Wáng Fāng"),
    ("小王", "Xiǎo Wáng"),
]

BASIC_PLACES = ["学校", "公园", "图书馆", "超市", "家里"]

BASIC_LINES_POOL = [
    {"cn": "你好！", "romanization": "Nǐ hǎo!", "en": "Hello!"},
    {"cn": "今天天气很好。", "romanization": "Jīntiān tiānqì hěn hǎo.", "en": "The weather is nice today."},
    {"cn": "我们在学校。", "romanization": "Wǒmen zài xuéxiào.", "en": "We are at school."},
    {"cn": "我们在公园聊天。", "romanization": "Wǒmen zài gōngyuán liáotiān.", "en": "We chat in the park."},
    {"cn": "我喜欢喝茶。", "romanization": "Wǒ xǐhuān hē chá.", "en": "I like to drink tea."},
    {"cn": "我不喜欢咖啡。", "romanization": "Wǒ bù xǐhuān kāfēi.", "en": "I don’t like coffee."},
    {"cn": "请坐。", "romanization": "Qǐng zuò.", "en": "Please sit."},
    {"cn": "谢谢！", "romanization": "Xièxie!", "en": "Thanks!"},
    {"cn": "再见！", "romanization": "Zàijiàn!", "en": "Goodbye!"},
]

HANZI_RE = re.compile(r"[\u4e00-\u9fff]")

# =========================
# ====== UTILITIES ========
# =========================

def has_hanzi(s: str) -> bool:
    return bool(HANZI_RE.search(s or ""))

def cn_len(s: str) -> int:
    # count hanzi; fallback to words for non-CJK
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

def compute_target(i: int, total: int, min_w: int, max_w: int, ramp: bool) -> int:
    if not ramp or total <= 1: return (min_w + max_w) // 2
    t = i / (total - 1)
    return int(min_w + (max_w - min_w) * (t ** 1.4))

# =========================
# === STRUCTURED STORIES ==
# =========================

def structured_hsk1_story(topic: str, target_len: int, show_pinyin: bool, seed: int) -> Dict:
    """Coherent, simple story: intro → mini-problem → resolution (+ 2–4 lines dialogue)."""
    rnd = random.Random(seed)
    (name_a_hz, name_a_py) = rnd.choice(BASIC_NAMES)
    (name_b_hz, name_b_py) = rnd.choice([n for n in BASIC_NAMES if n != (name_a_hz, name_a_py)])
    place = rnd.choice(BASIC_PLACES)

    def l(cn, py, en):
        d = {"cn": cn, "en": en}
        if show_pinyin: d["romanization"] = py
        return d

    story: List[Dict[str, str]] = []
    # Intro (2–3 short lines)
    story.append(l(f"{name_a_hz}在{place}。", f"{name_a_py} zài {place}。", f"{name_a_py} is at the {place.lower()}." if place!="家里" else f"{name_a_py} is at home."))
    story.append(l(f"{name_b_hz}来了。", f"{name_b_py} lái le.", f"{name_b_py} arrives."))
    # Dialogue setup
    story.append(l("“你今天怎么样？”", "“Nǐ jīntiān zěnmeyàng?”", "“How are you today?”"))
    story.append(l("“我很好。你呢？”", "“Wǒ hěn hǎo. Nǐ ne?”", "“I’m fine. And you?”"))
    # Mini problem based on topic
    problem_cn = f"{name_b_hz}忘了带雨伞。"
    problem_py = f"{name_b_py} wàng le dài yǔsǎn."
    problem_en = f"{name_b_py} forgot an umbrella."
    if "吃" in topic or "饭" in topic:
        problem_cn = f"{name_b_hz}没带钱。"
        problem_py = f"{name_b_py} méi dài qián."
        problem_en = f"{name_b_py} didn’t bring money."
    story.append(l(problem_cn, problem_py, problem_en))
    # Resolution with one more dialogue
    story.append(l("“没关系，我有。”", "“Méi guānxi, wǒ yǒu.”", "“No problem, I have one.”"))
    story.append(l("他们一起走。", "Tāmen yīqǐ zǒu.", "They walk together."))
    # Fill to target length with safe lines
    rnd.shuffle(BASIC_LINES_POOL)
    total = sum(cn_len(x["cn"]) for x in story)
    for base in BASIC_LINES_POOL:
        if total >= target_len: break
        story.append({k: base[k] for k in (["cn","en","romanization"] if show_pinyin else ["cn","en"])})
        total += cn_len(base["cn"])

    # Glossary extraction (from mapping if present)
    glossary, seen = [], set()
    for ln in story:
        cn_text = ln["cn"]
        for term, (py, en) in VOCAB_MAPPING.items():
            if term in cn_text and term not in seen:
                g = {"term": term, "en": en}
                if show_pinyin: g["romanization"] = py
                glossary.append(g); seen.add(term)
                if len(glossary) >= 8: break
        if len(glossary) >= 8: break

    # Grammar + simple comprehension
    grammar = random.choice(GRAMMAR_PATTERNS)
    ref = story[0]["cn"].replace("。", "")
    questions = [{"type":"tf", "q": f"（T/F）{ref}", "answer":"T"}]

    return {"title": topic, "story": story, "glossary": glossary, "grammar_note": grammar, "questions": questions}

# =========================
# ====== OPENAI GEN =======
# =========================

def build_prompt(lang: str, level: LevelProfile, romanization_on: bool, topic: str, subtopics: List[str], target: int) -> str:
    ro = "ON" if romanization_on else "OFF"
    subs = ", ".join(subtopics) if subtopics else "None"
    return f"""
You are an expert graded-reader writer. Produce ONE coherent HSK1 story as JSON.

LANGUAGE: {lang}
LEVEL: {level.name}
ROMANIZATION: {ro}
TOPIC: {topic}
SUBTOPICS: {subs}
TARGET_CHARS: {target} (±10%)

STRICT JSON SCHEMA:
{{
  "title": "string",
  "story": [{{"cn": "Hanzi sentence", "romanization": "pinyin (omit field if ROMANIZATION=OFF)", "en":"English"}}],
  "glossary": [{{"term":"Hanzi word from story","romanization":"pinyin (omit if OFF)","pos":"","en":"meaning"}}],
  "grammar_note": {{"point":"short pattern","examples":["ex1","ex2"]}},
  "questions": [{{"type":"tf","q":"T/F question","answer":"T"}}]
}}

STYLE & CONTENT (HSK1 best practice):
- Coherent mini-plot: scene setup → tiny problem → resolution; include 2–4 lines of dialogue in quotes.
- Short sentences, mostly 6–14 Hanzi; use high-frequency HSK1 vocabulary.
- If ROMANIZATION=OFF, DO NOT include the 'romanization' field at all.
- Glossary: 6–10 items that actually appear in the story.
- Return JSON ONLY.
"""

def validate_story_payload(data: dict, min_hanzi_lines: int = 6) -> bool:
    if not isinstance(data, dict): return False
    lines = data.get("story", [])
    if not isinstance(lines, list) or not lines: return False
    hanzi_count = sum(1 for ln in lines if has_hanzi(ln.get("cn","")))
    return hanzi_count >= min_hanzi_lines

def try_openai_story(client: OpenAI, lang: str, level_key: str, topic: str, subtopics: List[str], target: int, romanization_on: bool, model: str) -> Optional[dict]:
    level = LEVELS[level_key]
    prompt = build_prompt(lang, level, romanization_on, topic, subtopics, target)
    try:
        resp = client.responses.create(
            model=model,
            input=[{"role":"system","content":prompt},{"role":"user","content":"Generate the JSON now."}],
            response_format={"type":"json_object"},
            temperature=0.4,
            presence_penalty=0.2,
            frequency_penalty=0.2,
        )
        txt = getattr(resp, "output_text", None)
        if not txt and hasattr(resp, "output") and resp.output and hasattr(resp.output[0], "content"):
            txt = "".join([c.text for c in resp.output[0].content if getattr(c, "type", "") == "output_text"])
        if not txt: return None
        data = json.loads(txt)
        if not romanization_on and "story" in data:
            for ln in data["story"]:
                ln.pop("romanization", None)
            for g in data.get("glossary", []):
                g.pop("romanization", None)
        return data if validate_story_payload(data) else None
    except Exception:
        return None

def generate_story(client: Optional[OpenAI], use_ai: bool, **kwargs) -> dict:
    if use_ai and client:
        data = try_openai_story(client, **kwargs)
        if data: return data
    # fallback structured placeholder with deterministic seed for variety
    seed = random.randint(1, 10_000)
    return structured_hsk1_story(kwargs["topic"], kwargs["target"], kwargs["romanization_on"], seed)

# =========================
# ====== TTS / AUDIO ======
# =========================

def synthesize_tts_mp3(client: OpenAI, text: str, voice: str, model: str) -> bytes:
    resp = client.audio.speech.create(model=model, voice=voice, input=text)
    if isinstance(resp, (bytes, bytearray)): return bytes(resp)
    if hasattr(resp, "read"): return resp.read()
    content = getattr(resp, "content", None)
    if isinstance(content, (bytes, bytearray)): return bytes(content)
    try:
        b64 = getattr(resp, "audio", {}).get("data", None)
        if b64: return base64.b64decode(b64)
    except Exception:  # noqa
        pass
    if isinstance(resp, dict):
        if "data" in resp and isinstance(resp["data"], (bytes, bytearray)):
            return bytes(resp["data"])
        audio = resp.get("audio")
        if isinstance(audio, dict) and isinstance(audio.get("data"), str):
            return base64.b64decode(audio["data"])
    raise RuntimeError("Unexpected TTS response shape.")

def make_zip(files: Dict[str, bytes]) -> bytes:
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as z:
        for name, data in files.items():
            z.writestr(name, data)
    buf.seek(0)
    return buf.read()

# =========================
# ======== PDF ============
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
        pdf.add_font(PDF_FONT_NAME, "", FONT_CJK_PATH, uni=True)
        pdf.add_font(PDF_FONT_NAME, "B", FONT_CJK_PATH, uni=True)
        pdf.add_font(PDF_FONT_NAME, "I", FONT_CJK_PATH, uni=True)
        pdf._has_cjk, added = True, True
    if not added and os.path.exists(FONT_LATIN_PATH):
        pdf.add_font(PDF_FONT_NAME, "", FONT_LATIN_PATH, uni=True)
        pdf.add_font(PDF_FONT_NAME, "B", FONT_LATIN_PATH, uni=True)
        pdf.add_font(PDF_FONT_NAME, "I", FONT_LATIN_PATH, uni=True)
        added = True
    if not added:
        raise RuntimeError("No Unicode font found. Add NotoSansSC-Regular.ttf/otf and DejaVuSans.ttf under fonts/.")

def intro_page(pdf: FPDF):
    pdf.add_page()
    pdf.set_font(PDF_FONT_NAME, "B", 18)
    pdf.cell(0, 12, "Introduction", ln=1)
    pdf.ln(2)
    pdf.set_font(PDF_FONT_NAME, "", 12)
    text = (
        "This graded reader uses high-frequency words and simple patterns.\n\n"
        "How to use it:\n"
        "1) Listen first without reading. Try to get the gist.\n"
        "2) Read once quickly; don’t stop for every new word.\n"
        "3) Read again carefully; use pinyin/translation only when needed.\n"
        "4) Review often. Repetition builds fluency.\n\n"
        "Vocabulary lists highlight key words. Grammar notes point out useful patterns."
    )
    for p in text.split("\n\n"):
        mc_full_width(pdf, p, 6); pdf.ln(2)

def render_pdf(book_title: str, lang: str, level_key: str, stories: List[Dict], show_pinyin: bool) -> bytes:
    pdf = ReaderPDF()
    register_fonts(pdf)
    needs_cjk = ("Chinese" in lang or "HSK" in level_key)
    pdf.book_title = f"{book_title}{'' if (not needs_cjk or pdf._has_cjk) else '  [CJK font missing]'}"
    pdf.set_auto_page_break(auto=True, margin=15)

    # Title
    pdf.add_page()
    pdf.set_font(PDF_FONT_NAME, "B", 22)
    pdf.cell(0, 16, book_title, ln=1)
    pdf.set_font(PDF_FONT_NAME, "", 12)
    mc_full_width(pdf, f"Language: {lang} • Level: {level_key}", 8)
    pdf.ln(6)
    mc_full_width(pdf, "Compiled with Graded Reader Builder", 6)

    # Intro
    intro_page(pdf)

    # Contents placeholder
    pdf.add_page()
    toc_page, toc_y = pdf.page_no(), pdf.get_y()
    pdf.set_font(PDF_FONT_NAME, "B", 16); pdf.cell(0, 12, "Contents", ln=1)
    pdf.set_font(PDF_FONT_NAME, "", 12); pdf.ln(2)
    toc_entries: List[Tuple[str,int]] = []

    # Chapters
    for i, s in enumerate(stories, 1):
        toc_entries.append((f"{i}. {s['title']}", pdf.page_no()+1))
        pdf.add_page()
        pdf.set_font(PDF_FONT_NAME, "B", 16); pdf.cell(0, 10, f"{i}. {s['title']}", ln=1)
        pdf.ln(2)
        pdf.set_font(PDF_FONT_NAME, "", 12)
        for ln in s["story"]:
            mc_full_width(pdf, ln.get("cn",""), 7)
            if show_pinyin and ln.get("romanization"):
                mc_full_width(pdf, ln["romanization"], 6, (100,100,100))
            if ln.get("en"):
                mc_full_width(pdf, ln["en"], 6, (80,80,80))
            pdf.ln(1)
        # Vocab
        if s.get("glossary"):
            pdf.ln(2); pdf.set_font(PDF_FONT_NAME, "B", 13); pdf.cell(0,9,"Vocabulary", ln=1)
            pdf.set_font(PDF_FONT_NAME, "", 12)
            for g in s["glossary"]:
                line = g["term"]
                if show_pinyin and g.get("romanization"): line += f" [{g['romanization']}]"
                if g.get("en"): line += f": {g['en']}"
                mc_full_width(pdf, "- " + line, 6)
        # Grammar
        if s.get("grammar_note") and s["grammar_note"].get("point"):
            pdf.ln(2); pdf.set_font(PDF_FONT_NAME, "B", 13); pdf.cell(0,9,"Grammar note", ln=1)
            pdf.set_font(PDF_FONT_NAME, "", 12)
            mc_full_width(pdf, s["grammar_note"]["point"], 6)
            for ex in s["grammar_note"].get("examples", []):
                mc_full_width(pdf, "- " + ex, 6)
        # Questions
        if s.get("questions"):
            pdf.ln(2); pdf.set_font(PDF_FONT_NAME, "B", 13); pdf.cell(0,9,"Comprehension", ln=1)
            pdf.set_font(PDF_FONT_NAME, "", 12)
            for q in s["questions"]:
                mc_full_width(pdf, f"- {q.get('q','')}", 6)

    # Fill ToC
    pdf.page = toc_page; pdf.set_y(toc_y); pdf.set_font(PDF_FONT_NAME, "", 12)
    for title, pg in toc_entries:
        dots = "." * max(4, 64 - len(title))
        pdf.cell(0, 8, f"{title} {dots} {pg}", ln=1)

    out = pdf.output(dest="S")
    return bytes(out)

# =========================
# ===== STREAMLIT UI ======
# =========================

st.set_page_config(page_title="Graded Reader Builder (PDF + MP3)", page_icon="📘", layout="wide")
st.title("📘 Graded Reader Builder")
st.caption("HSK-style graded readers with coherent mini-plots, audio, vocab, and ToC.")

def font_status():
    ok_cjk = os.path.exists(FONT_CJK_PATH)
    ok_lat = os.path.exists(FONT_LATIN_PATH)
    st.caption(
        f"Font check — CJK: {'✅' if ok_cjk else '❌'} ({FONT_CJK_PATH}) | "
        f"Latin: {'✅' if ok_lat else '❌'} ({FONT_LATIN_PATH})"
    )
font_status()

# keep artifacts across reruns so downloads never disappear
if "artifacts" not in st.session_state:
    st.session_state["artifacts"] = {
        "stories": None, "pdf": None, "mp3zip": None, "vocab_json": None, "summary": None
    }

with st.sidebar:
    st.header("Book Settings")
    lang = st.selectbox("Language", ["Chinese (Simplified)"], index=0)
    level_key = st.selectbox("Level", list(LEVELS.keys()), index=0)
    topic = st.text_input("Topic/Area", "Daily routine")
    subtopics_raw = st.text_input("Subtopics (comma-separated)", "friends, weekend, park")
    n_stories = st.slider("Number of stories", 1, MAX_STORIES, 5)

    st.subheader("Story Length (≈ characters)")
    min_len = st.number_input("Min per story", 50, 2000, 120)
    max_len = st.number_input("Max per story", 50, 4000, 220)
    ramp = st.toggle("Difficulty ramp", True)

    show_pinyin = st.toggle("Show pinyin", value=("HSK1" in level_key or "HSK2" in level_key))
    slow_audio = st.toggle("Add slow audio", False)

    st.subheader("Generation")
    use_ai = st.toggle("Use OpenAI to write stories", True)
    text_model = st.text_input("Text model", DEFAULT_TEXT_MODEL)
    retries = st.slider("LLM retries if invalid", 1, 5, 2)
    temperature = st.slider("Creativity (AI mode)", 0.0, 1.0, 0.4, 0.1)

    st.subheader("Audio (TTS)")
    voice = st.text_input("TTS voice", DEFAULT_TTS_VOICE)
    tts_model = st.text_input("TTS model", DEFAULT_TTS_MODEL)

    st.divider()
    api_key = st.text_input("OPENAI_API_KEY", type="password")
    build_btn = st.button("Generate Book", use_container_width=True)

@st.cache_resource(show_spinner=False)
def get_openai_client(api_key: str):
    if not api_key: return None
    os.environ["OPENAI_API_KEY"] = api_key
    return OpenAI()

def make_artifacts(lang, level_key, topic, subtopics, n_stories, min_len, max_len, ramp, show_pinyin,
                   use_ai, text_model, retries, voice, tts_model, slow_audio, api_key):
    client = get_openai_client(api_key) if api_key else None
    stories = []
    for i in range(n_stories):
        target = compute_target(i, n_stories, min_len, max_len, ramp)
        data = None
        if use_ai and client:
            for _ in range(retries):
                data = try_openai_story(client, lang, level_key, topic, subtopics, target, show_pinyin, text_model)
                if data: break
        if not data:
            data = structured_hsk1_story(topic, target, show_pinyin, seed=random.randint(1, 10_000))
        # rotate grammar if missing
        if not data.get("grammar_note") or not data["grammar_note"].get("point"):
            data["grammar_note"] = GRAMMAR_PATTERNS[i % len(GRAMMAR_PATTERNS)]
        data["title"] = f"{topic} ({i+1})"
        stories.append(data)

    # PDF
    book_title = f"{lang} - {level_key} - {topic}"
    pdf_bytes = render_pdf(book_title, lang, level_key, stories, show_pinyin)

    # Audio (optional)
    mp3_map = {}
    if api_key:
        try:
            for i, s in enumerate(stories, 1):
                cn_text = " ".join([ln.get("cn","") for ln in s["story"] if ln.get("cn")]).strip()
                if not cn_text: continue
                normal = synthesize_tts_mp3(client, cn_text, voice, tts_model)
                mp3_map[f"story_{i:02d}_normal.mp3"] = normal
                if slow_audio:
                    prefix = "（慢速朗读） "
                    slow = synthesize_tts_mp3(client, prefix + cn_text, voice, tts_model)
                    mp3_map[f"story_{i:02d}_slow.mp3"] = slow
        except Exception as e:
            st.warning(f"Audio generation issue: {e}")
    mp3zip = make_zip(mp3_map) if mp3_map else None

    # Vocab export
    vocab = {}
    for s in stories:
        for g in s.get("glossary", []):
            vocab[g["term"]] = {"pinyin": g.get("romanization",""), "meaning": g.get("en","")}
    vocab_json = json.dumps(vocab, ensure_ascii=False, indent=2).encode("utf-8") if vocab else None

    return stories, pdf_bytes, mp3zip, vocab_json

if build_btn:
    subs = [s.strip() for s in subtopics_raw.split(",") if s.strip()]
    stories, pdf_bytes, mp3zip, vocab_json = make_artifacts(
        lang, level_key, topic, subs, n_stories, min_len, max_len, ramp, show_pinyin,
        use_ai, text_model, retries, voice, tts_model, slow_audio, api_key
    )
    st.session_state["artifacts"].update({
        "stories": stories, "pdf": pdf_bytes, "mp3zip": mp3zip, "vocab_json": vocab_json,
        "summary": {"lang":lang,"level":level_key,"topic":topic,"n":n_stories}
    })

# -------- Presentation: tabs keep downloads visible after clicks --------
a = st.session_state["artifacts"]
tabs = st.tabs(["📖 Preview & Download", "🎧 Audio", "🔤 Vocabulary", "🧩 Contents"])

with tabs[0]:
    st.subheader("Preview & Download")
    if a["summary"]:
        st.write(f"**Book:** {a['summary']['lang']} — {a['summary']['level']} — {a['summary']['topic']}  |  Stories: {a['summary']['n']}")
    if a["pdf"]:
        st.download_button("📕 Download PDF", data=a["pdf"], file_name="graded_reader.pdf", mime="application/pdf", key="dl_pdf")
    else:
        st.info("Generate a book to enable downloads.")
with tabs[1]:
    st.subheader("Audio (ZIP)")
    if a["mp3zip"]:
        st.download_button("🎧 Download MP3 ZIP", data=a["mp3zip"], file_name="audio_stories.zip", mime="application/zip", key="dl_zip")
    else:
        st.info("Add an API key and regenerate to get MP3s.")
with tabs[2]:
    st.subheader("Vocabulary JSON")
    if a["vocab_json"]:
        st.download_button("🔤 Download vocab.json", data=a["vocab_json"], file_name="vocab.json", mime="application/json", key="dl_vocab")
    else:
        st.info("Generate a book to export vocabulary.")
with tabs[3]:
    st.subheader("Table of Contents")
    if a["stories"]:
        for i, s in enumerate(a["stories"], 1):
            st.markdown(f"**{i}. {s['title']}**")
            # brief first three lines as a preview
            preview = s["story"][:3]
            for ln in preview:
                st.write(ln.get("cn",""))
    else:
        st.info("Generate a book to see the contents here.")
