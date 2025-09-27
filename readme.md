# Graded Reader Builder (PDF + MP3 + Images)

Create **graded language readers** from **beginner to advanced** (HSK1–HSK6 and CEFR A1–C1), with:
- User-defined **story length** (words/story) and optional **difficulty ramp**
- **Audio (MP3)** per story via OpenAI TTS (`gpt-4o-mini-tts` / `tts-1`)
- Optional **images**: upload per story or **auto-generate** from prompts (OpenAI Images)
- Clean **PDF** export (title page, table of contents, story → vocab → grammar note → comprehension)

## Quick start
```bash
git clone <your-repo-url>
cd <repo>
python -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
cp .env.example .env        # put your OPENAI_API_KEY here
streamlit run app.py
