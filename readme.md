# Graded Reader Builder (PDF + MP3 + Images)

Create **graded language readers** from **beginner to advanced** (HSK1–HSK6, CEFR A1–C1) with:
- User-defined **story length** (min/max words) + optional **difficulty ramp**
- **Audio (MP3)** per story via OpenAI TTS
- Optional **images**: upload or **auto-generate** using OpenAI Images
- Clean **PDF** export (Unicode-capable fonts)

## Quick start
```bash
git clone <your-repo-url>
cd <repo>
python -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
cp .env.example .env        # add your OPENAI_API_KEY
