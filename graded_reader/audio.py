"""Audio generation utilities for graded readers."""
from __future__ import annotations

from typing import Dict, Optional

from openai import OpenAI

from . import config
from .models import StoryPackage


class AudioGenerator:
    """Generate narration audio using OpenAI's TTS models."""

    def __init__(self, client: Optional[OpenAI], model: str = config.DEFAULT_TTS_MODEL, voice: str = config.DEFAULT_TTS_VOICE):
        self.client = client
        self.model = model
        self.voice = voice

    def generate_audio_bundle(self, story: StoryPackage, speed: float = 1.0) -> Dict[str, bytes]:
        """Create a single audio file narrating the entire story."""

        if not self.client:
            return {}

        full_text = story.primary_text.strip()
        if not full_text:
            return {}

        return {"story_full.mp3": self._synthesize_audio(full_text, speed=speed)}

    # ------------------------------------------------------------------
    def _synthesize_audio(self, text: str, speed: float = 1.0) -> bytes:
        if not self.client:
            raise RuntimeError("No OpenAI client configured")

        response = self.client.audio.speech.create(
            model=self.model,
            voice=self.voice,
            input=text,
            speed=speed,
        )

        if hasattr(response, "content"):
            return response.content
        if hasattr(response, "read"):
            return response.read()
        if isinstance(response, bytes):
            return response

        return bytes(response)
