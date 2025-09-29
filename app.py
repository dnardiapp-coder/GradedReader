"""Streamlit entrypoint for the Graded Reader production application."""
from __future__ import annotations

import io
import os
import re
import zipfile
from typing import Optional

import streamlit as st
from openai import OpenAI

from graded_reader import config
from graded_reader.audio import AudioGenerator
from graded_reader.data import LANGUAGE_CONFIGS, LEVEL_PROFILES, STORY_STRUCTURES
from graded_reader.generation import StoryGenerator
from graded_reader.models import PedagogicalFeatures, StoryPackage
from graded_reader.pdf_builder import PDFBuilder


def _get_openai_client() -> Optional[OpenAI]:
    """Return an OpenAI client if credentials are available."""

    api_key = os.getenv("OPENAI_API_KEY", "")

    # Streamlit secrets may store the key at the root or within an "openai" mapping.
    if not api_key:
        api_key = st.secrets.get("OPENAI_API_KEY", "")
    if not api_key and "openai" in st.secrets:
        api_key = st.secrets["openai"].get("api_key", "")

    if not api_key:
        return None
    return OpenAI(api_key=api_key)


def _render_story(story: StoryPackage, *, heading_level: str = "header") -> None:
    heading_func = getattr(st, heading_level, st.header)
    heading_func(story.title)
    if story.title_translated:
        st.caption(story.title_translated)

    meta_cols = st.columns(3)
    meta_cols[0].metric("Level", story.level.value)
    meta_cols[1].metric("Difficulty", f"{story.difficulty_score:.2f}")
    meta_cols[2].metric("Reading time", f"{story.estimated_reading_time} min")

    if story.summary:
        st.subheader("Summary")
        st.write(story.summary)

    st.subheader("Story")
    for paragraph in story.story:
        with st.expander(f"Paragraph {paragraph.paragraph_id}"):
            st.write(paragraph.text)
            if paragraph.romanization:
                st.markdown(f"**{LANGUAGE_CONFIGS[story.language].romanization_name}:** {paragraph.romanization}")
            if paragraph.translation:
                st.markdown(f"**English:** {paragraph.translation}")

    if story.vocabulary:
        st.subheader("Key Vocabulary")
        for item in story.vocabulary:
            components = [item.term]
            if item.romanization:
                components.append(f"({item.romanization})")
            if item.translation:
                components.append(f"– {item.translation}")
            st.markdown(" ".join(components))
            if item.example:
                st.caption(item.example)

    if story.grammar_points:
        st.subheader("Grammar Focus")
        for point in story.grammar_points:
            st.markdown(f"**{point.structure}**")
            st.write(point.explanation)
            if point.examples:
                st.caption("Examples: " + " | ".join(point.examples))
            if point.practice:
                st.caption(f"Practice: {point.practice}")

    if story.comprehension_questions:
        st.subheader("Comprehension Questions")
        for idx, question in enumerate(story.comprehension_questions, 1):
            st.markdown(f"{idx}. {question.question}")
            if question.question_english:
                st.caption(question.question_english)
            if question.options:
                st.write("Options: " + ", ".join(question.options))
            if question.answer:
                st.caption(f"Answer: {question.answer}")

    if story.cultural_notes:
        st.subheader("Cultural Notes")
        for note in story.cultural_notes:
            st.write(f"- {note.get('topic', '')}: {note.get('explanation', '')}")

    if story.discussion_prompts:
        st.subheader("Discussion Prompts")
        for prompt in story.discussion_prompts:
            st.write(f"- {prompt}")

    if story.writing_tasks:
        st.subheader("Writing Tasks")
        for task in story.writing_tasks:
            st.write(f"- {task.get('prompt', '')}")
            if task.get("word_limit"):
                st.caption(f"Word limit: {task['word_limit']}")
            if task.get("focus"):
                st.caption(f"Focus: {task['focus']}")


def _prepare_audio_zip(audio_files: dict) -> Optional[bytes]:
    if not audio_files:
        return None
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w", zipfile.ZIP_DEFLATED) as archive:
        for filename, payload in audio_files.items():
            archive.writestr(filename, payload)
    buffer.seek(0)
    return buffer.read()


def main() -> None:
    st.set_page_config(page_title="Graded Reader Studio", layout="wide")
    st.title("Graded Reader Studio")
    st.write(
        "Create pedagogically rich graded readers complete with printable PDFs and audio narration."
    )

    if "generated_stories" not in st.session_state:
        st.session_state.generated_stories = []
    if "generation_error" not in st.session_state:
        st.session_state.generation_error = None

    with st.sidebar:
        st.header("Reader Settings")
        language = st.selectbox(
            "Target language",
            list(config.Language),
            format_func=lambda lang: LANGUAGE_CONFIGS[lang].name,
        )
        level = st.selectbox(
            "Level",
            list(config.ProficiencyLevel),
            format_func=lambda lvl: LEVEL_PROFILES[lvl].name,
        )
        length_choice = st.selectbox(
            "Story length",
            list(config.StoryLength),
            format_func=lambda length: length.label,
        )

        available_topics = LEVEL_PROFILES[level].themes
        topic = st.text_input(
            "Theme or topic",
            value=available_topics[0] if available_topics else "",
            help="Describe the main focus for the story in your own words.",
        ).strip()
        subtopic_options = st.multiselect(
            "Additional themes",
            available_topics,
            default=available_topics[:2],
        )

        custom_subtopic_text = st.text_area(
            "Custom themes",
            placeholder="Enter additional themes, separated by commas or new lines.",
            help="Use this to guide the AI beyond the suggested themes.",
        )
        custom_subtopics = [
            item.strip()
            for item in re.split(r"[\n,]", custom_subtopic_text)
            if item.strip()
        ]
        subtopics_combined = list(dict.fromkeys(subtopic_options + custom_subtopics))

        structure_options = [
            key for key, structure in STORY_STRUCTURES.items() if level in structure.suitable_levels
        ] or list(STORY_STRUCTURES.keys())
        structure_key = st.selectbox(
            "Narrative structure",
            structure_options,
            format_func=lambda key: STORY_STRUCTURES[key].name,
        )

        st.header("Pedagogical Extras")
        features = PedagogicalFeatures(
            vocabulary_preview=st.checkbox("Vocabulary list", value=True),
            grammar_focus=st.checkbox("Grammar focus", value=True),
            comprehension_questions=st.checkbox("Comprehension questions", value=True),
            cultural_notes=st.checkbox("Cultural notes", value=False),
            discussion_prompts=st.checkbox("Discussion prompts", value=False),
            writing_exercises=st.checkbox("Writing tasks", value=False),
        )

        voice = st.text_input("Voice (TTS)", value=config.DEFAULT_TTS_VOICE)
        speech_speed = st.slider("Narration speed", 0.8, 1.2, 1.0, 0.05)
        temperature = st.slider("Creativity", 0.0, 1.0, 0.7, 0.05)
        story_count = st.slider(
            "Number of stories",
            1,
            3,
            1,
            help="Generate multiple variants with the same settings.",
        )

    if st.button("Generate reader", type="primary"):
        st.session_state.generated_stories = []
        st.session_state.generation_error = None

        with st.spinner("Crafting your graded reader(s)..."):
            try:
                client = _get_openai_client()
                story_generator = StoryGenerator(client)
                pdf_builder = PDFBuilder(language)
                audio_generator = AudioGenerator(client, voice=voice)

                stories = []
                for _ in range(story_count):
                    story = story_generator.generate_story(
                        language=language,
                        level=level,
                        topic=topic or (available_topics[0] if available_topics else "General interest"),
                        subtopics=subtopics_combined,
                        structure_key=structure_key,
                        story_length=length_choice.target_words,
                        features=features,
                        temperature=temperature,
                    )

                    pdf_bytes = pdf_builder.build(story)
                    audio_files = audio_generator.generate_audio_bundle(story, speed=speech_speed)
                    audio_zip = _prepare_audio_zip(audio_files)

                    stories.append(
                        {
                            "story": story,
                            "pdf_bytes": pdf_bytes,
                            "audio_zip": audio_zip,
                        }
                    )

                st.session_state.generated_stories = stories
            except Exception as exc:
                st.session_state.generation_error = str(exc)

    if st.session_state.generation_error:
        st.error(f"Generation failed: {st.session_state.generation_error}")

    if st.session_state.generated_stories:
        if len(st.session_state.generated_stories) == 1:
            st.success("Reader ready!")
        else:
            st.success("Readers ready! Explore each version below.")

        for idx, package in enumerate(st.session_state.generated_stories, start=1):
            if idx > 1:
                st.divider()

            heading_level = "header" if idx == 1 else "subheader"
            _render_story(package["story"], heading_level=heading_level)

            st.download_button(
                "Download PDF",
                data=package["pdf_bytes"],
                file_name=f"{package['story'].title}_graded_reader.pdf",
                mime="application/pdf",
                key=f"pdf-download-{idx}",
            )

            if package["audio_zip"]:
                st.download_button(
                    "Download audio bundle",
                    data=package["audio_zip"],
                    file_name=f"{package['story'].title}_audio.zip",
                    mime="application/zip",
                    key=f"audio-download-{idx}",
                )
            elif idx == 1:
                st.info("Provide an OpenAI API key to enable audio narration exports.")

    st.caption(
        "Tip: set the OPENAI_API_KEY environment variable or add it to Streamlit secrets to unlock full AI generation."
    )


if __name__ == "__main__":
    main()
