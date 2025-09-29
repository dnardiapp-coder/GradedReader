"""PDF rendering layer for graded readers."""
from __future__ import annotations

import os
from typing import Callable

from fpdf import FPDF
from fpdf.enums import XPos

from . import config
from .data import LANGUAGE_CONFIGS
from .models import StoryPackage

MARGIN = 15
LINE_HEIGHT = 8
SECTION_SPACING = 4


class ReaderPDF(FPDF):
    """Customised FPDF with branded header/footer."""

    def __init__(self, title: str, subtitle: str):
        super().__init__(format="A4")
        self.title = title
        self.subtitle = subtitle
        self.set_auto_page_break(auto=True, margin=MARGIN)
        self.set_margins(MARGIN, MARGIN, MARGIN)

    def header(self) -> None:  # noqa: D401 - FPDF API
        self.set_font("Helvetica", "B", 14)
        self.set_text_color(30, 30, 30)
        self.cell(0, 10, self.title, ln=True, align="C")
        if self.subtitle:
            self.set_font("Helvetica", size=9)
            self.set_text_color(90, 90, 90)
            self.cell(0, 6, self.subtitle, ln=True, align="C")
        self.ln(4)

    def footer(self) -> None:  # noqa: D401 - FPDF API
        self.set_y(-15)
        self.set_font("Helvetica", size=8)
        self.set_text_color(140, 140, 140)
        self.cell(0, 10, f"Page {self.page_no()}", align="C")


class PDFBuilder:
    """Create production-ready PDFs from :class:`StoryPackage` data."""

    def __init__(self, language_code: config.Language):
        self.language_config = LANGUAGE_CONFIGS[language_code]
        self.primary_font = config.PDF_FONT_NAME
        self._has_custom_font = False

    # ------------------------------------------------------------------
    def build(self, story: StoryPackage) -> bytes:
        pdf = ReaderPDF(title=story.title, subtitle=story.title_translated or "")
        self._register_fonts(pdf)
        pdf.add_page()
        self._write_cover(pdf, story)
        self._write_summary(pdf, story)
        self._write_story(pdf, story)
        self._write_vocabulary(pdf, story)
        self._write_grammar(pdf, story)
        self._write_questions(pdf, story)
        self._write_extras(pdf, story)
        output = pdf.output(dest="S")
        if isinstance(output, str):
            return output.encode("latin1")
        return bytes(output)

    # ------------------------------------------------------------------
    def _register_fonts(self, pdf: FPDF) -> None:
        font_path = self.language_config.font_path
        if font_path and os.path.exists(font_path):
            pdf.add_font(self.primary_font, "", font_path, uni=True)
            self._has_custom_font = True
            pdf.set_font(self.primary_font, size=12)
        else:
            self.primary_font = "Helvetica"
            pdf.set_font(self.primary_font, size=12)

    def _set_font(self, pdf: FPDF, size: int, style: str = "") -> None:
        """Apply either the custom typeface or a safe fallback."""

        if self._has_custom_font and style == "":
            pdf.set_font(self.primary_font, style=style, size=size)
            return

        # Bold/italic variants are not guaranteed for custom fonts, so fall back
        # to built-in Helvetica when a styled variant is requested or the custom
        # font is unavailable.
        pdf.set_font("Helvetica", style=style, size=size)

    def _multi_cell(self, pdf: FPDF, w: float, h: float, text: str, **kwargs) -> None:
        """Wrapper around :meth:`FPDF.multi_cell` that resets X to the margin."""

        kwargs.setdefault("new_x", XPos.LMARGIN)
        pdf.multi_cell(w, h, text, **kwargs)

    def _write_section(self, pdf: ReaderPDF, title: str, body: Callable[[], None]) -> None:
        pdf.add_page()
        self._set_font(pdf, 14, "B")
        pdf.set_text_color(25, 25, 25)
        pdf.cell(0, LINE_HEIGHT, title, ln=True)
        pdf.ln(SECTION_SPACING)
        body()

    # ------------------------------------------------------------------
    def _write_cover(self, pdf: ReaderPDF, story: StoryPackage) -> None:
        self._set_font(pdf, 22, "B")
        pdf.set_text_color(15, 15, 15)
        self._multi_cell(pdf, 0, LINE_HEIGHT + 4, story.title, align="C")
        if story.title_translated:
            self._set_font(pdf, 12)
            pdf.set_text_color(90, 90, 90)
            self._multi_cell(pdf, 0, LINE_HEIGHT, story.title_translated, align="C")
        pdf.ln(SECTION_SPACING)

    def _write_summary(self, pdf: ReaderPDF, story: StoryPackage) -> None:
        self._set_font(pdf, 14, "B")
        pdf.set_text_color(25, 25, 25)
        pdf.cell(0, LINE_HEIGHT, "Overview", ln=True)
        self._set_font(pdf, 11)
        self._multi_cell(pdf, 0, LINE_HEIGHT, story.summary or "Summary unavailable.")
        pdf.ln(2)
        meta = (
            f"Level: {story.level.value}  |  Difficulty: {story.difficulty_score:.2f}  |  "
            f"Estimated reading time: {story.estimated_reading_time} min"
        )
        pdf.set_text_color(100, 100, 100)
        self._set_font(pdf, 9)
        pdf.cell(0, LINE_HEIGHT, meta, ln=True)
        pdf.ln(SECTION_SPACING)

    def _write_story(self, pdf: ReaderPDF, story: StoryPackage) -> None:
        self._set_font(pdf, 14, "B")
        pdf.set_text_color(25, 25, 25)
        pdf.cell(0, LINE_HEIGHT, "Story", ln=True)
        pdf.ln(SECTION_SPACING)
        for paragraph in story.story:
            self._set_font(pdf, 12)
            pdf.set_text_color(35, 35, 35)
            self._multi_cell(pdf, 0, LINE_HEIGHT, paragraph.text)
            if paragraph.romanization:
                self._set_font(pdf, 9)
                pdf.set_text_color(120, 120, 120)
                self._multi_cell(
                    pdf,
                    0,
                    LINE_HEIGHT - 1,
                    f"{self.language_config.romanization_name}: {paragraph.romanization}",
                )
            if paragraph.translation:
                self._set_font(pdf, 9)
                pdf.set_text_color(90, 90, 90)
                self._multi_cell(pdf, 0, LINE_HEIGHT - 1, f"EN: {paragraph.translation}")
            pdf.ln(1)
        pdf.ln(SECTION_SPACING)

    def _write_vocabulary(self, pdf: ReaderPDF, story: StoryPackage) -> None:
        if not story.vocabulary:
            return

        def body() -> None:
            self._set_font(pdf, 11)
            pdf.set_text_color(30, 30, 30)
            for entry in story.vocabulary:
                line = entry.term
                if entry.romanization:
                    line += f" ({entry.romanization})"
                if entry.translation:
                    line += f" – {entry.translation}"
                self._multi_cell(pdf, 0, LINE_HEIGHT, line)
                if entry.example:
                    self._set_font(pdf, 9)
                    pdf.set_text_color(120, 120, 120)
                    self._multi_cell(pdf, 0, LINE_HEIGHT - 1, f"Example: {entry.example}")
                    self._set_font(pdf, 11)
                    pdf.set_text_color(30, 30, 30)
                pdf.ln(1)

        self._write_section(pdf, "Key Vocabulary", body)

    def _write_grammar(self, pdf: ReaderPDF, story: StoryPackage) -> None:
        if not story.grammar_points:
            return

        def body() -> None:
            for point in story.grammar_points:
                self._set_font(pdf, 12, "B")
                pdf.set_text_color(25, 25, 25)
                self._multi_cell(pdf, 0, LINE_HEIGHT, point.structure)
                self._set_font(pdf, 11)
                self._multi_cell(pdf, 0, LINE_HEIGHT, point.explanation)
                for example in point.examples:
                    self._set_font(pdf, 9)
                    pdf.set_text_color(120, 120, 120)
                    self._multi_cell(pdf, 0, LINE_HEIGHT - 1, f"Example: {example}")
                if point.practice:
                    self._set_font(pdf, 9)
                    pdf.set_text_color(110, 110, 110)
                    self._multi_cell(pdf, 0, LINE_HEIGHT - 1, f"Try: {point.practice}")
                pdf.ln(1)

        self._write_section(pdf, "Grammar Focus", body)

    def _write_questions(self, pdf: ReaderPDF, story: StoryPackage) -> None:
        if not story.comprehension_questions:
            return

        def body() -> None:
            self._set_font(pdf, 11)
            for idx, question in enumerate(story.comprehension_questions, start=1):
                pdf.set_text_color(30, 30, 30)
                self._multi_cell(pdf, 0, LINE_HEIGHT, f"{idx}. {question.question}")
                if question.question_english:
                    self._set_font(pdf, 9)
                    pdf.set_text_color(120, 120, 120)
                    self._multi_cell(
                        pdf,
                        0,
                        LINE_HEIGHT - 1,
                        f"EN: {question.question_english}",
                    )
                    self._set_font(pdf, 11)
                if question.options:
                    self._set_font(pdf, 9)
                    pdf.set_text_color(90, 90, 90)
                    for option in question.options:
                        self._multi_cell(pdf, 0, LINE_HEIGHT - 1, f"- {option}")
                    self._set_font(pdf, 11)
                if question.answer:
                    self._set_font(pdf, 9)
                    pdf.set_text_color(70, 120, 70)
                    self._multi_cell(pdf, 0, LINE_HEIGHT - 1, f"Answer: {question.answer}")
                    self._set_font(pdf, 11)
                if question.explanation:
                    self._set_font(pdf, 9)
                    pdf.set_text_color(120, 120, 120)
                    self._multi_cell(pdf, 0, LINE_HEIGHT - 1, f"Why: {question.explanation}")
                    self._set_font(pdf, 11)
                pdf.ln(1)

        self._write_section(pdf, "Comprehension", body)

    def _write_extras(self, pdf: ReaderPDF, story: StoryPackage) -> None:
        if story.cultural_notes:
            def body_notes() -> None:
                self._set_font(pdf, 11)
                pdf.set_text_color(30, 30, 30)
                for note in story.cultural_notes:
                    topic = note.get("topic", "")
                    explanation = note.get("explanation", "")
                    self._multi_cell(pdf, 0, LINE_HEIGHT, f"{topic}: {explanation}".strip())
                    pdf.ln(1)

            self._write_section(pdf, "Cultural Notes", body_notes)

        if story.discussion_prompts:
            def body_discussion() -> None:
                self._set_font(pdf, 11)
                pdf.set_text_color(30, 30, 30)
                for prompt in story.discussion_prompts:
                    self._multi_cell(pdf, 0, LINE_HEIGHT, f"- {prompt}")
                    pdf.ln(0.5)

            self._write_section(pdf, "Discussion Prompts", body_discussion)

        if story.writing_tasks:
            def body_writing() -> None:
                self._set_font(pdf, 11)
                pdf.set_text_color(30, 30, 30)
                for task in story.writing_tasks:
                    prompt = task.get("prompt", "")
                    self._multi_cell(pdf, 0, LINE_HEIGHT, f"Prompt: {prompt}")
                    word_limit = task.get("word_limit")
                    if word_limit:
                        self._set_font(pdf, 9)
                        pdf.set_text_color(120, 120, 120)
                        self._multi_cell(pdf, 0, LINE_HEIGHT - 1, f"Word limit: {word_limit}")
                        self._set_font(pdf, 11)
                    focus = task.get("focus")
                    if focus:
                        self._set_font(pdf, 9)
                        pdf.set_text_color(120, 120, 120)
                        self._multi_cell(pdf, 0, LINE_HEIGHT - 1, f"Focus: {focus}")
                        self._set_font(pdf, 11)
                    pdf.ln(1)

            self._write_section(pdf, "Writing Tasks", body_writing)
