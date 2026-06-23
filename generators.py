"""DOCX export helpers — IEEE, Elsevier, and Blog formats."""

import io
from datetime import datetime

from docx import Document
from docx.shared import Pt, Inches
from docx.enum.text import WD_ALIGN_PARAGRAPH


def _body(doc: Document, text: str, size: int = 10,
          align=WD_ALIGN_PARAGRAPH.JUSTIFY) -> None:
    p = doc.add_paragraph()
    p.alignment = align
    run = p.add_run(text)
    run.font.size = Pt(size)


def _heading(doc: Document, text: str, size: int = 11,
             align=WD_ALIGN_PARAGRAPH.LEFT) -> None:
    p = doc.add_paragraph()
    p.alignment = align
    run = p.add_run(text)
    run.bold = True
    run.font.size = Pt(size)


def _margins(doc: Document, top=1.0, bottom=1.0, left=1.25, right=1.25) -> None:
    for sec in doc.sections:
        sec.top_margin    = Inches(top)
        sec.bottom_margin = Inches(bottom)
        sec.left_margin   = Inches(left)
        sec.right_margin  = Inches(right)


def _to_bytes(doc: Document) -> bytes:
    buf = io.BytesIO()
    doc.save(buf)
    return buf.getvalue()


# ── IEEE ──────────────────────────────────────────────────────────────────────
def generate_ieee_docx(
    title: str,
    authors: str,
    affiliation: str,
    abstract: str,
    keywords: str,
    sections: dict,
    references: str = "",
) -> bytes:
    doc = Document()
    _margins(doc, top=0.75, bottom=0.75, left=0.75, right=0.75)

    # Title
    p = doc.add_paragraph()
    p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    r = p.add_run(title)
    r.bold = True
    r.font.size = Pt(22)

    # Authors / affiliation
    for text, size, italic in [(authors, 11, False), (affiliation, 10, True)]:
        p = doc.add_paragraph()
        p.alignment = WD_ALIGN_PARAGRAPH.CENTER
        r = p.add_run(text)
        r.font.size = Pt(size)
        r.italic = italic

    doc.add_paragraph()

    # Abstract
    _heading(doc, "Abstract", size=10, align=WD_ALIGN_PARAGRAPH.CENTER)
    _body(doc, abstract, size=10)

    # Keywords
    p = doc.add_paragraph()
    r = p.add_run("Keywords: ")
    r.bold = True
    r.font.size = Pt(10)
    p.add_run(keywords).font.size = Pt(10)

    doc.add_paragraph()

    # Body sections
    for i, (name, content) in enumerate(sections.items(), start=1):
        _heading(doc, f"{i}. {name.upper()}", size=10)
        _body(doc, content, size=10)
        doc.add_paragraph()

    # References
    if references:
        _heading(doc, "REFERENCES", size=10)
        _body(doc, references, size=9)

    return _to_bytes(doc)


# ── Elsevier ──────────────────────────────────────────────────────────────────
def generate_elsevier_docx(
    title: str,
    authors: str,
    affiliation: str,
    abstract: str,
    keywords: str,
    sections: dict,
    highlights: str = "",
    references: str = "",
) -> bytes:
    doc = Document()
    _margins(doc)

    # Title
    p = doc.add_paragraph()
    r = p.add_run(title)
    r.bold = True
    r.font.size = Pt(16)

    doc.add_paragraph()

    for text, size, italic in [(authors, 11, False), (affiliation, 10, True)]:
        p = doc.add_paragraph()
        r = p.add_run(text)
        r.font.size = Pt(size)
        r.italic = italic

    doc.add_paragraph()

    # Highlights (Elsevier-specific)
    if highlights:
        _heading(doc, "Highlights", size=11)
        for line in highlights.splitlines():
            if line.strip():
                p = doc.add_paragraph(style="List Bullet")
                r = p.add_run(line.strip())
                r.font.size = Pt(10)
        doc.add_paragraph()

    # Abstract
    _heading(doc, "Abstract", size=12)
    _body(doc, abstract, size=10)

    p = doc.add_paragraph()
    r = p.add_run("Keywords: ")
    r.bold = True
    r.font.size = Pt(10)
    p.add_run(keywords).font.size = Pt(10)

    doc.add_paragraph()

    for i, (name, content) in enumerate(sections.items(), start=1):
        _heading(doc, f"{i}. {name}", size=12)
        _body(doc, content, size=10)
        doc.add_paragraph()

    if references:
        _heading(doc, "References", size=12)
        _body(doc, references, size=9)

    return _to_bytes(doc)


# ── Blog post ─────────────────────────────────────────────────────────────────
def generate_blog_docx(
    title: str,
    author: str,
    introduction: str,
    sections: dict,
    conclusion: str,
    tags: str = "",
) -> bytes:
    doc = Document()
    _margins(doc)

    p = doc.add_paragraph()
    p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    r = p.add_run(title)
    r.bold = True
    r.font.size = Pt(20)

    p = doc.add_paragraph()
    p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    r = p.add_run(f"By {author}  |  {datetime.now().strftime('%B %d, %Y')}")
    r.italic = True
    r.font.size = Pt(10)

    if tags:
        p = doc.add_paragraph()
        p.alignment = WD_ALIGN_PARAGRAPH.CENTER
        p.add_run(f"Tags: {tags}").font.size = Pt(9)

    doc.add_paragraph()

    _heading(doc, "Introduction", size=13)
    _body(doc, introduction, size=11)
    doc.add_paragraph()

    for name, content in sections.items():
        _heading(doc, name, size=13)
        _body(doc, content, size=11)
        doc.add_paragraph()

    if conclusion:
        _heading(doc, "Conclusion", size=13)
        _body(doc, conclusion, size=11)

    return _to_bytes(doc)
