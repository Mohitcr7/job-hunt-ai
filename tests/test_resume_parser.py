# tests/test_resume_parser.py
#
# Unit tests for the resume parser: PDF text extraction and the LLM parsing
# contract. No real PDFs and no API calls — pdfplumber is monkeypatched with
# fake pages, and the LLM is replaced by a RunnableLambda returning fixed JSON.
#
# The error paths get most of the attention here. A resume that fails to parse
# takes the whole pipeline down with it (the graph ends early without a resume),
# so the failure messages need to say what actually went wrong.

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


class _FakePage:
    def __init__(self, text):
        self._text = text

    def extract_text(self):
        return self._text


class _FakePDF:
    def __init__(self, pages):
        self.pages = [_FakePage(t) for t in pages]

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False


def _stub_pdfplumber(monkeypatch, pages):
    import tools.resume_parser_tool as rp

    monkeypatch.setattr(rp.pdfplumber, "open", lambda path: _FakePDF(pages))


# --- PDF extraction ---------------------------------------------------------

def test_missing_file_names_the_path(tmp_path):
    from tools.resume_parser_tool import extract_text_from_pdf

    missing = tmp_path / "nope.pdf"
    with pytest.raises(FileNotFoundError) as excinfo:
        extract_text_from_pdf(str(missing))

    assert "nope.pdf" in str(excinfo.value)


def test_a_non_pdf_is_rejected_before_parsing(tmp_path):
    """Pointing the parser at a .docx should fail clearly, not deep inside pdfplumber."""
    from tools.resume_parser_tool import extract_text_from_pdf

    docx = tmp_path / "resume.docx"
    docx.write_text("not a pdf")

    with pytest.raises(ValueError) as excinfo:
        extract_text_from_pdf(str(docx))

    assert ".docx" in str(excinfo.value)


def test_text_is_extracted_with_page_markers(tmp_path, monkeypatch):
    from tools.resume_parser_tool import extract_text_from_pdf

    pdf = tmp_path / "resume.pdf"
    pdf.write_bytes(b"%PDF-1.4")
    _stub_pdfplumber(monkeypatch, ["Page one text", "Page two text"])

    text = extract_text_from_pdf(str(pdf))

    assert "Page one text" in text
    assert "Page two text" in text
    assert "--- Page 1 ---" in text and "--- Page 2 ---" in text


def test_pages_without_extractable_text_are_skipped(tmp_path, monkeypatch):
    from tools.resume_parser_tool import extract_text_from_pdf

    pdf = tmp_path / "resume.pdf"
    pdf.write_bytes(b"%PDF-1.4")
    _stub_pdfplumber(monkeypatch, ["Real text", None])

    text = extract_text_from_pdf(str(pdf))

    assert "Real text" in text
    assert "--- Page 2 ---" not in text


def test_a_scanned_pdf_says_so(tmp_path, monkeypatch):
    """Every page image, no text layer — the message should point at OCR."""
    from tools.resume_parser_tool import extract_text_from_pdf

    pdf = tmp_path / "scanned.pdf"
    pdf.write_bytes(b"%PDF-1.4")
    _stub_pdfplumber(monkeypatch, [None, None])

    with pytest.raises(ValueError) as excinfo:
        extract_text_from_pdf(str(pdf))

    assert "OCR" in str(excinfo.value)


# --- the ParsedResume contract ----------------------------------------------

def test_parsed_resume_defaults_to_empty_collections():
    from tools.resume_parser_tool import ParsedResume

    resume = ParsedResume()

    assert resume.skills == []
    assert resume.work_experience == []
    assert resume.education == []
    assert resume.name == ""


def test_work_experience_requires_company_and_role():
    from pydantic import ValidationError
    from tools.resume_parser_tool import WorkExperience

    WorkExperience(company="Acme", role="Engineer")        # duration is optional
    with pytest.raises(ValidationError):
        WorkExperience(company="Acme")


def test_nested_structures_are_parsed_from_plain_dicts():
    from tools.resume_parser_tool import ParsedResume

    resume = ParsedResume.model_validate({
        "name": "Test Candidate",
        "skills": ["Python", "PyTorch"],
        "work_experience": [{"company": "Acme", "role": "ML Engineer", "duration": "2024-2025"}],
        "education": [{"institution": "IIT", "degree": "B.Tech CS", "year": "2024"}],
    })

    assert resume.work_experience[0].company == "Acme"
    assert resume.education[0].degree == "B.Tech CS"


# --- LLM parsing ------------------------------------------------------------

def test_llm_output_is_validated_into_a_parsed_resume(monkeypatch):
    import json
    from langchain_core.runnables import RunnableLambda
    import tools.resume_parser_tool as rp

    payload = json.dumps({
        "name": "Test Candidate",
        "email": "test@example.com",
        "summary": "ML engineer",
        "skills": ["Python", "FAISS"],
        "work_experience": [{"company": "Acme", "role": "ML Engineer"}],
        "education": [],
    })
    monkeypatch.setattr(rp, "get_llm", lambda temperature=0.0: RunnableLambda(lambda _: payload))

    resume = rp.parse_resume_with_llm("some resume text")

    assert resume.name == "Test Candidate"
    assert "FAISS" in resume.skills
    assert resume.work_experience[0].role == "ML Engineer"


def test_partial_llm_output_still_parses(monkeypatch):
    """The model routinely omits optional sections — that must not be fatal."""
    import json
    from langchain_core.runnables import RunnableLambda
    import tools.resume_parser_tool as rp

    monkeypatch.setattr(rp, "get_llm",
                        lambda temperature=0.0: RunnableLambda(lambda _: json.dumps({"name": "X"})))

    resume = rp.parse_resume_with_llm("text")

    assert resume.name == "X"
    assert resume.skills == []


def test_raw_text_is_attached_to_the_parsed_resume(tmp_path, monkeypatch):
    """
    Downstream matching embeds raw_text, so losing it silently degrades every
    match score without any visible error.
    """
    import json
    from langchain_core.runnables import RunnableLambda
    import tools.resume_parser_tool as rp

    pdf = tmp_path / "resume.pdf"
    pdf.write_bytes(b"%PDF-1.4")
    _stub_pdfplumber(monkeypatch, ["Distinctive resume body text"])
    monkeypatch.setattr(rp, "get_llm",
                        lambda temperature=0.0: RunnableLambda(lambda _: json.dumps({"name": "X"})))

    resume = rp.load_and_parse_resume(str(pdf))

    assert "Distinctive resume body text" in resume.raw_text
