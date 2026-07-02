# tools/resume_parser_tool.py
#
# WHAT THIS FILE DOES:
# Reads your PDF resume and extracts structured information from it
# using an LLM. Instead of writing complex regex patterns to find
# your skills or experience, we just ask Gemini to read the text
# and return structured JSON.
#
# KEY CONCEPTS IN THIS FILE:
# - pdfplumber : a library that extracts raw text from PDF files
# - Pydantic   : a library for defining data shapes with validation.
#                Think of it as a "contract" — if the data doesn't
#                match the shape you defined, it raises an error immediately
#                rather than silently passing wrong data through your pipeline.
# - JSON       : JavaScript Object Notation — a standard text format
#                for structured data. Looks like a Python dict.
# - LangChain chain : a "chain" is a sequence of steps connected
#                together. input → prompt → LLM → output parser → result

import json
import pdfplumber
from pathlib import Path      # cleaner way to work with file pathsZ
from pydantic import BaseModel, Field
from typing import List, Optional
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import JsonOutputParser

from config import get_llm


# ---------------------------------------------------------------------------
# STEP 1: Define the data shape using Pydantic
# ---------------------------------------------------------------------------
# BaseModel is Pydantic's base class. When you inherit from it,
# your class gets automatic validation, type checking, and a
# clean .model_dump() method to convert it to a plain dict.
#
# Field(...) marks a field as required (the ... means "no default").
# Field("") gives it a default empty string.

class WorkExperience(BaseModel):
    company: str = Field(..., description="Company name")
    role: str = Field(..., description="Job title / role")
    duration: str = Field("", description="e.g. Jan 2022 - Mar 2024")
    description: str = Field("", description="Key responsibilities and achievements")

class Education(BaseModel):
    institution: str = Field(..., description="University or college name")
    degree: str = Field(..., description="Degree and field of study")
    year: str = Field("", description="Graduation year")

class ParsedResume(BaseModel):
    """
    This is the exact structure we want Gemini to fill in.
    Every agent downstream will receive a ParsedResume object —
    so they always know exactly what fields are available.
    """
    name: str = Field("", description="Candidate full name")
    email: str = Field("", description="Email address")
    phone: str = Field("", description="Phone number")
    location: str = Field("", description="Current city/location")
    summary: str = Field("", description="Professional summary or objective")
    skills: List[str] = Field(default_factory=list, description="List of technical and soft skills")
    work_experience: List[WorkExperience] = Field(default_factory=list)
    education: List[Education] = Field(default_factory=list)
    certifications: List[str] = Field(default_factory=list)
    languages: List[str] = Field(default_factory=list)
    raw_text: str = Field("", description="The original extracted PDF text")


# ---------------------------------------------------------------------------
# STEP 2: PDF text extractor
# ---------------------------------------------------------------------------

def extract_text_from_pdf(pdf_path: str) -> str:
    """
    Opens the PDF and pulls out all the raw text, page by page.
    pdfplumber is better than pypdf for resumes because it handles
    tables and multi-column layouts more cleanly.
    """
    path = Path(pdf_path)

    if not path.exists():
        raise FileNotFoundError(f"Resume not found at: {pdf_path}")

    if path.suffix.lower() != ".pdf":
        raise ValueError(f"Expected a PDF file, got: {path.suffix}")

    full_text = ""
    with pdfplumber.open(pdf_path) as pdf:
        for page_number, page in enumerate(pdf.pages, start=1):
            # extract_text() returns a string with the page content,
            # or None if the page has no extractable text (e.g. scanned image)
            text = page.extract_text()
            if text:
                full_text += f"\n--- Page {page_number} ---\n{text}"

    if not full_text.strip():
        raise ValueError(
            "No text could be extracted from the PDF. "
            "If it's a scanned resume, you'll need OCR (we can add that later)."
        )

    return full_text.strip()


# ---------------------------------------------------------------------------
# STEP 3: LLM-powered parser
# ---------------------------------------------------------------------------

def parse_resume_with_llm(raw_text: str) -> ParsedResume:
    """
    Sends the raw PDF text to Gemini with a structured prompt,
    asking it to return a JSON object matching our ParsedResume shape.

    HOW A LANGCHAIN CHAIN WORKS:
    prompt_template | llm | output_parser

    The | (pipe) operator connects steps.
    Data flows left to right — the prompt feeds the LLM,
    the LLM output feeds the parser, the parser returns the final result.
    """

    # JsonOutputParser tells LangChain: "expect JSON back, parse it automatically"
    # We pass our Pydantic model so it can include the schema in the prompt
    parser = JsonOutputParser(pydantic_object=ParsedResume)

    # PromptTemplate is a string with {placeholders} that get filled in at runtime.
    # {format_instructions} is automatically filled by JsonOutputParser —
    # it injects a description of the expected JSON structure into the prompt.
    prompt = PromptTemplate(
        template="""
You are an expert resume parser. Extract all information from the resume text below
and return it as a valid JSON object.

Be thorough — extract every skill, every job, every certification you can find.
For skills, include both technical skills (Python, TensorFlow, SQL) and
soft skills (leadership, communication) if mentioned.

{format_instructions}

Resume text:
{resume_text}

Return ONLY valid JSON. No explanation, no markdown, no code blocks.
""",
        input_variables=["resume_text"],
        partial_variables={"format_instructions": parser.get_format_instructions()},
    )

    # temperature=0.0 → we want deterministic, accurate extraction
    # not creative writing
    llm = get_llm(temperature=0.0)

    # Build the chain: prompt → llm → json parser
    # This is the core LangChain pattern you'll use everywhere
    chain = prompt | llm | parser

    # .invoke() runs the chain with the given inputs
    result_dict = chain.invoke({"resume_text": raw_text})

    # The parser returns a plain dict — convert it back to our Pydantic model
    # model_validate() creates a ParsedResume from a dict, with full validation
    return ParsedResume.model_validate(result_dict)


# ---------------------------------------------------------------------------
# STEP 4: Main public function — this is what other modules import
# ---------------------------------------------------------------------------

def load_and_parse_resume(pdf_path: str = "data/resume.pdf") -> ParsedResume:
    """
    Full pipeline: PDF path → extracted text → LLM parsing → ParsedResume object.
    This is the single function all agents will call.
    """
    print(f"Reading resume from: {pdf_path}")

    raw_text = extract_text_from_pdf(pdf_path)
    print(f"Extracted {len(raw_text)} characters of text from PDF")

    print("Sending to Gemini for structured parsing...")
    resume = parse_resume_with_llm(raw_text)

    # Store the original text on the object so agents can use it later
    resume.raw_text = raw_text

    print(f"Parsed successfully — found {len(resume.skills)} skills, "
          f"{len(resume.work_experience)} jobs, "
          f"{len(resume.education)} education entries")

    return resume


# ---------------------------------------------------------------------------
# Quick test — run this file directly to verify it works
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    resume = load_and_parse_resume("data/resume.pdf")

    print("\n=== PARSED RESUME ===")
    print(f"Name     : {resume.name}")
    print(f"Email    : {resume.email}")
    print(f"Skills   : {', '.join(resume.skills[:10])}...")
    print(f"\nWork Experience:")
    for job in resume.work_experience:
        print(f"  - {job.role} at {job.company} ({job.duration})")
    print(f"\nEducation:")
    for edu in resume.education:
        print(f"  - {edu.degree} from {edu.institution} ({edu.year})")

    # Save parsed resume to JSON for inspection
    output_path = "data/parsed_resume.json"
    with open(output_path, "w") as f:
        json.dump(resume.model_dump(), f, indent=2)
    print(f"\nSaved full parsed resume to {output_path}")