# tools/vector_store_tool.py
#
# WHAT THIS FILE DOES:
# Converts text (resume + job descriptions) into "embeddings" —
# lists of numbers that capture semantic meaning — then uses FAISS
# to find which job descriptions are closest in meaning to your resume.
#
# KEY CONCEPTS:
# - Embedding  : a list of ~384 numbers that represents the "meaning"
#                of a piece of text. Similar texts have similar numbers.
#                Think of it like GPS coordinates — nearby coordinates
#                mean nearby locations. Nearby embeddings mean similar meaning.
# - FAISS      : a library from Meta that can search through millions of
#                embeddings very fast using vector math (dot products).
# - Cosine similarity : the math used to compare two embeddings.
#                Returns a score from 0 (completely different) to 1 (identical).
# - Sentence Transformers : a model that converts text into embeddings locally
#                on your machine — no API call, no cost, works offline.

import json
import numpy as np
import faiss
from pathlib import Path
from typing import List, Tuple, TYPE_CHECKING
from dataclasses import dataclass

from config import MATCH_THRESHOLD

if TYPE_CHECKING:
    from sentence_transformers import SentenceTransformer


# ---------------------------------------------------------------------------
# STEP 1: Define what a Job looks like as a data structure
# ---------------------------------------------------------------------------
# @dataclass is Python's lightweight alternative to Pydantic for simple
# data containers. It auto-generates __init__, __repr__ etc.
# We use it here (instead of Pydantic) because jobs come from scrapers
# and we want a simple, fast container — not strict validation.

@dataclass
class Job:
    title: str
    company: str
    location: str
    description: str
    url: str
    platform: str           # "linkedin", "indeed", "naukri", etc.
    posted_date: str = ""
    salary: str = ""
    job_id: str = ""        # platform's own ID for deduplication
    experience: str = ""    # "1-4 yrs" (Naukri) or "Entry level" (LinkedIn)
    job_type: str = ""      # "Full-time", "Contract", "Internship", ...


@dataclass
class JobMatch:
    job: Job
    score: float            # 0.0 to 1.0 — how well it matches your resume
    score_percent: int      # score as 0-100 integer, easier to read


# ---------------------------------------------------------------------------
# STEP 2: The embedding model
# ---------------------------------------------------------------------------
# We use 'all-MiniLM-L6-v2' — a small, fast, free model that runs locally.
# It converts any text into a 384-dimensional embedding vector.
# First run will download ~90MB model weights — subsequent runs are instant.
#
# Why local instead of OpenAI embeddings?
# - Free (no API cost per embedding)
# - Fast (no network round trip)
# - Private (your resume never leaves your machine for this step)

EMBEDDING_MODEL_NAME = "all-MiniLM-L6-v2"
_embedding_model = None     # module-level cache — load once, reuse everywhere

def get_embedding_model() -> "SentenceTransformer":
    """
    Loads the embedding model once and caches it.
    The underscore prefix on _embedding_model is a Python convention
    meaning "this is a private/internal variable, not part of the public API".
    """
    global _embedding_model
    if _embedding_model is None:
        # Imported lazily — sentence-transformers pulls in torch (~2GB), which
        # only actual embedding calls need. Callers that just use Job/JobMatch
        # or the FAISS index (e.g. the test suite) never pay for it.
        from sentence_transformers import SentenceTransformer

        print(f"Loading embedding model: {EMBEDDING_MODEL_NAME}")
        print("(First run downloads ~90MB — subsequent runs are instant)")
        _embedding_model = SentenceTransformer(EMBEDDING_MODEL_NAME)
        print("Embedding model ready.")
    return _embedding_model


def embed_text(text: str) -> np.ndarray:
    """
    Converts a string into a 384-dimensional numpy array (embedding).
    numpy array = a fast, math-friendly list of numbers.
    """
    model = get_embedding_model()
    # encode() returns a numpy array of shape (384,)
    embedding = model.encode(text, convert_to_numpy=True)
    return embedding


def embed_texts(texts: List[str]) -> np.ndarray:
    """
    Converts a list of strings into a 2D numpy array of shape (N, 384).
    Much faster than calling embed_text() in a loop because the model
    processes them in batches internally.
    """
    model = get_embedding_model()
    embeddings = model.encode(texts, convert_to_numpy=True, show_progress_bar=True)
    return embeddings


# ---------------------------------------------------------------------------
# STEP 3: The FAISS vector store
# ---------------------------------------------------------------------------

class JobVectorStore:
    """
    Stores job embeddings in a FAISS index and provides
    fast similarity search against your resume embedding.

    HOW FAISS WORKS (simplified):
    1. You add N job embeddings to the index (like inserting rows into a table)
    2. You give it a query embedding (your resume)
    3. It returns the K most similar jobs, ranked by cosine similarity
    4. This search happens in milliseconds even with thousands of jobs
    """

    def __init__(self, index_path: str = "data/job_index.faiss"):
        self.index_path = index_path
        self.jobs: List[Job] = []       # stores the actual job objects
        self.index = None               # the FAISS index (stores embeddings)

    def _build_index(self, embeddings: np.ndarray):
        """
        Creates a FAISS index from a matrix of embeddings.
        IndexFlatIP = flat index using Inner Product (dot product).
        After normalising vectors to length 1, inner product == cosine similarity.
        """
        # Normalise vectors so inner product = cosine similarity
        # faiss.normalize_L2 modifies the array in-place
        faiss.normalize_L2(embeddings)

        # dimension = number of features per embedding (384 for MiniLM)
        dimension = embeddings.shape[1]

        # IndexFlatIP: brute-force exact search using inner product
        # Fine for up to ~100k jobs; for millions you'd use IndexIVFFlat
        self.index = faiss.IndexFlatIP(dimension)

        # add() inserts all embeddings into the index
        self.index.add(embeddings)
        print(f"FAISS index built with {self.index.ntotal} job embeddings")

    def add_jobs(self, jobs: List[Job]):
        """
        Embeds a list of jobs and adds them to the FAISS index.
        Call this after your scout agent scrapes fresh job listings.
        """
        if not jobs:
            print("No jobs to add.")
            return

        print(f"Embedding {len(jobs)} jobs...")

        # Build one text per job that captures the most important fields
        # More context = better embeddings = better matching
        job_texts = []
        for job in jobs:
            text = (
                f"Job Title: {job.title}\n"
                f"Company: {job.company}\n"
                f"Location: {job.location}\n"
                f"Description: {job.description}"
            )
            job_texts.append(text)

        embeddings = embed_texts(job_texts).astype("float32")
        # FAISS requires float32 — astype() converts the numpy array

        self.jobs.extend(jobs)
        self._build_index(embeddings)

    def find_matching_jobs(
        self,
        resume_text: str,
        top_k: int = 20,
        min_score: int = None
    ) -> List[JobMatch]:
        """
        Finds the top_k jobs most similar to the resume text.

        resume_text : the full text of your resume
        top_k       : how many candidates to return before threshold filtering
        min_score   : minimum match % to include (defaults to MATCH_THRESHOLD from config)
        """
        if self.index is None or self.index.ntotal == 0:
            print("Vector store is empty — add jobs first.")
            return []

        if min_score is None:
            min_score = MATCH_THRESHOLD

        # Embed the resume and reshape to (1, 384) — FAISS expects 2D input
        resume_embedding = embed_text(resume_text).astype("float32")
        resume_embedding = resume_embedding.reshape(1, -1)
        faiss.normalize_L2(resume_embedding)

        # .search() returns two arrays:
        # scores  : similarity scores, shape (1, top_k)
        # indices : positions in self.jobs, shape (1, top_k)
        scores, indices = self.index.search(resume_embedding, top_k)

        matches = []
        for score, idx in zip(scores[0], indices[0]):
            if idx == -1:
                # FAISS returns -1 when fewer than top_k results exist
                continue

            score_percent = int(score * 100)

            if score_percent >= min_score:
                matches.append(JobMatch(
                    job=self.jobs[idx],
                    score=float(score),
                    score_percent=score_percent,
                ))

        # Sort by score descending — best matches first
        matches.sort(key=lambda m: m.score, reverse=True)

        print(f"Found {len(matches)} jobs above {min_score}% match threshold")
        return matches

    def save(self):
        """Saves the FAISS index and job list to disk for reuse."""
        Path(self.index_path).parent.mkdir(parents=True, exist_ok=True)

        faiss.write_index(self.index, self.index_path)

        jobs_path = self.index_path.replace(".faiss", "_jobs.json")
        with open(jobs_path, "w") as f:
            # dataclasses.asdict() converts a dataclass to a plain dict
            import dataclasses
            json.dump([dataclasses.asdict(job) for job in self.jobs], f, indent=2)

        print(f"Saved index to {self.index_path}")
        print(f"Saved job data to {jobs_path}")

    def load(self):
        """Loads a previously saved index from disk."""
        if not Path(self.index_path).exists():
            print("No saved index found — starting fresh.")
            return

        self.index = faiss.read_index(self.index_path)

        jobs_path = self.index_path.replace(".faiss", "_jobs.json")
        with open(jobs_path, "r") as f:
            jobs_data = json.load(f)
            self.jobs = [Job(**job) for job in jobs_data]

        print(f"Loaded {self.index.ntotal} jobs from saved index")


# ---------------------------------------------------------------------------
# Quick test with dummy data
# ---------------------------------------------------------------------------

# if __name__ == "__main__":
#     # Create some fake jobs to test with
#     test_jobs = [
#         Job(
#             title="Senior Data Scientist",
#             company="TechCorp India",
#             location="Bangalore (Remote)",
#             description="Python, machine learning, TensorFlow, deep learning, NLP, data pipelines, SQL, cloud (AWS/GCP), LLM fine-tuning",
#             url="https://linkedin.com/jobs/123",
#             platform="linkedin",
#             posted_date="2025-06-20",
#         ),
#         Job(
#             title="Backend Java Developer",
#             company="OldBank Ltd",
#             location="Chennai",
#             description="Java Spring Boot, Oracle DB, COBOL, mainframe, enterprise banking software, on-site required",
#             url="https://naukri.com/jobs/456",
#             platform="naukri",
#             posted_date="2025-06-21",
#         ),
#         Job(
#             title="ML Engineer - GenAI",
#             company="AI Startup",
#             location="Remote",
#             description="LangChain, LLMs, RAG pipelines, vector databases, Python, MLOps, model deployment, Hugging Face, Gemini API",
#             url="https://indeed.com/jobs/789",
#             platform="indeed",
#             posted_date="2025-06-22",
#         ),
#     ]

#     # A mock resume summary to test matching
#     mock_resume = """
#     Data Scientist and ML Engineer with 3 years experience.
#     Skills: Python, TensorFlow, PyTorch, LangChain, LLMs, RAG,
#     NLP, SQL, GCP, Hugging Face, model deployment, data pipelines.
#     Built GenAI applications using Gemini and LangChain.
#     """

#     print("=== Testing Vector Store ===\n")

#     store = JobVectorStore()
#     store.add_jobs(test_jobs)

#     matches = store.find_matching_jobs(mock_resume, top_k=3, min_score=50)

#     print("\n=== TOP MATCHING JOBS ===")
#     for match in matches:
#         print(f"\n{match.score_percent}% match — {match.job.title} at {match.job.company}")
#         print(f"  Platform : {match.job.platform}")
#         print(f"  Location : {match.job.location}")
#         print(f"  URL      : {match.job.url}")