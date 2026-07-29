# config.py
# What this file does: reads the .env file and makes all settings
# available as Python variables throughout the project.
# Think of it as the "settings panel" of your app.

import os
from dotenv import load_dotenv  # reads the .env file

# load_dotenv() finds the .env file and loads every line into
# Python's environment variables (os.environ)
load_dotenv()

# --- LLM Provider selection ---
# This lets you switch between Gemini and OpenAI by just
# changing one line in your .env file — no code changes needed.
LLM_PROVIDER = os.getenv("LLM_PROVIDER", "gemini")  # default to gemini

# --- LLM (Large Language Model) settings ---
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
# os.getenv("KEY") reads a value from the .env file by its name

# --- AWS Bedrock (optional provider + AgentCore deployment) ---
# Setting LLM_PROVIDER=bedrock routes every agent (matcher, tailor, reviewer,
# resume parser) through Claude on Amazon Bedrock — no per-agent code changes.
# boto3 resolves credentials from the standard chain (env vars, ~/.aws/, or an
# IAM role when running on AgentCore Runtime), so no keys live in this file.
BEDROCK_REGION = (
    os.getenv("BEDROCK_REGION")
    or os.getenv("AWS_REGION")
    or os.getenv("AWS_DEFAULT_REGION")
    or "us-east-1"
)
# Haiku is cheap for dev; switch to a Sonnet inference profile for final quality.
BEDROCK_MODEL_ID = os.getenv(
    "BEDROCK_MODEL_ID", "us.anthropic.claude-3-5-haiku-20241022-v1:0"
)
# Platform Guardrail — created by provision_bedrock.py. When set, it is attached
# to every Bedrock call and enforces "zero hallucinated experience" centrally.
GUARDRAIL_ID = os.getenv("GUARDRAIL_ID")
GUARDRAIL_VERSION = os.getenv("GUARDRAIL_VERSION")
# Managed Knowledge Base — optional RAG source that replaces the local FAISS
# layer when deployed on AgentCore (see tools/bedrock_kb_tool.py).
MANAGED_KB_ID = os.getenv("MANAGED_KB_ID")

# --- OpenRouter fallback (NVIDIA Nemotron 3 Super) ---
# OpenRouter is an OpenAI-compatible gateway, so this needs NO extra dependency.
# We use it as an automatic FALLBACK: if the primary provider (e.g. Gemini)
# errors or hits a rate limit mid-run, the call transparently retries against
# this model instead of failing the whole pipeline. Leave the key unset and the
# pipeline behaves exactly as before.
#
# Super over Ultra: both are free with equivalent reasoning/modalities, but
# Super is ~46% lower latency and ~2.5x higher throughput (1.37s/59 tok/s vs
# 2.53s/24 tok/s on OpenRouter's published p50s). A fallback exists to keep the
# pipeline moving fast when the primary fails — Ultra's larger size (1M vs 262K
# context) buys nothing here since a resume + one JD is a few thousand tokens.
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")
# NOTE ON PRIVACY: the ':free' endpoint is zero-cost, but OpenRouter LOGS free
# requests and asks you NOT to send confidential/personal data — and resumes
# are personal data. Drop the ':free' suffix (paid) for stricter privacy terms.
FALLBACK_MODEL_ID = os.getenv("FALLBACK_MODEL_ID", "nvidia/nemotron-3-super-120b-a12b:free")
OPENROUTER_BASE_URL = os.getenv("OPENROUTER_BASE_URL", "https://openrouter.ai/api/v1")
# On by default, but only actually activates when OPENROUTER_API_KEY is present.
ENABLE_LLM_FALLBACK = os.getenv("ENABLE_LLM_FALLBACK", "true").lower() == "true"

# --- Reranker (OpenRouter /rerank — narrows FAISS candidates before the LLM) ---
# A dedicated cross-encoder relevance model (free) that sharpens which jobs reach
# the expensive LLM fit-scoring, in one batched call off the generation budget.
# Reuses OPENROUTER_API_KEY + OPENROUTER_BASE_URL above.
ENABLE_RERANK = os.getenv("ENABLE_RERANK", "true").lower() == "true"
RERANK_MODEL_ID = os.getenv("RERANK_MODEL_ID", "nvidia/llama-nemotron-rerank-vl-1b-v2:free")
RERANK_TOP_N = int(os.getenv("RERANK_TOP_N", "8"))

# How many LLM calls run concurrently within a stage (fit-scoring, tailoring,
# review). Higher = faster, but hits provider rate limits sooner. 4 is a safe
# balance; the fallback covers any calls that do get throttled.
LLM_CONCURRENCY = int(os.getenv("LLM_CONCURRENCY", "4"))

# --- LangSmith observability ---
LANGCHAIN_API_KEY = os.getenv("LANGCHAIN_API_KEY")
LANGCHAIN_TRACING_V2 = os.getenv("LANGCHAIN_TRACING_V2", "true")
LANGCHAIN_PROJECT = os.getenv("LANGCHAIN_PROJECT", "job-hunt-ai")

# --- Notification credentials ---
SLACK_BOT_TOKEN = os.getenv("SLACK_BOT_TOKEN")
SENDGRID_API_KEY = os.getenv("SENDGRID_API_KEY")

# --- Database ---
DATABASE_URL = os.getenv("DATABASE_URL", "sqlite:///./data/applications.db")

# --- Business logic settings ---
# int() converts the string from .env into a number
MATCH_THRESHOLD = int(os.getenv("MATCH_THRESHOLD", "70"))

# --- Job platforms to target ---
# These are the platforms your Scout Agent will scrape
TARGET_PLATFORMS = ["linkedin", "indeed", "naukri", "company_pages"]

# --- Your job search preferences ---
# The defaults below are deliberately generic. Set JOB_ROLES / JOB_LOCATIONS in
# .env (comma-separated) to make them yours — .env is gitignored, so your actual
# search stays private even if you fork or publish this repo. The dashboard's
# Search Job screen overrides both per-run.
def _csv_env(name: str, fallback: list) -> list:
    raw = os.getenv(name, "")
    values = [item.strip() for item in raw.split(",") if item.strip()]
    return values or fallback


JOB_PREFERENCES = {
    "roles": _csv_env("JOB_ROLES", ["Software Engineer"]),
    "locations": _csv_env("JOB_LOCATIONS", ["Remote"]),
    "experience_years": int(os.getenv("EXPERIENCE_YEARS", "0")),  # in years
    "salary_min": int(os.getenv("SALARY_MIN", "0")),  # in INR per year
}

# --- LLM factory functions ---
# A "factory function" is a function whose job is to CREATE and return an
# object — here, an LLM (language model) instance. Every agent calls get_llm()
# and gets back a ready-to-use model, so provider choices live in one place.

def _build_openrouter_llm(temperature: float):
    """
    Builds an LLM backed by OpenRouter (NVIDIA Nemotron 3 Ultra by default).
    OpenRouter speaks the OpenAI API, so we reuse ChatOpenAI — no new dependency.
    Used both as an optional primary provider and as the automatic fallback.
    """
    from langchain_openai import ChatOpenAI

    if not OPENROUTER_API_KEY:
        raise ValueError("OPENROUTER_API_KEY is missing in .env")

    return ChatOpenAI(
        model=FALLBACK_MODEL_ID,
        api_key=OPENROUTER_API_KEY,
        base_url=OPENROUTER_BASE_URL,
        temperature=temperature,
        default_headers={"X-Title": "Job Hunt AI"},  # OpenRouter attribution (optional)
    )


def _build_provider_llm(provider: str, temperature: float):
    """Builds a single provider's LLM. get_llm() wraps this with a fallback."""
    if provider == "gemini":
        # Import only when needed — keeps startup fast.
        from langchain_google_genai import ChatGoogleGenerativeAI

        return ChatGoogleGenerativeAI(
            model="gemini-2.5-flash",
            google_api_key=GOOGLE_API_KEY,
            temperature=temperature,
            # Gemini doesn't support system messages natively, so this flag
            # converts them to human-turn messages automatically.
            convert_system_message_to_human=True,
        )

    elif provider == "openai":
        from langchain_openai import ChatOpenAI

        if not OPENAI_API_KEY:
            raise ValueError(
                "LLM_PROVIDER is set to 'openai' but OPENAI_API_KEY is missing in .env"
            )
        return ChatOpenAI(model="gpt-4o-mini", api_key=OPENAI_API_KEY, temperature=temperature)

    elif provider == "bedrock":
        # ChatBedrockConverse speaks the same LangChain interface as the other
        # providers, so the agents don't know they're talking to Claude on AWS.
        from langchain_aws import ChatBedrockConverse

        kwargs = dict(
            model=BEDROCK_MODEL_ID,
            region_name=BEDROCK_REGION,
            temperature=temperature,
            max_tokens=4096,
        )
        # Attach the platform Guardrail if one has been provisioned.
        if GUARDRAIL_ID:
            kwargs["guardrail_config"] = {
                "guardrailIdentifier": GUARDRAIL_ID,
                "guardrailVersion": GUARDRAIL_VERSION or "DRAFT",
                "trace": "enabled",
            }
        return ChatBedrockConverse(**kwargs)

    elif provider in ("openrouter", "nemotron"):
        return _build_openrouter_llm(temperature)

    else:
        # Unknown provider — fail loudly with a clear message.
        raise ValueError(
            f"Unknown LLM_PROVIDER: '{provider}'. "
            f"Choose 'gemini', 'openai', 'bedrock', or 'openrouter' in your .env file."
        )


def get_llm(temperature: float = 0.3):
    """
    Returns a ready-to-use LLM for the agents.

    temperature controls how creative vs predictable the model is:
      0.0 = very deterministic (parsing, scoring)
      0.7 = more creative (cover letters)

    RESILIENCE: if ENABLE_LLM_FALLBACK is on and OPENROUTER_API_KEY is set, the
    primary provider is wrapped so that any failure (rate limit, quota, transient
    API error) transparently retries against NVIDIA Nemotron 3 Ultra on
    OpenRouter. This is invisible to the agents — they still just call get_llm().
    """
    primary = _build_provider_llm(LLM_PROVIDER, temperature)

    # Wrap with the OpenRouter/Nemotron fallback — unless it's already primary.
    if ENABLE_LLM_FALLBACK and OPENROUTER_API_KEY and LLM_PROVIDER not in ("openrouter", "nemotron"):
        try:
            fallback = _build_openrouter_llm(temperature)
            # .with_fallbacks() is a LangChain Runnable feature: if `primary`
            # raises during .invoke(), it automatically retries with `fallback`.
            return primary.with_fallbacks([fallback])
        except Exception as e:
            print(f"[config] LLM fallback unavailable ({e}) — using primary only")
            return primary

    return primary


# --- Startup validation ---
def validate_config():
    """
    Checks that the minimum required keys are present.
    Called once when the app starts so you catch problems early,
    not halfway through a pipeline run.
    """
    errors = []

    if LLM_PROVIDER == "gemini" and not GOOGLE_API_KEY:
        errors.append("GOOGLE_API_KEY is missing — get one free at aistudio.google.com")

    if LLM_PROVIDER == "openai" and not OPENAI_API_KEY:
        errors.append("OPENAI_API_KEY is missing but LLM_PROVIDER=openai")

    if LLM_PROVIDER == "bedrock" and not BEDROCK_MODEL_ID:
        errors.append("LLM_PROVIDER=bedrock but BEDROCK_MODEL_ID is missing")
        # Note: we can't cheaply validate AWS credentials here — boto3 resolves
        # them lazily on the first call. Run `aws sts get-caller-identity` to
        # confirm your credentials, and enable model access in the Bedrock console.

    if LLM_PROVIDER in ("openrouter", "nemotron") and not OPENROUTER_API_KEY:
        errors.append(f"LLM_PROVIDER={LLM_PROVIDER} but OPENROUTER_API_KEY is missing")

    if errors:
        print("\nConfig errors found:")
        for e in errors:
            print(f"  - {e}")
        print()
    else:
        model_name = {
            "gemini": "gemini-2.5-flash",
            "openai": "gpt-4o-mini",
            "bedrock": BEDROCK_MODEL_ID,
            "openrouter": FALLBACK_MODEL_ID,
            "nemotron": FALLBACK_MODEL_ID,
        }.get(LLM_PROVIDER, "unknown")
        print(f"Config OK — using {LLM_PROVIDER.upper()} ({model_name})")

        # Report the fallback so it's clear when resilience is active.
        if ENABLE_LLM_FALLBACK and OPENROUTER_API_KEY and LLM_PROVIDER not in ("openrouter", "nemotron"):
            print(f"Fallback ON   — {FALLBACK_MODEL_ID} via OpenRouter (if primary fails)")
        elif ENABLE_LLM_FALLBACK and not OPENROUTER_API_KEY:
            print("Fallback OFF  — set OPENROUTER_API_KEY to enable Nemotron fallback")

    return len(errors) == 0


if __name__ == "__main__":
    validate_config()