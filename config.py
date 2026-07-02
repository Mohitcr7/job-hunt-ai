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
# You'll update these to match what you're looking for
JOB_PREFERENCES = {
    "roles": ["Data Scientist", "AI Engineer", "ML Engineer", "GenAI Engineer"],
    "locations": ["Remote", "Bangalore", "Mumbai", "Pune", "Gurgaon", "Noida"],
    "experience_years": 2,  # in years
    "salary_min": 1200000,  # in INR per year
}

# --- LLM factory function ---
# A "factory function" is a function whose job is to CREATE and
# return an object — in this case, an LLM (language model) instance.
# Instead of every agent manually setting up the LLM, they all just
# call get_llm() and get back a ready-to-use model.
def get_llm(temperature: float = 0.3):
    """
    Returns a configured LLM based on the LLM_PROVIDER setting.

    temperature controls how creative vs predictable the model is:
      0.0 = very deterministic (good for structured tasks like parsing)
      0.7 = more creative (good for writing cover letters)
      1.0 = very random (rarely useful)

    The 'float' annotation after temperature is a type hint —
    it tells other developers (and you, later) what type of value
    this parameter expects. Python doesn't enforce it, but it
    makes the code much easier to read.
    """
    if LLM_PROVIDER == "gemini":
        # Import only when needed — keeps startup fast
        from langchain_google_genai import ChatGoogleGenerativeAI

        # ChatGoogleGenerativeAI is LangChain's wrapper around Gemini.
        # A "wrapper" is a class that takes a complex API and gives it
        # a simpler, standardised interface — so LangChain can treat
        # Gemini the same way it treats OpenAI or Anthropic.
        return ChatGoogleGenerativeAI(
            model="gemini-2.5-flash",
            google_api_key=GOOGLE_API_KEY,
            temperature=temperature,
            convert_system_message_to_human=True,
            # Gemini doesn't support system messages natively,
            # so this flag converts them to human-turn messages automatically.
        )

    elif LLM_PROVIDER == "openai":
        from langchain_openai import ChatOpenAI

        if not OPENAI_API_KEY:
            raise ValueError(
                "LLM_PROVIDER is set to 'openai' but OPENAI_API_KEY is missing in .env"
            )
        return ChatOpenAI(
            model="gpt-4o-mini",
            api_key=OPENAI_API_KEY,
            temperature=temperature,
        )

    else:
        # If someone types an unknown provider in .env, fail loudly
        # with a clear message rather than a cryptic crash later.
        raise ValueError(
            f"Unknown LLM_PROVIDER: '{LLM_PROVIDER}'. "
            f"Choose 'gemini' or 'openai' in your .env file."
        )


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

    if errors:
        print("\nConfig errors found:")
        for e in errors:
            print(f"  - {e}")
        print()
    else:
        print(f"Config OK — using {LLM_PROVIDER.upper()} ({"gemini-2.5-flash" if LLM_PROVIDER == 'gemini' else 'gpt-4o-mini'})")

    return len(errors) == 0


if __name__ == "__main__":
    validate_config()