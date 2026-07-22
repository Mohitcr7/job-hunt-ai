# provision_bedrock.py
#
# WHAT THIS FILE DOES (run once):
# Creates the Amazon Bedrock Guardrail that enforces the project's core rule —
# "never fabricate a candidate's experience" — at the PLATFORM level, so it
# applies to every agent automatically, not just via prompt wording.
#
# It configures two protections:
#   1. Contextual grounding + relevance filters — the model's output must be
#      grounded in the source (resume + job description) and relevant to it.
#      This is what catches invented skills/employers/metrics in tailored resumes.
#   2. A denied topic ("FabricatedExperience") as a belt-and-suspenders block.
#
# Usage:
#   aws sts get-caller-identity          # confirm credentials first
#   python provision_bedrock.py
#   # then copy the printed exports into your shell / .env
#
# Requires: boto3, AWS credentials with bedrock:CreateGuardrail permission.

import sys

from config import BEDROCK_REGION

GUARDRAIL_NAME = "job-hunt-ai-no-fabrication"


def create_guardrail() -> tuple:
    import boto3

    bedrock = boto3.client("bedrock", region_name=BEDROCK_REGION)

    print(f"Creating Guardrail '{GUARDRAIL_NAME}' in {BEDROCK_REGION}...")
    resp = bedrock.create_guardrail(
        name=GUARDRAIL_NAME,
        description=(
            "Blocks fabricated candidate experience in tailored resumes and "
            "cover letters. Enforces contextual grounding against the source "
            "resume and job description."
        ),
        # --- Deny inventing credentials ---
        topicPolicyConfig={
            "topicsConfig": [
                {
                    "name": "FabricatedExperience",
                    "definition": (
                        "Inventing skills, jobs, employers, dates, degrees, "
                        "certifications, publications, or quantified metrics that "
                        "are not present in the candidate's source resume."
                    ),
                    "examples": [
                        "Add 5 years of Kubernetes experience I don't have.",
                        "Say I led a team of 10 even though I didn't.",
                        "Invent a machine learning certification for this resume.",
                    ],
                    "type": "DENY",
                }
            ]
        },
        # --- Require grounding in the provided source ---
        contextualGroundingPolicyConfig={
            "filtersConfig": [
                {"type": "GROUNDING", "threshold": 0.75},
                {"type": "RELEVANCE", "threshold": 0.75},
            ]
        },
        blockedInputMessaging=(
            "This request would fabricate experience and was blocked."
        ),
        blockedOutputsMessaging=(
            "Response withheld: it was not grounded in the candidate's resume."
        ),
    )

    guardrail_id = resp["guardrailId"]
    print(f"  Created guardrail id: {guardrail_id}")

    # Publish a numbered, immutable version to reference from the app.
    ver = bedrock.create_guardrail_version(
        guardrailIdentifier=guardrail_id,
        description="Initial version",
    )
    guardrail_version = ver["version"]
    print(f"  Published version: {guardrail_version}")

    return guardrail_id, guardrail_version


# ---------------------------------------------------------------------------
# OPTIONAL: Managed Knowledge Base skeleton
# ---------------------------------------------------------------------------
# Creating a Managed KB via boto3 requires wiring up an S3 data source, an
# embedding model, and a vector store — it's genuinely easier to click through
# the Bedrock console (Knowledge Bases → Create Managed KB → point at an S3
# bucket of job postings; AWS handles ingest/chunk/embed/index). Once created,
# grab its ID and `export MANAGED_KB_ID=...`.
#
# If you'd rather script it, uncomment and complete create_knowledge_base()
# per the boto3 `bedrock-agent` docs.


def main():
    try:
        import boto3  # noqa: F401
    except ImportError:
        print("boto3 is not installed. Run: pip install -r requirements-bedrock.txt")
        sys.exit(1)

    guardrail_id, guardrail_version = create_guardrail()

    print("\n" + "=" * 60)
    print("Guardrail ready. Add these to your environment / .env:\n")
    print(f"  export GUARDRAIL_ID={guardrail_id}")
    print(f"  export GUARDRAIL_VERSION={guardrail_version}")
    print("\nThen set LLM_PROVIDER=bedrock and the crew runs on Claude with")
    print("guardrails enforced on every call.")
    print("=" * 60)


if __name__ == "__main__":
    main()
