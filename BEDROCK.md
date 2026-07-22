# Deploying Job Hunt AI to Amazon Bedrock AgentCore

Your LangGraph crew runs **unchanged**. AgentCore Runtime wraps it and adds serverless deployment, session isolation, memory, identity, and observability. The only project additions are:

| File | Role |
|---|---|
| [`job_hunt_agent.py`](job_hunt_agent.py) | AgentCore entrypoint — `BedrockAgentCoreApp` + `@app.entrypoint` wrapping the crew |
| [`provision_bedrock.py`](provision_bedrock.py) | One-time Guardrail provisioning (contextual grounding) |
| [`tools/bedrock_kb_tool.py`](tools/bedrock_kb_tool.py) | Managed Knowledge Base retrieval — replaces the local FAISS layer |
| `bedrock` branch in [`config.py`](config.py) | Routes all 5 agents through Claude on Bedrock |

## How the existing crew maps onto Bedrock

| Local component | On AgentCore |
|---|---|
| `get_llm()` → Gemini | `get_llm()` → **Claude via `ChatBedrockConverse`** (set `LLM_PROVIDER=bedrock`) |
| FAISS Stage-1 retrieval (`vector_store_tool.py`) | **Managed Knowledge Base** (`bedrock_kb_tool.py`) |
| Tailor prompt "never fabricate" rule | **Platform Guardrail** — contextual grounding on every call |
| LangSmith traces | **AgentCore Observability** (CloudWatch Transaction Search) |
| `python main.py run` | **serverless invocation**, scales to zero between runs |

Because `get_llm()` is the single factory every agent uses, switching the provider moves the entire crew (matcher, tailor, reviewer, resume parser) to Bedrock with no per-agent changes.

## Prerequisites

- AWS account with credentials configured (`aws sts get-caller-identity` works)
- Model access enabled in the Bedrock console for your Claude model (Haiku for cheap dev, Sonnet for quality)
- Python 3.10+

## 1. Install

```bash
uv venv --python 3.10 && source .venv/bin/activate
uv pip install -r requirements.txt -r requirements-bedrock.txt

# Deploy tooling — the AgentCore CLI is the current recommended path:
npm install -g @aws/agentcore
# (the older `bedrock-agentcore-starter-toolkit` pip CLI also still works)
```

## 2. Provision the Guardrail (one time)

```bash
python provision_bedrock.py
# copy the printed exports:
export GUARDRAIL_ID=...  GUARDRAIL_VERSION=...
```

For the **Managed Knowledge Base**, the console is easiest: Bedrock → Knowledge Bases → *Create Managed KB* → point it at an S3 bucket of job postings; AWS handles ingest/chunk/embed/index. Then:

```bash
export MANAGED_KB_ID=...
```

## 3. Deploy to AgentCore Runtime

```bash
# Framework auto-detected; entrypoint is our wrapper file
agentcore configure --entrypoint job_hunt_agent.py

# Launch serverless (default = direct code deploy, no Docker needed).
# Env vars are stored in Secrets Manager and injected at runtime.
agentcore launch \
  --env LLM_PROVIDER=bedrock \
  --env MANAGED_KB_ID=$MANAGED_KB_ID \
  --env GUARDRAIL_ID=$GUARDRAIL_ID \
  --env GUARDRAIL_VERSION=$GUARDRAIL_VERSION
```

> Using the newer AgentCore CLI scaffold instead? `agentcore create --framework LangChain_LangGraph --model-provider Bedrock`, then move the logic from `job_hunt_agent.py` into the generated `app/.../main.py`.

## 4. Invoke

```bash
agentcore invoke '{"resume": "<paste resume text>"}'
```

Optional payload fields: `search_terms`, `locations`, `min_score`, `top_k`, or `jobs` (pre-retrieved postings to skip KB retrieval). From an SDK: get the ARN with `agentcore status`, then call `bedrock-agentcore:InvokeAgentRuntime` via boto3.

Test locally before deploying:

```bash
python job_hunt_agent.py          # starts the AgentCore dev server
agentcore invoke '{"resume": "..."}'   # against localhost
```

## 5. Observe

Enable CloudWatch Transaction Search once, and every agent step — each of the 5 nodes, each KB retrieval, each guardrail check — is traced in **AgentCore Observability**, the drop-in replacement for LangSmith traces.

## Cost notes

- Runtime scales to zero between invocations; you pay per invocation + model tokens.
- Keep the **top-15 retrieval cap** — it preserves the ~85%-fewer-LLM-calls optimization.
- Use **Haiku** for dev; switch `BEDROCK_MODEL_ID` to a **Sonnet** inference profile only for final-quality tailoring.
- Managed KB storage is the main standing cost; a small postings corpus is cheap.

## Why AgentCore and not "Bedrock Agents"

Bedrock Agents (the 2023 low-code product) is now **Agents Classic** and closes to new customers on **2026-07-30**. AgentCore is the current production path and is explicitly **framework-agnostic** (LangGraph, CrewAI, Strands, AutoGen), which is why this crew ports without a rewrite.

## Resume line this supports

> Migrated a 5-agent LangGraph/CrewAI job-matching system to **Amazon Bedrock AgentCore**: serverless runtime with session isolation, a **Managed Knowledge Base** replacing the FAISS RAG layer, and platform-level **Guardrails** (contextual grounding) enforcing zero hallucinated experience in tailored resumes.
