# graph/pipeline.py
#
# WHAT THIS FILE DOES:
# Wires the agents into a LangGraph pipeline and provides run_pipeline(),
# the single entry point used by both the CLI and the web API.
#
# THE GRAPH:
#
#   scout ──► matcher ──► tailor ──► review ──► applier ──► END
#     │
#     └──(no jobs found)──────────────────────────────────► END
#
# Each node is a plain function that takes the state dict and returns
# the keys it updated. LangGraph merges those updates and passes the
# state to the next node.

from typing import Optional

from loguru import logger
from langgraph.graph import StateGraph, END

from graph.state import PipelineState
from agents.scout_agent import scout_node
from agents.matcher_agent import matcher_node
from agents.tailor_agent import tailor_node
from agents.applier_agent import applier_node
from review.autogen_crew import review_node


def _after_scout(state: PipelineState) -> str:
    """Conditional edge: skip the rest of the pipeline if scouting failed."""
    if not state.get("resume") or not state.get("jobs"):
        logger.warning("Nothing to process after scout — ending run")
        return "end"
    return "matcher"


def build_pipeline():
    """Constructs and compiles the LangGraph state machine."""
    graph = StateGraph(PipelineState)

    graph.add_node("scout", scout_node)
    graph.add_node("matcher", matcher_node)
    graph.add_node("tailor", tailor_node)
    graph.add_node("review", review_node)
    graph.add_node("applier", applier_node)

    graph.set_entry_point("scout")
    graph.add_conditional_edges("scout", _after_scout, {"matcher": "matcher", "end": END})
    graph.add_edge("matcher", "tailor")
    graph.add_edge("tailor", "review")
    graph.add_edge("review", "applier")
    graph.add_edge("applier", END)

    return graph.compile()


def run_pipeline(
    resume_path: str = "data/resume.pdf",
    search_terms: Optional[list] = None,
    locations: Optional[list] = None,
    hours_old: int = 24,
    min_score: Optional[int] = None,
) -> PipelineState:
    """
    Runs the full pipeline once and returns the final state.

    All arguments are optional — defaults come from config.py, so
    `run_pipeline()` with no arguments is a valid daily run.
    """
    from config import MATCH_THRESHOLD

    initial_state: PipelineState = {
        "resume_path": resume_path,
        "search_terms": search_terms,
        "locations": locations,
        "hours_old": hours_old,
        "min_score": min_score if min_score is not None else MATCH_THRESHOLD,
        "errors": [],
        "status": "starting",
    }

    logger.info("Starting Job Hunt AI pipeline run")
    app = build_pipeline()
    final_state = app.invoke(initial_state)

    logger.info("Pipeline finished")
    logger.info(f"  Jobs scraped : {len(final_state.get('jobs', []))}")
    logger.info(f"  Matches      : {len(final_state.get('matches', []))}")
    logger.info(f"  Kits created : {len(final_state.get('kits', []))}")
    logger.info(f"  New in tracker: {len(final_state.get('applied', []))}")
    for err in final_state.get("errors", []):
        logger.warning(f"  Non-fatal error: {err}")

    return final_state


if __name__ == "__main__":
    run_pipeline()
