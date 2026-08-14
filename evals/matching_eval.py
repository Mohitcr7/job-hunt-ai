"""
evals/matching_eval.py

Measures how well the matcher separates jobs worth surfacing from jobs that
are not, against hand-labelled pairs in data/pairs.json.

Run it:
    python evals/matching_eval.py                    # FAISS only — free, offline
    python evals/matching_eval.py --stage rerank     # + cross-encoder (needs OPENROUTER_API_KEY)
    python evals/matching_eval.py --stage llm        # + LLM fit scoring (costs real calls)
    python evals/matching_eval.py --json             # machine-readable output

WHY BOTH THRESHOLD AND RANKING METRICS:
The matcher is a ranker, not a classifier — the sheet shows everything and the
pipeline takes the top few. Precision/recall at a fixed threshold answers "if I
cut here, what do I get", but it moves whenever the score scale moves, and the
three stages are on three different scales (cosine, percentile, LLM 0-100).
Average precision is scale-free, so it is the number to compare stages on.

WHAT THE CATEGORY BREAKDOWN IS FOR:
The labels carry a category, and the categories are the actual argument for the
architecture. Embeddings should handle `off_domain` easily and fail on
`seniority_mismatch`, because a 12-years-required role reads as topically
identical to a job the candidate can do. If adding the LLM stage does not move
`seniority_mismatch`, the third stage is not earning its cost.
"""

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

DATA_DIR = Path(__file__).resolve().parent / "data"

# Each stage produces scores on its own scale, so each needs its own default cut.
# These are starting points for the sweep, not tuned constants.
DEFAULT_THRESHOLD = {"faiss": 40, "rerank": 55, "llm": 60}


def load_dataset():
    resumes = json.loads((DATA_DIR / "resumes.json").read_text())
    pairs = json.loads((DATA_DIR / "pairs.json").read_text())
    return resumes, pairs


def build_resume(profile: dict):
    """Turn a JSON profile into the ParsedResume the matcher expects."""
    from tools.resume_parser_tool import ParsedResume

    return ParsedResume(
        name=profile["name"],
        summary=profile["summary"],
        skills=profile["skills"],
        raw_text=profile["raw_text"],
    )


def build_jobs(pairs: list):
    """Turn labelled pairs into Job objects, carrying the pair id in job_id."""
    from tools.vector_store_tool import Job

    return [
        Job(
            title=p["title"],
            company=p["company"],
            location="Remote",
            description=p["jd"],
            url=f"https://eval.invalid/{p['id']}",
            platform="eval",
            job_id=p["id"],
        )
        for p in pairs
    ]


def score_faiss(resume, jobs) -> dict:
    from agents.matcher_agent import faiss_prefilter

    matches = faiss_prefilter(resume, jobs, min_score=0, top_k=len(jobs))
    return {m.job.job_id: m.score_percent for m in matches}


def score_rerank(resume, jobs) -> dict:
    from agents.matcher_agent import faiss_prefilter
    from tools.reranker_tool import score_all_jobs

    matches = faiss_prefilter(resume, jobs, min_score=0, top_k=len(jobs))
    query = f"{resume.summary} Skills: {', '.join(resume.skills)}"
    return {m.job.job_id: m.score_percent for m in score_all_jobs(query, matches)}


def score_llm(resume, jobs, experience_years: int) -> dict:
    """
    Real LLM fit scores. One call per job, run concurrently.

    A failed call keeps the embedding score rather than dropping the row, which
    mirrors what the pipeline does — an eval that silently discards failures
    would report a precision the production path never achieves.
    """
    from concurrent.futures import ThreadPoolExecutor, as_completed
    from agents.matcher_agent import faiss_prefilter, llm_score_job
    from config import LLM_CONCURRENCY

    matches = faiss_prefilter(resume, jobs, min_score=0, top_k=len(jobs))
    scores = {m.job.job_id: m.score_percent for m in matches}

    with ThreadPoolExecutor(max_workers=LLM_CONCURRENCY) as pool:
        futures = {
            pool.submit(llm_score_job, resume, m.job, experience_years): m
            for m in matches
        }
        for future in as_completed(futures):
            match = futures[future]
            try:
                scores[match.job.job_id] = future.result().score
            except Exception as e:
                print(f"  ! LLM scoring failed for {match.job.job_id}: {e}")

    return scores


def prf(rows: list, threshold: int) -> dict:
    """Precision, recall and F1 for 'predicted match' = score >= threshold."""
    tp = sum(1 for r in rows if r["positive"] and r["score"] >= threshold)
    fp = sum(1 for r in rows if not r["positive"] and r["score"] >= threshold)
    fn = sum(1 for r in rows if r["positive"] and r["score"] < threshold)

    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {"precision": precision, "recall": recall, "f1": f1,
            "tp": tp, "fp": fp, "fn": fn}


def average_precision(rows: list) -> float:
    """
    Scale-free ranking quality: mean of the precision after each true match,
    walking the list from best score to worst. 1.0 means every real match
    outranks every non-match. This is the number to compare stages on.
    """
    ranked = sorted(rows, key=lambda r: r["score"], reverse=True)
    positives = sum(1 for r in ranked if r["positive"])
    if not positives:
        return 0.0

    hits = 0
    total = 0.0
    for position, row in enumerate(ranked, start=1):
        if row["positive"]:
            hits += 1
            total += hits / position
    return total / positives


def precision_at_k(rows: list, k: int) -> float:
    ranked = sorted(rows, key=lambda r: r["score"], reverse=True)[:k]
    return sum(1 for r in ranked if r["positive"]) / k if k else 0.0


def best_threshold(rows: list) -> tuple:
    """Sweep every score present and return the cut with the highest F1."""
    best = (0, {"f1": -1.0})
    for candidate in sorted({r["score"] for r in rows}):
        stats = prf(rows, candidate)
        if stats["f1"] > best[1]["f1"]:
            best = (candidate, stats)
    return best


def evaluate(stage: str, threshold: int) -> dict:
    resumes, pairs = load_dataset()
    results = {"stage": stage, "threshold": threshold, "resumes": {}, "rows": []}

    for resume_id, profile in resumes.items():
        subset = [p for p in pairs if p["resume"] == resume_id]
        if not subset:
            continue

        print(f"\n=== {resume_id}: {len(subset)} labelled jobs ===")
        resume = build_resume(profile)
        jobs = build_jobs(subset)

        if stage == "faiss":
            scores = score_faiss(resume, jobs)
        elif stage == "rerank":
            scores = score_rerank(resume, jobs)
        elif stage == "llm":
            scores = score_llm(resume, jobs, profile.get("years_experience", 0))
        else:
            raise SystemExit(f"Unknown stage: {stage}")

        rows = [
            {
                "id": p["id"],
                "title": p["title"],
                "category": p["category"],
                "positive": p["label"] == "match",
                "score": scores.get(p["id"], 0),
            }
            for p in subset
        ]
        results["rows"].extend(rows)
        results["resumes"][resume_id] = {
            "n": len(rows),
            "positives": sum(1 for r in rows if r["positive"]),
            **prf(rows, threshold),
            "average_precision": average_precision(rows),
            "precision_at_k": precision_at_k(rows, sum(1 for r in rows if r["positive"])),
        }

    rows = results["rows"]
    results["overall"] = {
        "n": len(rows),
        "positives": sum(1 for r in rows if r["positive"]),
        **prf(rows, threshold),
        "average_precision": average_precision(rows),
    }

    cut, stats = best_threshold(rows)
    results["best_threshold"] = {"threshold": cut, **stats}

    # Per-category accuracy: did each row land on the right side of the cut?
    categories = {}
    for row in rows:
        bucket = categories.setdefault(row["category"], {"n": 0, "correct": 0})
        bucket["n"] += 1
        predicted = row["score"] >= threshold
        if predicted == row["positive"]:
            bucket["correct"] += 1
    results["by_category"] = categories

    return results


def print_report(results: dict):
    overall = results["overall"]
    print("\n" + "=" * 62)
    print(f"  MATCHER EVAL — stage: {results['stage']}   threshold: {results['threshold']}")
    print("=" * 62)
    print(f"  {overall['n']} labelled pairs, {overall['positives']} of them real matches\n")

    print(f"  Precision          {overall['precision']:.2f}"
          f"   ({overall['tp']} correct of {overall['tp'] + overall['fp']} surfaced)")
    print(f"  Recall             {overall['recall']:.2f}"
          f"   ({overall['tp']} found of {overall['positives']} real)")
    print(f"  F1                 {overall['f1']:.2f}")
    print(f"  Average precision  {overall['average_precision']:.2f}   (ranking quality, scale-free)")

    best = results["best_threshold"]
    print(f"\n  Best cut would be {best['threshold']} "
          f"(F1 {best['f1']:.2f}, precision {best['precision']:.2f}, recall {best['recall']:.2f})")

    print("\n  By category — did it land on the right side of the cut?")
    for name, bucket in sorted(results["by_category"].items()):
        rate = bucket["correct"] / bucket["n"]
        bar = "#" * int(round(rate * 20))
        print(f"    {name:<20} {bucket['correct']:>2}/{bucket['n']:<3} {rate:>5.0%}  {bar}")

    print("\n  Per resume:")
    for name, stats in results["resumes"].items():
        print(f"    {name:<20} P {stats['precision']:.2f}  R {stats['recall']:.2f}  "
              f"F1 {stats['f1']:.2f}  AP {stats['average_precision']:.2f}")

    misses = [r for r in results["rows"]
              if r["positive"] and r["score"] < results["threshold"]]
    false_hits = [r for r in results["rows"]
                  if not r["positive"] and r["score"] >= results["threshold"]]

    if misses:
        print(f"\n  Missed real matches ({len(misses)}):")
        for row in sorted(misses, key=lambda r: r["score"]):
            print(f"    {row['score']:>3}  {row['id']:<8} {row['title'][:44]}")
    if false_hits:
        print(f"\n  Surfaced non-matches ({len(false_hits)}):")
        for row in sorted(false_hits, key=lambda r: -r["score"]):
            print(f"    {row['score']:>3}  {row['id']:<8} [{row['category']}] {row['title'][:34]}")
    print()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", choices=["faiss", "rerank", "llm"], default="faiss",
                        help="how far down the matching funnel to run (default: faiss)")
    parser.add_argument("--threshold", type=int, default=None,
                        help="score at or above which a job counts as surfaced")
    parser.add_argument("--min-f1", type=float, default=None,
                        help="exit non-zero below this F1, for use as a CI gate")
    parser.add_argument("--json", action="store_true", help="print raw results as JSON")
    args = parser.parse_args()

    threshold = args.threshold
    if threshold is None:
        threshold = DEFAULT_THRESHOLD[args.stage]

    results = evaluate(args.stage, threshold)

    if args.json:
        print(json.dumps(results, indent=2))
    else:
        print_report(results)

    if args.min_f1 is not None and results["overall"]["f1"] < args.min_f1:
        print(f"FAIL: F1 {results['overall']['f1']:.2f} below floor {args.min_f1}")
        sys.exit(1)


if __name__ == "__main__":
    main()
