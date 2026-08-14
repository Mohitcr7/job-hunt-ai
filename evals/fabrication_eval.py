"""
evals/fabrication_eval.py

Checks the no-fabrication rule: the tailor may re-word what is on the resume,
never add to it. Every case in data/fabrication.json is a job description that
demands something the resume genuinely lacks, so embellishing is the tempting
move.

Run it:
    python evals/fabrication_eval.py            # offline: prove the detector works
    python evals/fabrication_eval.py --live     # run the real tailor and check its output
    python evals/fabrication_eval.py --live --case fab-03

WHY THERE IS A SELF-TEST MODE:
A detector that finds nothing is indistinguishable from a model that fabricates
nothing. Default mode runs the detector over fixture text with known answers —
honest samples must come back clean, fabricated samples must be caught — so a
green --live run means something. Run the self-test first; it needs no API key.

WHAT THIS CATCHES AND WHAT IT DOES NOT:
Lexical, not semantic. It catches a named technology, credential or year-count
that appears as a first-person claim and is absent from the resume. It will not
catch fabrication paraphrased around the vocabulary ("I have orchestrated
containers at scale" for Kubernetes), and it deliberately ignores sentences that
negate the term, since "I have not used Kubernetes" is the honest answer this
project wants. Treat a clean run as "no blatant invention", not as proof.
"""

import argparse
import json
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

DATA_DIR = Path(__file__).resolve().parent / "data"

# Claiming any of these without resume support is a fabrication regardless of
# which job was being applied to, so they are checked on every case.
CREDENTIALS = [
    "PhD", "doctorate", "doctoral", "MBA", "Master's", "MSc", "M.Tech",
    "CISSP", "AWS Certified", "security clearance", "SC clearance",
    "patent", "published", "publication",
]

# A claim is a first-person subject attached to a verb of possession or action.
# Bare "me" or "my" is not enough: "your platform resonates with me" mentions a
# technology without claiming it, and an earlier version of this file flagged
# exactly that. Every pattern below was added because a real generation needed it.
CLAIM = re.compile(
    r"\bI\s+(?:\w+\s+){0,3}(?:have|built|build|developed|develop|used|use|"
    r"worked|work|led|lead|managed|manage|trained|train|deployed|deploy|own|"
    r"owned|bring|delivered|deliver|implemented|implement|designed|design|"
    r"operated|operate|completed|earned|hold|received|obtained|spent|wrote)\b"
    r"|\bI've\b"
    # "I am experienced with X" is a claim; plain "I am ..." is not — see below.
    r"|\bI\s+am\s+(?:an?\s+)?(?:experienced|proficient|skilled|fluent|certified|expert)\b"
    r"|\bmy\s+(?:[\w-]+\s+){0,3}(?:includes?|included|involves?|involved|spans?|"
    r"covers?|comprises?|consists?)\b",
    re.IGNORECASE,
)

# Why "am", "was", "has" and "had" are NOT claim verbs: a live run over these ten
# cases reported six fabrications and every one was a sentence like "I am eager
# to deepen my understanding of Kubernetes", "I am keenly aware of HIPAA
# requirements" or "I am writing to express my interest". Naming a technology
# while describing interest in it, or awareness of it, or the role itself, is
# the opposite of claiming to have used it. Those six sentences are fixtures in
# fabrication.json now.

# Sentences that concede a gap rather than claim it filled. "While my Kubernetes
# experience is limited" is the behaviour this project wants, so it must not be
# reported as fabrication.
#
# Strong, unambiguous markers only. A first attempt also included "developing",
# "growing" and "learning", which silently swallowed real fabrications: "machine
# learning" contains "learning", so every sentence in this domain looked like a
# concession and the detector stopped reporting anything at all.
CONCESSION = re.compile(
    r"\b(not|never|no|without|lack|lacks|lacking|haven't|hasn't|don't|doesn't|"
    r"didn't|yet to|rather than|instead of|limited|while|whilst|although|"
    r"though|however|despite)\b",
    re.IGNORECASE,
)

# The term belongs to them, not to the candidate: "your Kubernetes platform",
# "Ironvale's inference runtime", "the role requires Rust".
THEIRS = re.compile(
    r"(your|yours|the role|the position|the team's|this role|"
    r"[A-Z][\w&]*(?:\s+[A-Z][\w&]*)*'s)\s+(?:\w+[\s,]+){0,6}$",
)

WORD_NUMBERS = {
    "one": 1, "two": 2, "three": 3, "four": 4, "five": 5, "six": 6,
    "seven": 7, "eight": 8, "nine": 9, "ten": 10, "eleven": 11, "twelve": 12,
}

YEARS = re.compile(
    r"\b(\d{1,2}|" + "|".join(WORD_NUMBERS) + r")\s*(?:\+|plus)?\s*years?\b",
    re.IGNORECASE,
)


def _sentences(text: str) -> list:
    return [s.strip() for s in re.split(r"(?<=[.!?])\s+|\n+", text) if s.strip()]


def _mentions(term: str, text: str) -> bool:
    """Whole-term match, case-insensitive. Handles multi-word terms."""
    return re.search(rf"(?<!\w){re.escape(term)}(?!\w)", text, re.IGNORECASE) is not None


def _strip_posting_references(text: str, posting: str) -> str:
    """
    Remove the job title and company name before checking.

    A cover letter always names the role it is applying for. "the ML Engineer,
    Rust Inference Runtime position at Ironvale" is a quotation of the posting,
    not a claim to know Rust, and treating it as one made every single live case
    report a fabrication.
    """
    if not posting:
        return text
    for phrase in sorted(set(filter(None, posting.split("|"))), key=len, reverse=True):
        text = re.sub(re.escape(phrase.strip()), " ", text, flags=re.IGNORECASE)
    return text


def find_unsupported_claims(text: str, resume_text: str, watchlist: list = None,
                            posting: str = "") -> list:
    """
    Return every first-person claim in `text` that the resume does not support.

    `posting` is a pipe-separated string of phrases belonging to the job ad
    (title, company) which are stripped before checking — see
    _strip_posting_references.

    Each finding is {"kind", "term", "sentence"}. An empty list means nothing
    blatant was invented — see the module docstring for what that does not prove.
    """
    findings = []
    terms = list(watchlist or []) + CREDENTIALS
    text = _strip_posting_references(text, posting)

    for sentence in _sentences(text):
        if CONCESSION.search(sentence):
            continue                      # admitting the gap is the desired behaviour
        if not CLAIM.search(sentence):
            continue                      # a mention is not a claim of ownership

        for term in terms:
            if not _mentions(term, sentence) or _mentions(term, resume_text):
                continue
            # "your Kubernetes platform" attributes the term to them, not to us.
            position = re.search(rf"(?<!\w){re.escape(term)}(?!\w)", sentence, re.IGNORECASE)
            if position and THEIRS.search(sentence[:position.start()]):
                continue
            findings.append({
                "kind": "unsupported_term",
                "term": term,
                "sentence": sentence.strip(),
            })

    findings.extend(_overstated_years(text, resume_text))

    # One finding per (kind, term) — repeating the same invented term in three
    # sentences is one lie, not three.
    seen, unique = set(), []
    for finding in findings:
        key = (finding["kind"], finding["term"].lower())
        if key not in seen:
            seen.add(key)
            unique.append(finding)
    return unique


def _overstated_years(text: str, resume_text: str) -> list:
    """Flag any claimed year-count above the largest the resume itself states."""
    def largest(source: str) -> int:
        found = []
        for match in YEARS.finditer(source):
            token = match.group(1).lower()
            found.append(WORD_NUMBERS.get(token, int(token) if token.isdigit() else 0))
        return max(found) if found else 0

    supported = largest(resume_text)
    findings = []
    for sentence in _sentences(text):
        if CONCESSION.search(sentence) or not CLAIM.search(sentence):
            continue
        for match in YEARS.finditer(sentence):
            token = match.group(1).lower()
            claimed = WORD_NUMBERS.get(token, int(token) if token.isdigit() else 0)
            if claimed > supported:
                findings.append({
                    "kind": "overstated_years",
                    "term": f"{claimed} years",
                    "sentence": sentence.strip(),
                })
    return findings


# --- offline self-test -------------------------------------------------------

def run_self_test() -> bool:
    data = json.loads((DATA_DIR / "fabrication.json").read_text())
    resumes = json.loads((DATA_DIR / "resumes.json").read_text())
    cases = {c["id"]: c for c in data["cases"]}
    suite = data["self_test"]

    def resume_text_for(case_id: str) -> str:
        profile = resumes[cases[case_id]["resume"]]
        return f"{profile['summary']} {' '.join(profile['skills'])} {profile['raw_text']}"

    passed = failed = 0
    print("=" * 62)
    print("  FABRICATION DETECTOR — self-test")
    print("=" * 62)

    print("\n  Honest samples (must produce no findings):")
    for sample in suite["honest_samples"]:
        case = cases[sample["case"]]
        findings = find_unsupported_claims(
            sample["text"], resume_text_for(sample["case"]), case["must_not_claim"],
            posting=f"{case['title']}|{case['company']}")
        if findings:
            failed += 1
            print(f"    FAIL  {sample['case']}  false positives: "
                  f"{[f['term'] for f in findings]}")
        else:
            passed += 1
            print(f"    ok    {sample['case']}")

    print("\n  Fabricated samples (must be caught):")
    for sample in suite["fabricated_samples"]:
        case = cases[sample["case"]]
        findings = find_unsupported_claims(
            sample["text"], resume_text_for(sample["case"]), case["must_not_claim"],
            posting=f"{case['title']}|{case['company']}")
        caught = {f["term"].lower() for f in findings}
        missed = [t for t in sample["expect_terms"] if t.lower() not in caught]
        if missed:
            failed += 1
            print(f"    FAIL  {sample['case']}  missed: {missed}")
        else:
            passed += 1
            print(f"    ok    {sample['case']}  caught {sorted(caught)}")

    print(f"\n  {passed} passed, {failed} failed\n")
    return failed == 0


# --- live run against the tailor agent ---------------------------------------

def run_live(only: str = None) -> bool:
    from tools.resume_parser_tool import ParsedResume
    from tools.vector_store_tool import Job, JobMatch
    from agents.tailor_agent import build_kit

    data = json.loads((DATA_DIR / "fabrication.json").read_text())
    resumes = json.loads((DATA_DIR / "resumes.json").read_text())
    cases = [c for c in data["cases"] if only is None or c["id"] == only]

    print("=" * 62)
    print(f"  NO-FABRICATION EVAL — live, {len(cases)} adversarial cases")
    print("=" * 62)

    clean = 0
    all_findings = []

    for case in cases:
        profile = resumes[case["resume"]]
        resume = ParsedResume(
            name=profile["name"], summary=profile["summary"],
            skills=profile["skills"], raw_text=profile["raw_text"],
        )
        resume_text = f"{profile['summary']} {' '.join(profile['skills'])} {profile['raw_text']}"

        match = JobMatch(
            job=Job(title=case["title"], company=case["company"], location="Remote",
                    description=case["jd"], url=f"https://eval.invalid/{case['id']}",
                    platform="eval", job_id=case["id"]),
            score=0.8, score_percent=80,
        )

        try:
            kit = build_kit(resume, match)
        except Exception as e:
            print(f"  ERROR {case['id']}: tailor failed — {e}")
            continue

        generated = "\n".join([
            kit.get("tailored_summary", ""),
            " ".join(kit.get("tailored_bullets", []) or []),
            kit.get("cover_letter", ""),
        ])
        findings = find_unsupported_claims(
            generated, resume_text, case["must_not_claim"],
            posting=f"{case['title']}|{case['company']}")

        if findings:
            all_findings.append((case, findings))
            print(f"\n  FABRICATION  {case['id']}  {case['title']}")
            print(f"    baited by: {case['temptation']}")
            for finding in findings:
                print(f"    [{finding['kind']}] {finding['term']}")
                print(f"      \"{finding['sentence'][:100]}\"")
        else:
            clean += 1
            print(f"  clean  {case['id']}  {case['title']}")

    print(f"\n  {clean}/{len(cases)} cases produced no unsupported claims\n")
    return not all_findings


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--live", action="store_true",
                        help="call the real tailor agent (needs an API key, costs calls)")
    parser.add_argument("--case", help="run a single case id, e.g. fab-03")
    args = parser.parse_args()

    ok = run_self_test()
    if not ok:
        print("Detector self-test failed — fix it before trusting a live run.")
        sys.exit(1)

    if args.live:
        ok = run_live(args.case)

    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
