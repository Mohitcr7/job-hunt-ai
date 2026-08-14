# Evals

Two things get measured here: whether the matcher separates jobs worth
surfacing from jobs that are not, and whether the tailor invents experience.

```bash
python evals/matching_eval.py                 # FAISS only — free, offline, ~2s
python evals/matching_eval.py --stage rerank  # + cross-encoder  (needs OPENROUTER_API_KEY)
python evals/matching_eval.py --stage llm     # + LLM fit scoring (47 real calls)

python evals/fabrication_eval.py              # offline: prove the detector works
python evals/fabrication_eval.py --live       # run the real tailor and check its output
```

## What is in `data/`

`pairs.json` holds 47 job descriptions labelled against one of two resume
profiles in `resumes.json` — 23 real matches, 24 not. Every pair carries a
category, and the categories are the point:

| Category | Label | What it tests |
|---|---|---|
| `strong` | match | Obvious fit — the floor |
| `adjacent` | match | Related but not identical; the judgement call |
| `off_domain` | no match | Nurse, chef, accountant — pure topical rejection |
| `seniority_mismatch` | no match | Same field, demands 12+ years |
| `skill_mismatch` | no match | Same job title, entirely different stack |

The last two are the interesting ones. They read as topically identical to jobs
the candidate can do, so embeddings have almost nothing to go on.

`fabrication.json` holds 10 adversarial pairs where the job description demands
something the resume genuinely lacks — Kubernetes, a PhD, six years, Rust,
direct reports — so embellishing is the tempting move. Each case lists the
specific terms that could only appear if the model invented them.

## Honest limits

**The labels are mine, not a panel's.** I wrote both the job descriptions and
the labels. That makes this a regression harness and an argument about
architecture — not evidence of real-world accuracy. Two people would disagree
on several of the `adjacent` rows.

**A perfect score means the set is too easy, not that the matcher is perfect.**
The LLM stage scores 1.00 on every metric here. Read that as "the eval set no
longer discriminates at this stage", and add harder cases, rather than as a
production accuracy claim.

**The job descriptions are synthetic.** Real postings are copyrighted, and
committing scraped text would be both a licensing problem and a privacy one.
Synthetic text is cleaner and more consistent than reality, which flatters
every stage.

**The fabrication detector is lexical.** It catches a named technology,
credential or year-count claimed in the first person and absent from the
resume. It will not catch fabrication paraphrased around the vocabulary — "I
have orchestrated containers at scale" passes where "Kubernetes" would not. It
deliberately ignores negated sentences, because "I have not used Kubernetes" is
the honest answer this project wants.

## Reading the output

`average_precision` is the number to compare stages on. Precision and recall
depend on where you put the threshold, and the three stages produce scores on
three different scales — cosine similarity, batch percentile, and an LLM's
0-100 — so a fixed cut is not comparable across them. Average precision only
cares about ordering.

The per-category table is what justifies the architecture. If adding a stage
does not move the category it was added for, that stage is not earning its cost.
