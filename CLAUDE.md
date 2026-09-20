# jev-compare

## Purpose
Compare the "Jev" model against frontier LLMs (Claude Haiku 4.5, a small OpenAI model, a Gemini
Flash-class model) and a TF-IDF + linear SVM on zero-shot / few-shot text classification
(Banking77, AG News, IMDb), then analyse calibration. **The full plan is in
`jev-experiment-plan.md`; read it, and the open issues in `plan-review.md`, before starting any
phase.** Work in phase order (1 mechanics
notebook, 2 benchmark, 3 calibration) and don't start a phase until asked.

## Tooling
- uv manages deps: `uv add`, `uv sync`, `uv run`. Never bare `pip`; don't activate the venv.
- Lint/format: `uv run ruff check .` and `uv run ruff format .`. No test suite yet.
- Notebooks are committed WITH outputs, and must run top to bottom (`uv run jupyter nbconvert
  --execute --to notebook --inplace <nb>`).

## Layout
- `src/jev_compare/` shared code (dataset loading, sampling, Jev client, LLM clients); reused by
  every notebook, not rewritten per notebook
- `configs/labels/` one shared label-description file per dataset, used by every model
- `notebooks/` the three phase notebooks
- `results/` tracked outputs: samples, raw `results_*.jsonl`, `summary.csv`. Not `outputs/` or
  `data/`, which are gitignored.
- `data/` local scratch, gitignored

## Rules from the plan
- Jev is called over plain HTTP (httpx). Check Vercel's current AI Gateway docs and print one
  full live response before writing any parser; don't assume the JS SDK's field names.
- Switching to TypeSafe's direct endpoint (`TYPESAFE_API_KEY`) needs user confirmation first.
- The official test split is touched only for final scoring; zero-shot calls never see train.
- Record exact model ID strings, save every raw response, and record API failures explicitly
  (never silently count them as wrong or right). No response caching for repeat runs.

## Secrets
Keys live in `.env` (gitignored); names go in `.env.example`. Never commit real secrets, and
don't print keys in notebook output.
