# jev-compare

Exploring a new AI model, "Jev", and evaluating how it compares to traditional LLMs for
zero-shot text classification.

The experiment is planned in [jev-experiment-plan.md](jev-experiment-plan.md), in three phases:

1. **Mechanics notebook**: Jev's `Noul`, `Choice` and `Score` primitives on realistic state.
2. **Benchmark**: Jev vs three frontier LLMs (zero-shot and few-shot) and a trained SVM on
   Banking77, AG News and IMDb, using balanced accuracy, cost and latency.
3. **Calibration**: reliability diagrams and ECE for Jev vs one frontier LLM on Banking77.

Status: all three phases are built and run. Phase 2 was run on 21 September 2026 with 1,000 rows
per dataset, sampled from each official test split. Phase 3 reuses those results and makes no API
calls. It covers more than the plan asks for: Jev against all three LLMs on all three datasets,
not one LLM on Banking77 alone.

## Setup

```
uv sync
```

Copy `.env.example` to `.env` and add your API keys (`AI_GATEWAY_API_KEY`, `ANTHROPIC_API_KEY`,
`OPENAI_API_KEY`, `GEMINI_API_KEY`). `.env` is gitignored.

## Usage

Run a notebook from the project root, for example:

```
uv run jupyter nbconvert --execute --to notebook --inplace --ExecutePreprocessor.timeout=-1 notebooks/02_jev_vs_llms_benchmark.ipynb
```

`02_jev_vs_llms_benchmark.ipynb` makes live API calls to all four models and trains three SVMs.
With `N_TEST` at 1,000 the API calls took 6.4 hours, 4.4 of them for Jev, because the gateway
limits it to 30 requests a minute. Fitting the SVMs took about 6 minutes more.
The notebook is in two parts. Part A (sections 1 to 5) makes the API calls and saves the results, and part B
(sections 6 onwards) reads the saved files and displays them, so it can be rerun alone in half a minute.
The `--ExecutePreprocessor.timeout=-1` option is needed because nbconvert stops a cell after 30
seconds by default. The test sample size is `N_TEST` in its first code cell. Model IDs and list
prices are in `configs/models.json`.

`03_calibration_analysis.ipynb` reads only the Phase 2 files in `results/` and makes no API calls.

## Development

```
uv run ruff check .
uv run ruff format .
```

Manage dependencies with `uv add <pkg>` (or `uv add --dev <pkg>`), not `pip`.

## Project layout

```
src/jev_compare/   Shared code: Jev client, datasets and sampling, LLM providers, benchmark runner
configs/labels/    Shared per-dataset label descriptions
notebooks/         Phase notebooks
results/           Samples, raw model responses and summary tables (tracked)
data/              Local scratch (contents gitignored)
```
