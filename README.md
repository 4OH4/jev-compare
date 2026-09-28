# jev-compare

Exploring a new AI model, [Jev](https://typesafe.ai/blog/introducing-system-one-models-and-jev), and evaluating how it compares to traditional LLMs for text classification tasks.

Three stages (a notebook each) were run:
1. [Exploring Jev: initial experiments on using the model](/notebooks/01_jev_mechanics.ipynb)
2. [Text classification: Comparing Jev vs LLMs (zero and few-shot) and a trained SVM](/notebooks/02_jev_vs_llms_benchmark.ipynb)
3. [Confidence calibration: Comparing Jev vs three frontier LLMs](/notebooks/03_calibration_analysis.ipynb)

## Key results

All figures are on 1,000 test rows per dataset. Jev is compared with the small models from
three labs (Claude Haiku 4.5, GPT-5.4 mini, Gemini 3.8 Flash) and a TF-IDF + linear SVM.

### Text classification accuracy and cost

Jev is as accurate as small LLMs for a fraction of the cost. It matches or beats Haiku and
GPT-5.4 mini on all three datasets, but trails Gemini 3.8 Flash by 2 to 4 points on AG News
and zero-shot Banking77. It costs 3 to 14 times less per row than the cheapest LLM. 

The SVM, trained on the full training set, still scores highest on topic (AG News) and
intent (Banking77), 3 to 8 points above Jev. On sentiment (IMDb) it is 6 points below every
model. Jev and the LLMs are the better choice when there is little or no labelled data, not
because they are more accurate.

![Comparing Jev, frontier LLMs and TF-IDF/SVM for text classification tasks](/results/balanced_accuracy.png)

### Reliability of confidence values

Jev is trained using Reinforcement Learning for Calibrated Decisions (RLCD). This means that the probability of its answers being correct should be approximately equal to its stated confidence value: i.e. `P(correct | p) ≈ p`. When it says p=0.9, it should be right about nine times in ten.

In practice it is no better calibrated than the LLMs' own stated confidence: it ranks third of four on
expected calibration error on every dataset. All four models are over-confident on `Banking77`
and `AG News` (Jev states 0.91 on average and is right 0.80 of the time on` Banking77`), and all
are close to calibrated on `IMDb`.

![Comparing Jev and frontier LLMs for the reliability of their confidence values](/results/reliability_banking77.png)

See https://quicqdev.github.io/Jev-vs-ML/ for a comparison of Jev against a variety of classical ML approaches for text classification.

## Datasets

These datasets were used:
- [Banking77](https://github.com/PolyAI-LDN/task-specific-datasets/tree/master/banking_data) (77-class intent, fine-grained)
- [AG News](https://huggingface.co/datasets/fancyzhx/ag_news) (4-class topic, coarse)
- [IMDb](https://huggingface.co/datasets/stanfordnlp/imdb) (binary sentiment)

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

`03_calibration_analysis.ipynb` reads only the notebook 02 files in `results/` and makes no API calls.

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
notebooks/         The three notebooks, numbered 01 to 03
results/           Samples, raw model responses and summary tables (tracked)
data/              Local scratch (contents gitignored)
```
