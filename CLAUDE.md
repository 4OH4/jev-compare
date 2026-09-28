# jev-compare

## Purpose
Initial experimentation with a new AI model, "Jev", comparing it against frontier LLMs (Claude 
Haiku 4.5, a small OpenAI model, a Gemini Flash-class model) and a TF-IDF + linear SVM on zero-shot / few-shot text classification (Banking77, AG News, IMDb), then analyse calibration. The experiments are described in `README.md`, with one Jupyter notebook for each.

## Tech stack (as built)
Pure Python (3.13 in `.python-version`), no JavaScript.

- **Jev**: plain HTTP with `httpx` to Vercel AI Gateway, `POST
  https://ai-gateway.vercel.sh/v1/evaluate`, `Authorization: Bearer $AI_GATEWAY_API_KEY`, model
  `typesafe-ai/jev`. There is no Python SDK for this route. The request carries `model`, `state`
  and `questions`; question types are `boolean` (TypeSafe's "Noul"), `choice` and `score`. The
  response shape, confirmed from a live call on 2026-09-20, is in the `src/jev_compare/jev.py`
  docstring; read it before changing the parser. The response carries no version string. Cost is
  taken from `providerMetadata.gateway.marketCost` (`gateway.cost` was "0"); list price is
  $0.042 per million input tokens. The gateway allows 30 requests and 250,000 tokens a minute;
  the client paces requests at 27 a minute.
- **TypeSafe direct endpoint** (`POST https://api.typesafe.ai/v1/systemone`, `TYPESAFE_API_KEY`)
  was not used. It is billed by TypeSafe rather than through Vercel, so switching to it needs
  user confirmation first.
- **LLMs** through the official SDKs, with SDK retries off (`max_retries=0`; the runner in
  `bench.py` retries transient failures) and structured output limited to the valid labels:
  - `anthropic`: Messages API with an `output_config` JSON schema and explicit prompt caching
    (`cache_control`). `claude-haiku-4-5-20251001`, temperature 0.
  - `openai`: Responses API with a strict `json_schema` text format and `prompt_cache_key`.
    `gpt-5.4-mini` (returned as `gpt-5.4-mini-2026-03-17`), reasoning effort `none`.
  - `google-genai`: `models.generate_content` with `response_json_schema`, and explicit context
    caches for the few-shot prefix. `gemini-3.8-flash`, temperature 0, thinking budget 0.
  Model IDs, parameters and list prices are in `configs/models.json` (checked 2026-09-20); the
  IDs actually run are in `results/models_used.csv`.
- **Datasets**: Hugging Face `datasets` for AG News (`fancyzhx/ag_news`) and IMDb
  (`stanfordnlp/imdb`). Banking77 comes from PolyAI's official CSVs on GitHub, because the Hub
  copy (`PolyAI/banking77`) is a loading script that `datasets` 5 no longer runs.
- **SVM**: scikit-learn `TfidfVectorizer(ngram_range=(1, 2), sublinear_tf=True)` with
  `LinearSVC`; `C` chosen by `GridSearchCV` with 5-fold `StratifiedKFold` on train.
- Also pandas, numpy, matplotlib and python-dotenv; ruff, ipykernel and nbconvert as dev
  dependencies.

## Tooling
- uv manages deps: `uv add`, `uv sync`, `uv run`. Never bare `pip`; don't activate the venv.
- Lint/format: `uv run ruff check .` and `uv run ruff format .`. No test suite yet.
- Notebooks are committed WITH outputs, and must run top to bottom (`uv run jupyter nbconvert
  --execute --to notebook --inplace <nb>`). For notebook 02 add
  `--ExecutePreprocessor.timeout=-1`: its run cell takes far longer than nbconvert's default
  30-second limit per cell.

## Layout
- `src/jev_compare/` shared code (dataset loading, sampling, Jev client, LLM clients); reused by
  every notebook, not rewritten per notebook
- `configs/labels/` one shared label-description file per dataset, used by every model
- `notebooks/` the three notebooks, numbered 01 to 03
- `results/` tracked outputs: samples, raw `results_*.jsonl`, `summary.csv`. Not `outputs/` or
  `data/`, which are gitignored.
- `data/` local scratch, gitignored

## Rules
- Before relying on a new Jev field or question type, check Vercel's current AI Gateway docs and
  print one full live response; don't assume the JS AI SDK's field names.
- The official test split is touched only for final scoring; zero-shot calls never see train.
- Record exact model ID strings, save every raw response, and record API failures explicitly
  (never silently count them as wrong or right). No response caching for repeat runs.

## Secrets
Keys live in `.env` (gitignored); names go in `.env.example`. Never commit real secrets, and
don't print keys in notebook output.

## Writing style

Applies to notebook prose, READMEs, docstrings, comments and anything else a person will read.

- Write plain, ordinary British English, as a careful engineer would in a report. Use the simplest
  word that is accurate. Avoid US tech-industry slang and marketing phrasing (e.g. "primitives",
  "rungs", "under the hood", "punchy", "landscape", "deep dive", "unlock", "collapse" for a
  drop in a value, "breaks" for "gives poor results").
- Don't use analogies or metaphors where a direct description works. If a metaphor is
  really needed, say what it stands for.
- Don't narrate or enumerate what the text is about ("six of these, two of those", "three
  hand-picked examples", "n=8"). Describe the thing itself: "the same bug reports", "a few
  tickets". Give a count only where it is the result being reported, or where the reader needs it
  to judge how far to trust the result, and then state it once.
- Use one name for one thing. Keep the terms the API and the code use for code and field names
  (`boolean`, `choice`, `score`, `confidence`), and explain them in ordinary words the first time.
- Report results as they are, including negative and mixed ones. Don't dress up a weak result,
  and don't use first-person hedging such as "I'd rather show that".
- Prefer short sentences. Put the finding first and the caveat second.
- Before finishing a notebook or document, reread the prose once for the points above.

