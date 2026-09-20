# jev-compare

Exploring a new AI model, "Jev", and evaluating how it compares to traditional LLMs for
zero-shot text classification.

## Setup

```
uv sync
```

Copy `.env.example` to `.env` and add any API keys you need. `.env` is gitignored.

## Usage

Nothing to run yet. The package (`src/jev_compare`) currently only loads `.env` on import.

## Development

```
uv run ruff check .
uv run ruff format .
```

Manage dependencies with `uv add <pkg>` (or `uv add --dev <pkg>`), not `pip`.

## Project layout

```
src/jev_compare/   Python package
data/              Local datasets (contents gitignored)
```
