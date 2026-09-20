# jev-compare

## Purpose
Exploring a new AI model, "Jev", and evaluating how it compares to traditional LLMs for
zero-shot text classification.

## Tooling
- Dependencies are managed by uv: `uv add`, `uv sync`, `uv run`. Never use bare `pip`, and don't
  activate the venv manually.
- Lint and format with Ruff: `uv run ruff check .` and `uv run ruff format .`.
- No test suite or type checker is configured yet.

## Layout
- `src/jev_compare/` - the package (loads `.env` via python-dotenv on import)
- `data/` - local datasets; contents are gitignored (only `.gitkeep` is tracked)

## Secrets
API keys live in `.env` (gitignored). Add new variable names to `.env.example` with empty
values. Never commit real secrets.
