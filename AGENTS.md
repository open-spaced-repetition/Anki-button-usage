# Repository Guidelines

## Project Structure & Module Organization
- `script.py`: Main batch analysis that reads Anki revlogs and writes `button_usage.jsonl`.
- `markov_chain.py`: Markov chain utilities used by the analysis.
- `analysis.ipynb`: Post-processing and summary reporting of `button_usage.jsonl`.
- `lag_dependence_table.py`: Standalone lag-dependence table analysis over revlogs.
- `button_usage.jsonl`: Generated output (large; only update when intentionally re-running analyses).
- `pyproject.toml` / `uv.lock`: Python project metadata and dependency lock.

## Build, Test, and Development Commands
This repository is script-driven (no build step).
- `uv run script.py`: Run the main analysis and append results to `button_usage.jsonl`.
- `uv run script.py --include-same-day`: Include same-day reviews in lag stats.
- `uv run script.py --max-workers 8 --chunksize 32`: Control parallelism.
- `uv run lag_dependence_table.py --data ../anki-revlogs-10k --max-lag 20`: Print the lag-dependence table.
- `uv run lag_dependence_table.py --include-same-day`: Include same-day reviews.
- `uv run ruff format`: Format Python code.

## Coding Style & Naming Conventions
- Language: Python 3.13+.
- Indentation: 4 spaces.
- Naming: `snake_case` for variables/functions, `CamelCase` for classes.
- Prefer `numpy`/`pandas` vectorized operations; avoid row-wise Python loops where possible.
- Formatting: run `uv run ruff format` before committing code changes.

## Testing Guidelines
- There is no automated test suite in this repo.
- Validate changes by running the relevant script(s) and checking outputs in `button_usage.jsonl` or notebook summaries.

## Commit & Pull Request Guidelines
- Commit messages in this repo are short, imperative summaries (e.g., “Add lag dependency summary and stats”).
- PRs should include:
  - A concise description of the analysis change.
  - The command(s) used to generate or update outputs.
  - Any changes to data assumptions (e.g., same-day review handling).

## Data & Configuration Notes
- The expected dataset path is `../anki-revlogs-10k` with `revlogs/` inside.
- Output files can be large; only regenerate `button_usage.jsonl` when needed and mention it explicitly in PRs.
