# Repository Guidelines

## Project Structure & Module Organization
Top-level scripts drive the research pipeline. `pipline_modified.py` builds `dataset_enhanced.parquet`; `tune_all_modified.py`, `robust_oos_search.py`, `scanner_grid_search.py`, and `tune_xgb_params.py` run search and evaluation; `train_xgb.py` and `train_multi_model.py` produce trained models and plots. Shared feature and labeling logic lives in `quant_modeling.py` and should remain the single source of truth. The live trading app is under `live/`: `main.py` starts the FastAPI dashboard, while `config.py`, `feature_engine.py`, `scanner.py`, `model_inference.py`, and `execution.py` handle runtime behavior. Generated artifacts include `plots/`, `live/data/`, model files, and CSV reports.

## Build, Test, and Development Commands
Activate the expected environment first: `conda activate quant`.

- `python pipline_modified.py`: build the enhanced parquet dataset from raw inputs.
- `python tune_all_modified.py`: search horizon/TP/SL settings and update `best_config.json`.
- `python robust_oos_search.py --scanner flow_reversal --label-mode first_touch`: run walk-forward validation.
- `python train_xgb.py` or `python train_multi_model.py`: train final models and write plots/results.
- `python -m live.main`: start the live FastAPI dashboard and trading loop.
- `python test_event_aligned.py` and `python test_horizon_close.py`: run the repository’s analysis-style validation scripts.

Use `--help` on the training and search scripts before adding new flags.

## Coding Style & Naming Conventions
Follow existing Python conventions: 4-space indentation, `snake_case` for functions/variables/files, and `UPPER_SNAKE_CASE` for constants. Keep type hints where the module already uses them, especially in `live/`. Prefer small helper functions over duplicating feature logic. Do not reimplement feature engineering inside `live/`; reuse pipeline logic from `pipline_modified.py` and `quant_modeling.py`.

## Testing Guidelines
This repo uses executable Python test scripts rather than a formal `pytest` suite. Name new checks `test_*.py` at the repository root. Tests should state required local artifacts up front, especially `dataset_enhanced.parquet`, `best_config.json`, and trained model files. When changing labeling, feature generation, or inference, include before/after metrics or output summaries in your PR notes.

## Commit & Pull Request Guidelines
Git history currently only shows `first commit`, so there is no strong legacy format to preserve. Use short imperative commit subjects such as `Add event-aligned feature guard`. Keep commits scoped to one change. PRs should summarize the affected workflow, list commands run, note regenerated artifacts, and include dashboard screenshots when `live/templates/index.html` or websocket UI behavior changes.

## Security & Configuration Tips
Never commit exchange credentials. Use environment variables such as `OKX_API_KEY`, `OKX_SECRET_KEY`, and `OKX_PASSPHRASE`. Treat generated datasets, model binaries, JSONL logs, and search-result CSVs as local artifacts unless a change explicitly intends to refresh them.
