# llm4cov
---

## Prerequisites
- Python ≥ 3.10
- [uv](https://docs.astral.sh/uv/) (fast Python package manager)

### Install uv:
```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

### Setup environment

Clone the repo and sync all dependencies:
```bash
git clone https://github.com/HejiaZ2023/llm4cov.git
cd llm4cov
uv sync --all-extras --dev
uv pip install -e .
```

### Run all checks
Install hooks to run automatically on each commit
```bash
uv run pre-commit install
```

## Naming Convention
- `direct_infer`: single-pass generation/evaluation flow without iterative correction rounds.
- `agentic`: iterative flow that uses feedback from parsing, patching, and EDA results across rounds.

## Script Guide
- `scripts/batch_query_direct_infer.py`: Runs direct inference TB generation in batch, executes remote EDA, and writes per-run results plus aggregate evaluation stats.
- `scripts/batch_query_agentic_round.py`: Runs iterative agentic rounds for each sample, applies feedback-driven updates, and tracks round history files.
- `scripts/batch_query_eval.py`: Unified batch evaluator supporting plain, agentic, and markov-agentic modes with pass@k/best@k style summaries.
- `scripts/agentic_rounds_metrics.py`: Scans saved per-round inputs and reports pass rates by round count.
- `scripts/select_direct_infer_samples.py`: Selects representative direct_infer runs (xrun-fail/median/worst splits) and exports compact run directories.
- `scripts/select_direct_infer_samples_ablation.py`: Performs repeated best/random sampling ablations over direct_infer runs and emits per-round subsets.
- `scripts/synthetic_data_parse.py`: Converts run artifacts (`direct_infer` or `agentic`) into JSONL supervision rows for synthetic training data.
- `scripts/synthetic_data_filter.py`: Filters and labels parsed synthetic rows (quality, improvement, contamination, token-length constraints).
- `scripts/upload_to_hub.py`: Loads filtered JSONL files, resizes/shuffles splits, and pushes a dataset to Hugging Face Hub.
- `scripts/data_contamination_detect.py`: Detects cross-dataset contamination by similarity matching between train/eval contexts.
- `scripts/datasets_stats.py`: Prints token-length and composition statistics for supported source datasets.
- `scripts/run_vllm_eval_config.py`: Executes evaluation command configs and manages `vllm serve` lifecycle for local model serving.
