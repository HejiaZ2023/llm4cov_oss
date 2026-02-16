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
