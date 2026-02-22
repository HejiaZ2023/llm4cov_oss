# LLM4Cov: Parallel Synthesis Framework for LLM-driven EDA

![llmcov-design-new.png](images/llmcov-design-new.png "llmcov-design-new.png")

This repository provides the infrastructure used to generate the large-scale synthetic datasets described in the **LLM4Cov** paper. It is designed to orchestrate agentic data generation by bridging modern ML environments with legacy EDA simulation constraints.

## Architecture & Motivation
Deploying LLM agents in EDA environments presents a significant infrastructure mismatch:
* **Legacy Simulation Constraints:** EDA tools are often tied to fixed environments (e.g., specific OS/GCC versions) and hardware-locked commercial licenses.
* **ML Serving Requirements:** Modern LLM training and serving require high-frequency software updates and cloud-based GPU access.

**LLM4Cov** resolves this through a distributed execution model, decoupling LLM orchestration from the simulation environment to maximize generation throughput.

## Core Features
### 1. Heterogeneous Server Querying
* **LLM Tier:** Requests are handled via OpenAI-compatible endpoints, supporting both commercial APIs and self-hosted model servers (local or remote).
* **EDA Tier:** Simulation tasks are dispatched via SSH to remote servers, where the framework manages concurrent subprocesses to interface with tool-chain CLIs.

### 2. Extensible Workflow & Managed Parallelism
The framework transparently handles coroutine and thread management. Users can specify a custom workflow and send it to the workflow entrypoint; the system then handles the scaling and orchestration across available remote resources. This architecture was utilized to generate millions of synthetic data points for the ablation studies in our research.

## Technical Constraints & Implementation
### 1. Repository Synchronization
The current implementation utilizes SSH-based synchronization of the hardware repository to the EDA server on each execution. While this ensures isolation and works efficiently for standard datasets (e.g., CVDP-Ecov, CodeV), it may introduce overhead for exceptionally large designs (e.g., full-scale RISC-V cores). 

### 2. EDA Server Implementation & Licensing
Due to proprietary output formatting and licensing restrictions associated with commercial EDA tools, the specific server-side parsing and tool-call logic is not included in this public release.
* **Protocol Schema:** we provide a standardized [schema](src/llm4cov/eda_client/protocol.md) for client-server communication.
* **Implementation:** Users with valid tool licenses can utilize this schema and LLM-assisted coding to implement the server-side hooks required for their specific environment.
* **Targeted Share** Source code may be shared with users who holds Cadence Academic License with non-redistribution agreement.


## Installation
### Prerequisites
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

## Citation
If you use this framework or the associated research in your work, please cite:

```bibtex
@article{zhang2026llm4cov,
  title={LLM4Cov: Execution-Aware Agentic Learning for High-coverage Testbench Generation},
  author={Zhang, Hejia and Yu, Zhongming and Ho, Chia-Tung and Ren, Haoxing and Khailany, Brucek and Zhao, Jishen},
  journal={arXiv preprint arXiv:2602.16953},
  year={2026}
}
```