import gzip
import json
from pathlib import Path
from typing import Any

import datasets as ds
import pandas as pd
from huggingface_hub import snapshot_download
from tqdm import tqdm
from transformers import AutoTokenizer

from llm4cov.datasets.rtl_preprocess import extract_potential_top
from llm4cov.datasets.types import CovExpectation, DataContext, DataFile, LlmGenTbContext, data_context_to_llm_gen_tb_context

TOKENIZER = None


def construct_tokenizer() -> Any:
    global TOKENIZER
    if TOKENIZER is None:
        TOKENIZER = AutoTokenizer.from_pretrained(
            "Qwen/Qwen3-Coder-30B-A3B-Instruct",
            trust_remote_code=True,  # Qwen uses custom tokenizer logic
            use_fast=True,
        )
    return TOKENIZER


# Suppose current directory is: llm4cov/src/llm4cov/datasets/load,
# Target cache dir is:          llm4cov/data/cache/datasets
CACHE_PATH = Path(__file__).parents[3] / "data/cache/datasets"


def _parse_if_str(v):
    """Parse a string that may be a JSON/Python-literal list or dict."""
    if isinstance(v, str):
        import json as _json
        import ast as _ast
        try:
            return _json.loads(v)
        except Exception:
            try:
                return _ast.literal_eval(v)
            except Exception:
                return v
    return v


def load_dataset_by_name(dataset_name: str, split: str = "train") -> list[DataContext]:
    CACHE_PATH.mkdir(parents=True, exist_ok=True)
    if dataset_name.endswith(".parquet") or (
        dataset_name.startswith("/") and Path(dataset_name).exists()
    ):
        return load_local_parquet(dataset_name, split)
    if dataset_name == "zhuyaoyu/CodeV-R1-dataset":
        return load_codev_r1_dataset(
            split, revision="ffc4698071098044c72bde14fdad309eb3a1c5da", subset_name="sft"
        )
    elif dataset_name == "wilyub/VeriThoughtsTrainSet":
        return load_verithoughts_dataset(split)
    elif dataset_name == "hez2024/cvdp_ecov_eval":
        return load_cvdp_ecov_dataset(split)
    elif dataset_name == "Senlimulin/2026UCSDIntern_SlimeRL_training_dataset":
        return load_slimerl_training_dataset(split)
    elif dataset_name == "Senlimulin/CodeV_R1_5918_dataset":
        return load_codev_r1_dataset(split, ds_name_overwrite=dataset_name)
    elif dataset_name.startswith("hez2024/CodeV-R1-dataset"):
        return load_codev_r1_dataset(split, ds_name_overwrite=dataset_name)
    else:
        raise ValueError(f"Unsupported dataset: {dataset_name}")


def _compute_rtl_tokens(rtl_files: list[DataFile]) -> int:
    tokenizer = construct_tokenizer()
    total_length = sum(len(tokenizer.encode(rtl_file.content)) for rtl_file in rtl_files)
    return total_length


def _codev_row_to_context(
    row: pd.Series,
    *,
    ds_name: str,
    rtl_tokens_cache: dict[str, int],
    potential_top_cache: dict[str, list[str]],
) -> tuple[DataContext, bool, bool]:
    spec = _parse_if_str(row["question"])[1]["content"]
    rtl = _parse_if_str(row["ground_truth"])[0]["content"]
    spec_files = [DataFile(name="design_requirements.txt", content=spec)]
    rtl_files = [DataFile(name="dut.sv", content=rtl)]
    id = str(row["problem_id"])

    rtl_dirty = False
    top_dirty = False
    if id in rtl_tokens_cache:
        rtl_tokens = rtl_tokens_cache[id]
    else:
        rtl_tokens = _compute_rtl_tokens(rtl_files)
        rtl_tokens_cache[id] = rtl_tokens
        rtl_dirty = True
    if id in potential_top_cache:
        potential_top = potential_top_cache[id]
    else:
        potential_top = list(extract_potential_top([rtl]))
        potential_top_cache[id] = potential_top
        top_dirty = True

    targets = _parse_if_str(row.get("targets", []))
    if hasattr(targets, "tolist"):
        targets = targets.tolist()
    if not isinstance(targets, list):
        targets = []
    context = DataContext(
        id=id,
        rtl_files=rtl_files,
        spec_files=spec_files,
        dataset_name=ds_name,
        rtl_tokens=rtl_tokens,
        potential_top=potential_top,
        misc={"targets": targets},
    )
    return context, rtl_dirty, top_dirty


def load_slimerl_training_dataset(split: str = "train") -> list[DataContext]:
    ds_name = "Senlimulin/2026UCSDIntern_SlimeRL_training_dataset"
    snapshot_dir = Path(snapshot_download(repo_id=ds_name, repo_type="dataset"))
    if split == "validation":
        parquet_path = snapshot_dir / "val_codev_rl_test_with_targets.parquet"
    elif split == "train":
        parquet_path = snapshot_dir / "train_clean.parquet"
    else:
        raise ValueError(f"Unsupported split for {ds_name}: {split}")

    df = pd.read_parquet(parquet_path)
    ret = []
    cache_prefix = "SlimeRL_training_dataset"
    rtl_tokens_cache_path = CACHE_PATH / f"{cache_prefix}_rtl_tokens_cache.json.gz"
    if rtl_tokens_cache_path.exists():
        with gzip.open(rtl_tokens_cache_path, "rt") as f:
            rtl_tokens_cache: dict[str, int] = json.load(f)
    else:
        rtl_tokens_cache = {}
    is_rtl_tokens_cache_dirty = False
    potential_top_cache_path = CACHE_PATH / f"{cache_prefix}_potential_top_cache.json.gz"
    if potential_top_cache_path.exists():
        with gzip.open(potential_top_cache_path, "rt") as f:
            potential_top_cache: dict[str, list[str]] = json.load(f)
    else:
        potential_top_cache = {}
    is_potential_top_cache_dirty = False

    for _, row in tqdm(df.iterrows(), total=len(df), desc="Loading SlimeRL training dataset"):
        context, rtl_dirty, top_dirty = _codev_row_to_context(
            row,
            ds_name=ds_name,
            rtl_tokens_cache=rtl_tokens_cache,
            potential_top_cache=potential_top_cache,
        )
        ret.append(context)
        is_rtl_tokens_cache_dirty = is_rtl_tokens_cache_dirty or rtl_dirty
        is_potential_top_cache_dirty = is_potential_top_cache_dirty or top_dirty
    if is_rtl_tokens_cache_dirty:
        with gzip.open(rtl_tokens_cache_path, "wt") as f:
            json.dump(rtl_tokens_cache, f)
    if is_potential_top_cache_dirty:
        with gzip.open(potential_top_cache_path, "wt") as f:
            json.dump(potential_top_cache, f)
    return ret


def load_codev_r1_dataset(
    split: str = "train",
    revision: str | None = None,
    ds_name_overwrite: str | None = None,
    subset_name: str | None = None,
) -> list[DataContext]:
    ds_name = "zhuyaoyu/CodeV-R1-dataset" if ds_name_overwrite is None else ds_name_overwrite
    load_ds_args = {"split": split}
    if revision is not None:
        load_ds_args["revision"] = revision
    if subset_name is not None:
        load_ds_args["name"] = subset_name
    data_ds = ds.load_dataset(ds_name, **load_ds_args)
    df = pd.DataFrame(data_ds)
    ret = []
    rtl_tokens_cache_path = CACHE_PATH / "CodeV_R1_rtl_tokens_cache.json.gz"
    if rtl_tokens_cache_path.exists():
        with gzip.open(rtl_tokens_cache_path, "rt") as f:
            rtl_tokens_cache: dict[str, int] = json.load(f)
    else:
        rtl_tokens_cache = {}
    is_rtl_tokens_cache_dirty = False
    potential_top_cache_path = CACHE_PATH / "CodeV_R1_potential_top_cache.json.gz"
    potential_top_cache: dict[str, list[str]] = dict()
    if potential_top_cache_path.exists():
        with gzip.open(potential_top_cache_path, "rt") as f:
            potential_top_cache = json.load(f)
    is_potential_top_cache_dirty = False
    for _, row in tqdm(df.iterrows(), total=len(df), desc="Loading CodeV-R1 dataset"):
        spec = row["question"][1]["content"]
        rtl = row["ground_truth"][0]["content"]
        spec_files = [DataFile(name="design_requirements.txt", content=spec)]
        rtl_files = [DataFile(name="dut.sv", content=rtl)]
        id = str(row["problem_id"])
        if id in rtl_tokens_cache:
            rtl_tokens = rtl_tokens_cache[id]
        else:
            rtl_tokens = _compute_rtl_tokens(rtl_files)
            rtl_tokens_cache[id] = rtl_tokens
            is_rtl_tokens_cache_dirty = True
        if id in potential_top_cache:
            potential_top = potential_top_cache[id]
        else:
            potential_top = list(extract_potential_top([rtl]))
            potential_top_cache[id] = potential_top
            is_potential_top_cache_dirty = True
        context = DataContext(
            id=id,
            rtl_files=rtl_files,
            spec_files=spec_files,
            dataset_name=ds_name,
            rtl_tokens=rtl_tokens,
            potential_top=potential_top,
        )
        ret.append(context)
    if is_rtl_tokens_cache_dirty:
        with gzip.open(rtl_tokens_cache_path, "wt") as f:
            json.dump(rtl_tokens_cache, f)
    if is_potential_top_cache_dirty:
        with gzip.open(potential_top_cache_path, "wt") as f:
            json.dump(potential_top_cache, f)
    return ret


def load_verithoughts_dataset(split: str = "train") -> list[DataContext]:
    ds_name = "wilyub/VeriThoughtsTrainSet"
    data_ds = ds.load_dataset(ds_name, split=split)
    df = pd.DataFrame(data_ds)
    ret = []
    rtl_tokens_cache_path = CACHE_PATH / "VeriThoughtsTrainSet_rtl_tokens_cache.json.gz"
    if rtl_tokens_cache_path.exists():
        with gzip.open(rtl_tokens_cache_path, "rt") as f:
            rtl_tokens_cache: dict[str, int] = json.load(f)
    else:
        rtl_tokens_cache = {}
    is_rtl_tokens_cache_dirty = False
    potential_top_cache_path = CACHE_PATH / "VeriThoughtsTrainSet_potential_top_cache.json.gz"
    potential_top_cache: dict[str, list[str]] = dict()
    if potential_top_cache_path.exists():
        with gzip.open(potential_top_cache_path, "rt") as f:
            potential_top_cache = json.load(f)
    is_potential_top_cache_dirty = False
    for i, row in tqdm(df.iterrows(), total=len(df), desc="Loading VeriThoughts dataset"):
        spec = row["question"]
        rtl = row["ground_truth"]
        verified = row["verified"]
        spec_files = []
        if verified:
            spec_files.append(DataFile(name="design_requirements.txt", content=spec))
        rtl_files = [DataFile(name="dut.sv", content=rtl)]
        id = str(i)
        if id in rtl_tokens_cache:
            rtl_tokens = rtl_tokens_cache[id]
        else:
            rtl_tokens = _compute_rtl_tokens(rtl_files)
            rtl_tokens_cache[id] = rtl_tokens
            is_rtl_tokens_cache_dirty = True
        if id in potential_top_cache:
            potential_top = potential_top_cache[id]
        else:
            potential_top = list(extract_potential_top([rtl]))
            potential_top_cache[id] = potential_top
            is_potential_top_cache_dirty = True
        context = DataContext(
            id=id,
            rtl_files=rtl_files,
            spec_files=spec_files,
            dataset_name=ds_name,
            rtl_tokens=rtl_tokens,
            potential_top=potential_top,
        )
        ret.append(context)
    if is_rtl_tokens_cache_dirty:
        with gzip.open(rtl_tokens_cache_path, "wt") as f:
            json.dump(rtl_tokens_cache, f)
    if is_potential_top_cache_dirty:
        with gzip.open(potential_top_cache_path, "wt") as f:
            json.dump(potential_top_cache, f)
    return ret


def load_cvdp_ecov_dataset(split: str = "eval") -> list[DataContext]:
    ds_name = "hez2024/cvdp_ecov_eval"
    data_ds = ds.load_dataset(ds_name, split=split)
    df = pd.DataFrame(data_ds)
    ret = []
    for _, row in tqdm(df.iterrows(), total=len(df), desc="Loading EcoV dataset"):
        spec_files = [DataFile(name=x["name"], content=x["content"]) for x in row["spec_files"]]
        rtl_files = [DataFile(name=x["name"], content=x["content"]) for x in row["rtl_files"]]
        id = str(row["id"])
        rtl_tokens = _compute_rtl_tokens(rtl_files)
        potential_top = [row["dut_module_name"]]
        dut_top_module_name = row["dut_module_name"]
        dut_top_instance_name = row["dut_instance_name"]
        instructions = row["instruction"]
        targets: list[CovExpectation] = [CovExpectation(**t) for t in row["targets"]]

        context = LlmGenTbContext(
            id=id,
            rtl_files=rtl_files,
            spec_files=spec_files,
            dataset_name=ds_name,
            rtl_tokens=rtl_tokens,
            potential_top=potential_top,
            misc={
                "targets": targets,
                "difficulty": row["difficulty"],
                "is_agentic": row["is_agentic"],
            },
            instructions=instructions,
            dut_top_module_name=dut_top_module_name,
            dut_top_instance_name=dut_top_instance_name,
        )
        ret.append(context)
    return ret


def load_local_parquet(path: str, split: str = "train") -> list[DataContext]:
    """Load a local .parquet file, auto-detecting CodeV-R1 or CVDP-ECov format by columns."""
    df = pd.read_parquet(path)
    cols = set(df.columns)
    ds_name = Path(path).stem  # use stem, not full path (avoids absolute path overriding LOCAL_TMP_DIR in remote_sync)

    if "problem_id" in cols and "question" in cols and "ground_truth" in cols:
        # ── CodeV-R1 format ──────────────────────────────────────────────────
        ret = []
        stem = Path(path).stem
        rtl_tokens_cache_path = CACHE_PATH / f"local_{stem}_rtl_tokens.json.gz"
        rtl_tokens_cache: dict[str, int] = {}
        if rtl_tokens_cache_path.exists():
            with gzip.open(rtl_tokens_cache_path, "rt") as f:
                rtl_tokens_cache = json.load(f)
        is_rtl_dirty = False
        potential_top_cache_path = CACHE_PATH / f"local_{stem}_potential_top.json.gz"
        potential_top_cache: dict[str, list[str]] = {}
        if potential_top_cache_path.exists():
            with gzip.open(potential_top_cache_path, "rt") as f:
                potential_top_cache = json.load(f)
        is_pt_dirty = False
        for _, row in tqdm(df.iterrows(), total=len(df), desc=f"Loading {Path(path).name}"):
            question = _parse_if_str(row["question"])
            ground_truth = _parse_if_str(row["ground_truth"])
            spec = question[1]["content"] if len(question) > 1 else question[0]["content"]
            # numpy.ndarray (from parquet) and list both support [0]["content"] indexing
            rtl = ground_truth[0]["content"]
            spec_files = [DataFile(name="design_requirements.txt", content=spec)]
            rtl_files = [DataFile(name="dut.sv", content=rtl)]
            id_ = str(row["problem_id"])
            if id_ in rtl_tokens_cache:
                rtl_tokens = rtl_tokens_cache[id_]
            else:
                rtl_tokens = _compute_rtl_tokens(rtl_files)
                rtl_tokens_cache[id_] = rtl_tokens
                is_rtl_dirty = True
            if id_ in potential_top_cache:
                potential_top = potential_top_cache[id_]
            else:
                potential_top = list(extract_potential_top([rtl]))
                potential_top_cache[id_] = potential_top
                is_pt_dirty = True
            # Build misc dict: include targets if present in this row
            _misc: dict[str, Any] = {}
            if "targets" in df.columns:
                _targets_raw = _parse_if_str(row["targets"]) if "targets" in row else []
                if _targets_raw:
                    _misc["targets"] = [
                        CovExpectation(**t) if isinstance(t, dict) else t
                        for t in _targets_raw
                    ]

            if _misc.get("targets"):
                # Has coverage targets: create DataContext with misc, then convert to
                # LlmGenTbContext so eval_cov_result_against_expectations can use targets
                _dc = DataContext(
                    id=id_,
                    rtl_files=rtl_files,
                    spec_files=spec_files,
                    dataset_name=ds_name,
                    rtl_tokens=rtl_tokens,
                    potential_top=potential_top,
                    misc=_misc,
                )
                ret.append(data_context_to_llm_gen_tb_context(_dc))
            else:
                ret.append(DataContext(
                    id=id_,
                    rtl_files=rtl_files,
                    spec_files=spec_files,
                    dataset_name=ds_name,
                    rtl_tokens=rtl_tokens,
                    potential_top=potential_top,
                ))
        if is_rtl_dirty:
            with gzip.open(rtl_tokens_cache_path, "wt") as f:
                json.dump(rtl_tokens_cache, f)
        if is_pt_dirty:
            with gzip.open(potential_top_cache_path, "wt") as f:
                json.dump(potential_top_cache, f)
        return ret

    elif "id" in cols and "rtl_files" in cols and "instruction" in cols:
        # ── CVDP-ECov format ─────────────────────────────────────────────────
        ret = []
        for _, row in tqdm(df.iterrows(), total=len(df), desc=f"Loading {Path(path).name}"):
            spec_files_raw = _parse_if_str(row["spec_files"])
            rtl_files_raw = _parse_if_str(row["rtl_files"])
            spec_files = [DataFile(name=x["name"], content=x["content"]) for x in spec_files_raw]
            rtl_files = [DataFile(name=x["name"], content=x["content"]) for x in rtl_files_raw]
            id_ = str(row["id"])
            rtl_tokens = _compute_rtl_tokens(rtl_files)
            potential_top = [row["dut_module_name"]]
            targets_raw = _parse_if_str(row["targets"])
            targets: list[CovExpectation] = [CovExpectation(**t) for t in targets_raw]
            ret.append(LlmGenTbContext(
                id=id_,
                rtl_files=rtl_files,
                spec_files=spec_files,
                dataset_name=ds_name,
                rtl_tokens=rtl_tokens,
                potential_top=potential_top,
                misc={
                    "targets": targets,
                    "difficulty": row["difficulty"],
                    "is_agentic": row["is_agentic"],
                },
                instructions=row["instruction"],
                dut_top_module_name=row["dut_module_name"],
                dut_top_instance_name=row["dut_instance_name"],
            ))
        return ret

    else:
        raise ValueError(
            f"Cannot detect format for local parquet {path!r}: columns={sorted(cols)}"
        )
