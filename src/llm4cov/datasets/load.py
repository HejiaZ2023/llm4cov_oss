import gzip
import json
from pathlib import Path
from typing import Any

import datasets as ds
import pandas as pd
from tqdm import tqdm
from transformers import AutoTokenizer

from llm4cov.datasets.rtl_preprocess import extract_potential_top
from llm4cov.datasets.types import CovExpectation, DataContext, DataFile, LlmGenTbContext

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


def load_dataset_by_name(dataset_name: str, split: str = "train") -> list[DataContext]:
    CACHE_PATH.mkdir(parents=True, exist_ok=True)
    if dataset_name == "zhuyaoyu/CodeV-R1-dataset":
        return load_codev_r1_dataset(split)
    elif dataset_name == "wilyub/VeriThoughtsTrainSet":
        return load_verithoughts_dataset(split)
    elif dataset_name == "hez2024/cvdp_ecov_eval":
        return load_cvdp_ecov_dataset(split)
    else:
        raise ValueError(f"Unsupported dataset: {dataset_name}")


def _compute_rtl_tokens(rtl_files: list[DataFile]) -> int:
    tokenizer = construct_tokenizer()
    total_length = sum(len(tokenizer.encode(rtl_file.content)) for rtl_file in rtl_files)
    return total_length


def load_codev_r1_dataset(split: str = "train") -> list[DataContext]:
    ds_name = "zhuyaoyu/CodeV-R1-dataset"
    data_ds = ds.load_dataset(
        ds_name, name="sft", revision="ffc4698071098044c72bde14fdad309eb3a1c5da", split=split
    )
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
