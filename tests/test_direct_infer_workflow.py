import asyncio
import random as rd

from openai import AsyncOpenAI

from llm4cov.datasets.filter import filter_data_by_length, filter_single_top_data
from llm4cov.datasets.load import load_dataset_by_name
from llm4cov.datasets.types import LlmGenTbContext, data_context_to_llm_gen_tb_context
from llm4cov.eda_client.remote_exec import run_remote_cov_job_pipeline
from llm4cov.llm_query.formatted_query import OpenAIQueryArgs, query_one_file
from llm4cov.llm_query.prompt_build import build_initial_prompt_from_context

SERVER = "wolverine_centos"
MODEL = "Qwen/Qwen3-Coder-30B-A3B-Instruct"
REMOTE_REPO_DIR = "/workspace/llm4cov_eda"


def load_sample_dataset() -> list[LlmGenTbContext]:
    codev_r1_dataset = load_dataset_by_name("zhuyaoyu/CodeV-R1-dataset", split="train")
    target_dataset_raw = filter_single_top_data(codev_r1_dataset)
    target_dataset_raw = filter_data_by_length(
        target_dataset_raw, max_rtl_length=800, keep_long=False
    )
    # random pick 2 samples for smoke test
    target_dataset_raw = rd.sample(target_dataset_raw, k=2)
    target_dataset = [data_context_to_llm_gen_tb_context(ctx) for ctx in target_dataset_raw]

    verithought_dataset = load_dataset_by_name("wilyub/VeriThoughtsTrainSet", split="train")
    target_dataset_raw = filter_single_top_data(verithought_dataset)
    target_dataset_raw = filter_data_by_length(
        target_dataset_raw, max_rtl_length=800, keep_long=False
    )
    target_dataset_raw = rd.sample(target_dataset_raw, k=2)  # Small sample for smoke test
    target_dataset += [data_context_to_llm_gen_tb_context(ctx) for ctx in target_dataset_raw]

    return target_dataset


async def direct_infer_workflow_sequential() -> None:
    target_dataset = load_sample_dataset()
    client = AsyncOpenAI(base_url="http://localhost:8000/v1", api_key="EMPTY")

    for context in target_dataset:
        print(f"=== Processing context ID: {context.id} in dataset {context.dataset_name} ===")
        messages = build_initial_prompt_from_context(context)
        query_args = OpenAIQueryArgs(
            client=client,
            model=MODEL,
            temperature=0.0,
            max_completion_tokens=131072,
            timeout_seconds=180,
        )

        # Key LLM query step: should be parallelized in real use cases
        tb_file, stats = await query_one_file(
            messages,
            query_args,
            max_retries=1,
            debug=False,
        )

        if tb_file is None:
            print(f"❌ Failed to generate TB for context ID: {context.id}")
            continue
        else:
            print(f"✅ Generated TB file '{tb_file.name}' for context ID: {context.id}")
            print(f"   Stats: {stats}")

        # Key remote coverage job step: should be parallelized in real use cases
        result = run_remote_cov_job_pipeline(
            server=SERVER,
            eda_repo_dir=REMOTE_REPO_DIR,
            context=context,
            tb_file=tb_file,
            timeout=180,
        )

        print(f"   Coverage result status: {result.get('status')}")


def main() -> None:
    asyncio.run(direct_infer_workflow_sequential())


if __name__ == "__main__":
    main()
