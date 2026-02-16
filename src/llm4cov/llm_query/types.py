import time
from typing import Any

import google.auth
import google.auth.transport.requests
import openai
from openai import AsyncOpenAI
from openai.types.chat import ChatCompletion
from pydantic import BaseModel, field_serializer
from transformers import AutoTokenizer

TOKENIZER_DICT: dict[str, AutoTokenizer] = {}


def _get_tokenizer_for_model(model_or_local_dir: str, trust_remote_code: bool) -> AutoTokenizer:
    global TOKENIZER_DICT
    if model_or_local_dir not in TOKENIZER_DICT:
        start = time.time()
        TOKENIZER_DICT[model_or_local_dir] = AutoTokenizer.from_pretrained(
            model_or_local_dir, trust_remote_code=trust_remote_code
        )
        elapsed = time.time() - start
        if elapsed > 1.0:
            print(f"Tokenizer for {model_or_local_dir} loaded in {elapsed:.1f}s.")
    return TOKENIZER_DICT[model_or_local_dir]


class AsyncOpenAICredentialsRefresher:
    def __init__(self, project_id: str, location: str, **kwargs: Any) -> None:
        # Set a placeholder key here
        endpoint = (
            "aiplatform.googleapis.com"
            if location == "global"
            else f"{location}-aiplatform.googleapis.com"
        )
        base_url = f"https://{endpoint}/v1/projects/{project_id}/locations/{location}/endpoints/openapi/chat/completions?"
        self.client = openai.AsyncOpenAI(**kwargs, api_key="PLACEHOLDER", base_url=base_url)
        self.creds, self.project = google.auth.default(
            scopes=["https://www.googleapis.com/auth/cloud-platform"]
        )

    def __getattr__(self, name: str) -> Any:
        if not self.creds.valid:
            self.creds.refresh(google.auth.transport.requests.Request())

            if not self.creds.valid:
                raise RuntimeError("Unable to refresh auth")

            self.client.api_key = self.creds.token
        return getattr(self.client, name)


class LLMQueryStats(BaseModel):
    prompt_tokens: int
    completion_tokens: int
    reasoning_tokens: int
    total_tokens: int
    latency_seconds: float
    reasoning_content: str = ""

    @field_serializer("latency_seconds")
    def serialize_latency(self, v: float) -> float:
        return round(v, 3)


class OpenAIQueryArgs(BaseModel):
    client: AsyncOpenAI
    model: str
    max_completion_tokens: int
    temperature: float = 1.0
    top_p: float = 1.0
    timeout_seconds: float | None = None
    tokenizer_dir: str | None = None
    use_responses_api: bool = False
    reasoning_delimiter: tuple[str, str] = ("<think>", "</think>")
    use_vertex: bool = False
    vertex_client: AsyncOpenAICredentialsRefresher | None = None

    class Config:
        arbitrary_types_allowed = True


def extract_reasoning_from_text(
    content: str, reasoning_delimiter: tuple[str, str] = ("<think>", "</think>")
) -> str:
    """Extract the <think>...</think> block from the content."""
    start_delim, end_delim = reasoning_delimiter
    start_idx = content.find(start_delim)
    end_idx = content.find(end_delim, start_idx + len(start_delim))
    if start_idx == -1 or end_idx == -1:
        return ""
    return content[start_idx + len(start_delim) : end_idx].strip()


async def openai_query_one_plain(
    args: OpenAIQueryArgs, messages: list[dict], debug: bool = False
) -> tuple[str, LLMQueryStats]:
    if args.use_responses_api:
        return await openai_query_one_plain_responses(args, messages, debug=debug)
    else:
        return await openai_query_one_plain_completion(args, messages, debug=debug)


async def openai_query_one_plain_responses(
    args: OpenAIQueryArgs, messages: list[dict], debug: bool = False
) -> tuple[str, LLMQueryStats]:
    """Send async request via Responses API and return plain text + stats."""
    if args.use_vertex:
        raise NotImplementedError("Vertex AI Responses API not implemented yet.")

    start_time = time.time()

    resp = await args.client.responses.create(
        model=args.model,
        input=messages,
        max_output_tokens=args.max_completion_tokens,
        temperature=args.temperature,
        top_p=args.top_p,
        timeout=args.timeout_seconds,
    )

    end_time = time.time()
    latency = end_time - start_time

    if debug:
        print(resp)

    if resp.incomplete_details and resp.incomplete_details.reason == "max_output_tokens":
        raise RuntimeError(
            f"OpenAI response truncated: reached max_output_tokens={args.max_completion_tokens}."
        )

    # -------- extract text output --------
    # Responses API can return multiple content blocks; we want plain text
    output_text_parts: list[str] = []

    for item in resp.output:
        if item.type == "message":
            for block in item.content:
                if block.type == "output_text":
                    output_text_parts.append(block.text)

    raw_content = "".join(output_text_parts)

    # -------- token accounting --------
    usage = resp.usage

    prompt_tokens = usage.input_tokens or 0
    completion_tokens = usage.output_tokens or 0
    total_tokens = usage.total_tokens or (prompt_tokens + completion_tokens)

    reasoning_tokens = 0
    reasoning_content = extract_reasoning_from_text(raw_content, args.reasoning_delimiter)
    if usage.output_tokens_details:
        reasoning_tokens = usage.output_tokens_details.reasoning_tokens or 0

    # -------- fallback reasoning token inference --------
    if not reasoning_tokens and reasoning_content:
        tokenizer = _get_tokenizer_for_model(
            args.tokenizer_dir or args.model,
            trust_remote_code=bool(args.tokenizer_dir),
        )
        reasoning_tokens = len(tokenizer.encode(reasoning_content))

    stats = LLMQueryStats(
        prompt_tokens=prompt_tokens,
        completion_tokens=completion_tokens,
        reasoning_tokens=reasoning_tokens,
        total_tokens=total_tokens,
        latency_seconds=latency,
        reasoning_content=reasoning_content,
    )

    return raw_content, stats


async def openai_query_one_plain_completion(
    args: OpenAIQueryArgs, messages: list[dict], debug: bool = False
) -> tuple[str, LLMQueryStats]:
    """Send async request and write output to file, with retry/backoff."""
    start_time = time.time()
    client = args.vertex_client if args.use_vertex else args.client
    assert isinstance(client, (AsyncOpenAI, AsyncOpenAICredentialsRefresher))
    resp: ChatCompletion = await client.chat.completions.create(
        messages=messages,
        model=args.model,
        max_completion_tokens=args.max_completion_tokens,
        temperature=args.temperature,
        top_p=args.top_p,
        timeout=args.timeout_seconds,
    )
    end_time = time.time()
    latency = end_time - start_time

    if debug:
        print(resp)

    if resp.choices and resp.choices[0].finish_reason == "length":
        raise RuntimeError(
            "OpenAI response truncated: "
            f"reached max_completion_tokens={args.max_completion_tokens}."
        )

    reasoning_tokens = 0
    reasoning_content = extract_reasoning_from_text(
        resp.choices[0].message.content or "", args.reasoning_delimiter
    )
    if resp.usage.completion_tokens_details:
        reasoning_tokens = resp.usage.completion_tokens_details.reasoning_tokens
    if not reasoning_tokens and reasoning_content:
        model = args.model.removesuffix("-maas") if args.use_vertex else args.model
        tokenizer = _get_tokenizer_for_model(  # trust remote code if using local dir
            args.tokenizer_dir or model, trust_remote_code=bool(args.tokenizer_dir)
        )
        reasoning_tokens = len(tokenizer.encode(reasoning_content))
    stats: LLMQueryStats = LLMQueryStats(
        prompt_tokens=resp.usage.prompt_tokens,
        completion_tokens=resp.usage.completion_tokens,
        reasoning_tokens=reasoning_tokens,
        total_tokens=resp.usage.total_tokens,
        latency_seconds=latency,
        reasoning_content=reasoning_content,
    )
    return resp.choices[0].message.content or "", stats
