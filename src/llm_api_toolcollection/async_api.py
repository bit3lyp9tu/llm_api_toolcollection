import asyncio
from typing import Any
from llm_api_toolcollection.schemas.config_schema import API
from openai import AsyncOpenAI

from llm_api_toolcollection.api import LLM_API


sem = asyncio.Semaphore(10)
async def embed_batch(api_config: API, text, timeout=60, dimensions=2560):
    client = AsyncOpenAI(
        base_url=api_config.base_url,
        api_key=api_config.key_value,
        timeout=timeout
    )
    # TODO: Catch this: openai.RateLimitError: Error code: 429 - {'error': {'message': 'litellm.RateLimitError: Rate limit exceeded for model_per_key: xxx:Qwen/Qwen3-Embedding-4B. Limit type: requests. Current limit: 150, Remaining: 0. Limit resets at: 2026-09-29 09:20:44 UTC', 'type': 'None', 'param': 'None', 'code': '429'}}
    async with sem:
        response = await client.embeddings.create(
            model="Qwen/Qwen3-Embedding-4B",
            input=text,
            # TODO (no support for matryoshka representation): dimensions=dimensions
        )
        return [x.embedding for x in response.data]



