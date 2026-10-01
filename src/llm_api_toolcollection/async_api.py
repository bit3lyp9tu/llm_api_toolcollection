import asyncio
from pathlib import Path
from typing import Any

from openai import AsyncOpenAI

from llm_api_toolcollection.config_parser import YAMLConfig
from llm_api_toolcollection.schemas.config_schema import API, find_child


sem = asyncio.Semaphore(10)
async def embed_batch(config: YAMLConfig, text, timeout=60, dimensions=2560):
    api_config = find_child(config.config, API)

    # TODO: should move to schema???
    if not api_config.key_value:
        key_location = str(api_config.key_location)

        path = Path(key_location).expanduser()
        with path.open("r", encoding="utf-8") as f:
            llm_key = f.read().strip()
    else:
        llm_key = api_config.key_value

    client = AsyncOpenAI(
        base_url=api_config.base_url,
        api_key=llm_key,
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



