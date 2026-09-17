from datetime import datetime
from pathlib import Path
from time import sleep
from typing import Any, Generic, TypeVar, cast

from openai import APIConnectionError, OpenAI, PermissionDeniedError
from pydantic import BaseModel, ValidationError
import requests

from schemas.config_schema import LLMService, find_base_model
from schemas.llm_service_state_schema import ScadsAIModelsStatus


T = TypeVar("T", bound=BaseModel)
class LLM_API(Generic[T]):
    def __init__(self, config: T, model="") -> None:
        self.config = config

        self.llm_service = find_base_model(config, LLMService)

        self.base_url = self.llm_service.api.base_url
        self.meta_data: dict = {}

        if not self.llm_service.api.key_value:
            key_location = str(self.llm_service.api.key_location)

            path = Path(key_location).expanduser()
            with path.open("r", encoding="utf-8") as f:
                self.llm_key = f.read().strip()
        else:
            self.llm_key = self.llm_service.api.key_value

        self.model = model


    def check_model_status(self, model, hasPermission=True) -> tuple[bool, str, float]:
        if not self.llm_service.status.pre_check_connection or not hasPermission:
            return (False, "", 0)

        try:
            response = requests.get(self.llm_service.status.url).json()
        except requests.exceptions.RequestException as e:
            print(f"Connection to LLM status page failed: {e}")
            return (False, "", 0)

        try:
            data = ScadsAIModelsStatus.model_validate(response)
        except ValidationError as e:
            print(e.errors())
            return (False, "", 0)

        if self.llm_service.status.type in data.models.keys():
            for i in data.models[self.llm_service.status.type]:
                if i.real_name==model:
                    return (
                        i.state.lower()=="up",
                        i.state,
                        i.time_taken
                    )
            else:
                raise ValueError("Model not in list")
        else:
            raise ValueError("LLM type is not in list")


    def request(self, rule, prompt, timeout):
        client = OpenAI(
            base_url=self.base_url,
            api_key=self.llm_key,
            timeout=timeout
        )

        success, status, time_taken = self.check_model_status(model=self.model)
        if success:
            print(f"Ping: {self.model} [{status}] ({time_taken}s)")

            try:
                response = client.responses.create(
                    model=self.model,
                    instructions=rule,
                    input=prompt,
                    timeout=200,
                    temperature=1
                )
                return response.output_text

            except APIConnectionError as e:
                print(e)
                return ""
            except PermissionDeniedError as p:
                print(p)
                return ""
        else:
            print("Connection to API failed")
            return ""


    def request_stream(self, rule, prompt, timeout, check_for_alt_models: bool = True):
        wait = 10
        attempts = 10

        client = OpenAI(
            base_url=self.base_url,
            api_key=self.llm_key,
            timeout=timeout,
            max_retries=5
        )
        models = [self.model]
        if check_for_alt_models:
            models.extend(self.llm_service.alt_models)

        time = datetime.now()

        for model in models:
            if not model:
                continue

            try:
                success, status, time_taken = self.check_model_status(model)
            except ValueError as e:
                success, status, time_taken = False, 'not-found', 0

            if success:
                print(f"Ping: {model} [{status}] ({time_taken}s)")

                self.meta_data["model_name"] = model

                stream = None
                for attempt in range(attempts):
                    try:
                        responses = client.responses
                        stream = responses.create(
                            model=model,
                            instructions=rule,
                            input=prompt,
                            stream=True
                        )
                    except (APIConnectionError, PermissionDeniedError) as err:
                        if attempt < attempts - 1:
                            print(f"Process failed, Reason: {err} \nWaiting {wait * (attempt + 1)}s and Retrying...")
                            sleep(wait * (attempt + 1))
                        else:
                            raise

                if not stream:
                    return ""

                for ev in stream:
                    event = cast(Any, ev)

                    match event.type:
                        case "response.output_text.delta":
                            yield event.delta

                        case "response.output_text.done":
                            # print("\nText complete")
                            pass

                        case "response.completed":
                            response = event.response
                            max_tokens = response.max_output_tokens
                            if max_tokens:
                                self.meta_data["max_tokens"] = max_tokens

                            temp = response.temperature
                            if max_tokens:
                                self.meta_data["temperature"] = temp

                            usage = response.usage
                            if usage:
                                self.meta_data["input_tokens"] = usage.input_tokens
                                self.meta_data["output_tokens"] = usage.output_tokens
                                self.meta_data["total_tokens"] = usage.total_tokens
                            break

                        case "response.error":
                            self.meta_data["error_msg"] = event.error
                            print(f"ERROR: {event.error}")
                            continue
                        case _:
                            pass
            else:
                print(f"Ping Failed: {model} [{status}] ({time_taken}s). Looking for alternative Model...")
                continue

            self.meta_data["response_time_seconds"] = round((datetime.now() - time).total_seconds(), 2)

            return ""
        else:
            print("Connection to API failed")
            return ""


    def request_contextualized_stream(self):

        # System instructions
        # Conversation summary
        # Last 10 messages
        # Relevant retrieved messages
        # Current user message

        pass


