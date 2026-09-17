from pathlib import Path
from typing import Any, Self, TypeVar

from pydantic import BaseModel, field_validator, model_validator

from pathlib import Path


def find_project_root(start: Path) -> Path:
    for path in (start.resolve(), *start.resolve().parents):
        if (path / "pyproject.toml").exists():
            return path
    raise FileNotFoundError("Could not find project root.")

T = TypeVar("T", bound=BaseModel)
def _find_child(
    value: Any,
    target_type: type[T],
    visited: set[int] = set(),
) -> T | None:
    if isinstance(value, target_type):
        return value

    if not isinstance(value, (BaseModel, dict, list, tuple, set)):
        # print("1", value, target_type, visited)
        return None

    value_id = id(value)

    if value_id in visited:
        # print("2", value, target_type, visited)
        return None

    visited.add(value_id)

    if isinstance(value, BaseModel):
        for field_name in type(value).model_fields:
            result = _find_child(
                getattr(value, field_name),
                target_type,
                visited,
            )
            if result is not None:
                return result

    elif isinstance(value, dict):
        for child in value.values():
            result = _find_child(child, target_type, visited)
            if result is not None:
                return result

    else:  # list, tuple, set
        if type(value) == str:
            print(value)
            for child in value:
                print(type(child))
                result = _find_child(child, target_type, visited)
                if result is not None:
                    return result

    # print("3", value, target_type, visited)
    return None

def find_child(
    obj: BaseModel,
    target_type: type[T],
) -> T:
    result = _find_child(obj, target_type)

    if result is None:
        raise ValueError(
            f"No {target_type.__name__} found in {type(obj).__name__}"
        )

    return result


class API(BaseModel):
    base_url: str
    key_value: str | None = None
    key_location: str | None = None
    request_delay_seconds: int = 5

    @model_validator(mode="after")
    def validate_key_present(self) -> Self:
        if self.key_value is None and self.key_location is None:
            raise ValueError("Either 'key_value' or 'key_location' must be provided.")
        return self

    @field_validator("key_location")
    @classmethod
    def validate_path(cls, value: str | None) -> str | None:
        if value is not None:
            path = Path(value).expanduser()
            if not path.exists():
                raise ValueError(f"Path does not exist: {value}")
            if not path.is_file():
                raise ValueError(f"Not a file: {value}")
        return value

class LLMStatus(BaseModel):
    type: str
    pre_check_connection: bool = True
    show_model_response_time: bool = True
    show_model_status: bool = True
    url: str

class LLMService(BaseModel):
    api: API
    status: LLMStatus
    alt_models: list[str] = []



class SysPrompt(BaseModel):
    file_path: str

    # @field_validator("file_path")
    # @classmethod
    # def validate_path(cls, value: str | None) -> str | None:
    #     if value is not None:
    #         path = Path(value)
    #         if not path.exists():
    #             raise ValueError(f"Path does not exist: {value}")
    #         if not path.is_file():
    #             raise ValueError(f"Not a file: {value}")
    #     return value


class Prompts(BaseModel):
    path: str
    header_path: str = ""
    footer_path: str = ""
    # TODO file validation???

class LLM_Log(BaseModel):
    add_thinking_response: bool = True

class Logs(BaseModel):
    llm_log: LLM_Log
