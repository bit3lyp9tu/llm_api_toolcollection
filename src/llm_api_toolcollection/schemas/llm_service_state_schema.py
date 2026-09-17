from pydantic import BaseModel


class Model(BaseModel):
    name: str
    real_name: str
    state: str
    time_taken: float
    max_tokens: int
    supports_reasoning: bool | None = None
    supports_vision: bool | None = None
    supports_tools: bool | None = None

class ScadsAIModelsStatus(BaseModel):
    timestamp: str
    models: dict[str, list[Model]]
    all_up: bool
