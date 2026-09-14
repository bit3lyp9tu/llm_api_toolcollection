from contextlib import contextmanager
import json
import yaml
from typing import IO, Callable, ClassVar, Generic, TypeVar

from pydantic import BaseModel
from pydantic_core import ValidationError


class ConfigError(Exception):
    pass

T = TypeVar("T", bound=BaseModel)

class ConfigBase(Generic[T]):
    schema: type[T]
    loader: ClassVar[Callable[[IO[str]], dict]]
    dumper: ClassVar[Callable[[dict], str]]

    default_path = ""

    def __init__(self, config_path: str | None = None) -> None:
        if config_path:
            self.config_path = config_path
        else:
            if self.default_path:
                self.config_path = self.default_path
            else:
                raise ValueError("No config path defined")

        with open(self.config_path, 'r') as f:
            data = type(self).loader(f)

        try:
            self.config: T = self.schema.model_validate(data)
        except ValidationError as e:
            messages = []

            for err in e.errors():
                location = ".".join(map(str, err["loc"]))
                messages.append(f"{location}: {err['msg']}")

            raise ConfigError(
                f"Configuration file '{self.config_path}' is invalid:\n"
                + "\n".join(messages)
            ) from e

    @contextmanager
    def open(self):
        try:
            yield self.config
        except Exception:
            raise
        finally:
            self.createFile(self.config_path)

    def createFile(self, path):
        with open(path, "w") as w:
            w.write(type(self).dumper(self.config.model_dump()))


S = TypeVar("S", bound=BaseModel)
class YAMLConfig(ConfigBase[S], Generic[S]):
    schema: type[S]

    loader = staticmethod(yaml.safe_load)
    dumper = staticmethod(lambda data: yaml.dump(
        data,
        default_flow_style=False,
        indent=4,
    ))

    def __init__(
        self,
        schema: type[S],
        config_path: str = "config.yaml",
    ) -> None:
        self.schema = schema
        super().__init__(config_path)


