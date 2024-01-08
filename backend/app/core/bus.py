import asyncio
import os
from abc import abstractmethod
from collections.abc import Callable
from typing import ClassVar

from pydantic import BaseModel
from app.core.logger import logger


def handler(func: Callable[[any], any]) -> Callable[[any], any]:
    def wrapper(*args: any, **kwargs: any) -> any:
        try:
            return func(*args, **kwargs)
        except (KeyError, StopIteration, ):
            attr = func.__name__.replace("get_", "").capitalize()
            entity = args[1]
            logger.warning(f"{attr} {entity} not found")

    return wrapper


class Adapter:
    @abstractmethod
    def init(self) -> None:
        pass

    @abstractmethod
    def healthcheck(self) -> bool:
        pass

    @abstractmethod
    def close(self) -> None:
        pass


class BackgroundTasks:
    running_tasks: ClassVar[set[any]] = set()

    def add_task(self, func: Callable[[any], any], *args: any, **kwargs: any) -> None:
        task = asyncio.create_task(func(*args, **kwargs))
        self.running_tasks.add(task)
        task.add_done_callback(lambda t: self.running_tasks.remove(t))


class Bus:
    parameters: dict[str, dict[str, any]]
    adapters: list[Adapter]
    background_tasks: dict[str, BackgroundTasks]

    def __init__(self) -> None:
        self.parameters = {}
        self.adapters = []
        self.background_tasks = {
            "default": BackgroundTasks(),
        }

    async def init(self) -> None:
        await asyncio.gather(*(adapter.init() for adapter in self.adapters))

    async def healthcheck(self) -> bool:
        return all(await asyncio.gather(*(adapter.healthcheck() for adapter in self.adapters)))

    async def close(self) -> None:
        await asyncio.gather(*(adapter.close() for adapter in self.adapters))

    def register_adapter(self, adapter: Adapter | list[Adapter]) -> None:
        self.adapters.extend(adapter if isinstance(adapter, list) else [adapter])

    def register_parameters(self, parameters: dict[str, any], source: any = None) -> None:
        source = {k.lower(): v for k, v in dict(source or os.environ).items()}
        parameters = {
            name: param.model_validate(source).model_dump(mode="json")
            for name, param in parameters.items()
        }
        self.parameters.update(parameters)

    def register_background_tasks(self, background_tasks: dict[str, BackgroundTasks],) -> None:
        self.background_tasks.update(background_tasks)

    @handler
    def get_adapter(self, adapter: str) -> Adapter:
        return next(ad for ad in self.adapters if adapter.__class__.__name__ == adapter)

    @handler
    def get_parameters(self, entity: str = "llm_parameters") -> dict[str, any]:
        return self.parameters[entity]

    @handler
    def get_background_tasks(self, engine: str = "default") -> BackgroundTasks:
        return self.background_tasks[engine]


BUS = Bus()

