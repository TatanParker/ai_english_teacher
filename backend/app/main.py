from collections.abc import AsyncGenerator
from contextlib import asynccontextmanager

from fastapi import FastAPI

from app.api.router import api_router
from app.core.bus import Bus, BUS
from app.core.settings import settings
from app.core.schemas import LLMParams


# Instantiate Bus and Register Adapters and Parameters (Dependency Injection)
bus = BUS
parameters = {"llm_params": LLMParams}
bus.register_parameters(parameters, source=settings)


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    # Init bus
    bus: Bus = app.extra["bus"]  # type: ignore
    await bus.init()
    if not await bus.healthcheck():
        raise Exception("Bus is not healthy")
    # Run app
    yield
    # Close bus
    await bus.close()


# Initiate API

app = FastAPI(
    title=settings.PROJECT_NAME,
    description=settings.PROJECT_DESCRIPTION,
    version=settings.PROJECT_VERSION,
    lifespan=lifespan,
    bus=bus,
)

app.include_router(api_router)
