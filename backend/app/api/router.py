from fastapi import APIRouter

from app.api.entrypoints.stream import router as stream_router


api_router = APIRouter(
    prefix="/api",
)

api_router.include_router(stream_router)
