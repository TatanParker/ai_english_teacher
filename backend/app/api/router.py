from fastapi import APIRouter

from app.api.entrypoints.stream import router as stream_router
from app.api.entrypoints.ingest import router as ingest_router


api_router = APIRouter(
    prefix="/api",
)

api_router.include_router(stream_router)
api_router.include_router(ingest_router)
