from fastapi.requests import Request

from app.core.bus import Bus


def get_bus(request: Request) -> Bus:
    return request.app.extra["bus"]
