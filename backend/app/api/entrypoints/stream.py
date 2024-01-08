import asyncio
from collections.abc import AsyncIterable
from queue import Queue

from fastapi import Depends, Query, UploadFile, File
from fastapi import APIRouter
from fastapi.responses import StreamingResponse
from langchain.schema.runnable import RunnableConfig
from langchain.callbacks import AsyncIteratorCallbackHandler
from langchain.callbacks.tracers.run_collector import RunCollectorCallbackHandler
from pydantic import AnyHttpUrl

from app.core.bus import Bus
from app.core.enums import OpenAIModels, TeacherActions, StyleTypes, SummarizationTypes
from app.service.teacher import TeacherService
from app.api.deps import get_bus


router = APIRouter(
    prefix="/stream",
    tags=["stream"],
)


@router.post("/")
async def stream(
    text: str,
    model: OpenAIModels = OpenAIModels.GPT3,
    action: TeacherActions = TeacherActions.GRAMMAR,
    style_type: StyleTypes = StyleTypes.FREE,
    summarization_type: SummarizationTypes = SummarizationTypes.DEFAULT,
    style_context: str | None = None,
    style_rules: list[str | None] = Query(default="The first rule of the fight club ..."),
    webpage: AnyHttpUrl | None = Query(None),
    file: UploadFile | None = File(None),
    bus: Bus = Depends(get_bus),
) -> StreamingResponse:
    """
    Endpoint to handle chat.

    Args:
        text: The text to be processed.
        bus: The bus instance.

    Returns:
        StreamingResponse: The streaming response.
    """
    ree = await file.read()
    import pdb; pdb.set_trace()
    # Variables
    run_collector = RunCollectorCallbackHandler()
    runnable_config = RunnableConfig(callbacks=[run_collector])
    run_collector.traced_runs = []

    # Instances
    service = TeacherService()
    chain = object()
    callback = AsyncIteratorCallbackHandler()
    llm_params = bus.parameters.get("llm_params", {})
    llm_params.update({
        "model": model,
        "stream": True,
        "callbacks": [callback],
    })
    match action:
        case TeacherActions.GRAMMAR:
            chain = service.create_grammar_chain(llm_params=llm_params, chain=True)
        case TeacherActions.STYLE:
            chain = service.create_style_chain(
                llm_params=llm_params,
                style_type=style_type,
                style_context=style_context,
                style_rules=style_rules,
            )
        case TeacherActions.SUMMARIZATION:
            chain = service.create_summarization_chain(
                llm_params=llm_params,
                summarization_type=summarization_type,
                chain=True,
            )

    _stream = service.stream(
        input=text,
        chain=chain,
        callback=callback,
        config=runnable_config,
    )

    return StreamingResponse(_stream, media_type="text/event-stream")
