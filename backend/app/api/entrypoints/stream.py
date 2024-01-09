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


@router.post("/grammar")
async def grammar(
    text: str,
    model: OpenAIModels = OpenAIModels.GPT3,
    bus: Bus = Depends(get_bus),
) -> StreamingResponse:
    """
    Entrypoint to correct grammar.

    Args:
        text: The text to be processed.
        model: The model to be used.
        bus: The bus instance.

    Returns:
        StreamingResponse: The streaming response.
    """
    # Variables
    context_variable_name, multiple = "input", False
    run_collector = RunCollectorCallbackHandler()
    runnable_config = RunnableConfig(callbacks=[run_collector])
    run_collector.traced_runs = []

    # Instances
    service = TeacherService()
    callback = AsyncIteratorCallbackHandler()
    llm_params = bus.parameters.get("llm_params", {})
    llm_params.update({
        "model": model,
        "stream": True,
        "callbacks": [callback],
    })
    chain = service.create_grammar_chain(llm_params=llm_params, chain=True)
    _stream = service.stream(
        input=text,
        chain=chain,
        callback=callback,
        config=runnable_config,
        context_variable_name=context_variable_name,
        multiple=multiple,
    )

    return StreamingResponse(_stream, media_type="text/event-stream")


@router.post("/style")
async def style(
    text: str,
    model: OpenAIModels = OpenAIModels.GPT3,
    style_type: StyleTypes = StyleTypes.FREE,
    style_context: str | None = None,
    style_rules: list[str | None] = Query(default="The first rule of the fight club ..."),
    webpage: AnyHttpUrl | None = Query(None),
    file: UploadFile | None = File(None),
    bus: Bus = Depends(get_bus),
) -> StreamingResponse:
    """
    Entrypoint to style the input text.

    Args:
        text: The text to be processed.
        model: The model to be used.
        style_type: The style type.
        style_context: The style context.
        style_rules: The style rules.
        webpage: The webpage to be used.
        file: The file to be processed.
        bus: The bus instance.

    Returns:
        StreamingResponse: The streaming response.
    """
    # Variables
    context_variable_name, multiple = "input", True
    file = (await file.read()).decode("utf-8") if file else None
    run_collector = RunCollectorCallbackHandler()
    runnable_config = RunnableConfig(callbacks=[run_collector])
    run_collector.traced_runs = []

    # Instances
    service = TeacherService()
    callback = AsyncIteratorCallbackHandler()
    llm_params = bus.parameters.get("llm_params", {})
    llm_params.update({"model": model,})
    chain = service.create_style_chain(
        llm_params=llm_params,
        style_type=style_type,
        style_context=style_context,
        style_rules=style_rules,
        style_webpage=webpage,
        style_file=file,
    )

    _stream = service.stream(
        input=text,
        chain=chain,
        callback=callback,
        config=runnable_config,
        context_variable_name=context_variable_name,
        multiple=multiple,
    )

    return StreamingResponse(_stream, media_type="text/event-stream")


@router.post("/summarize")
async def summarization(
    text: str,
    model: OpenAIModels = OpenAIModels.GPT3,
    summarization_type: SummarizationTypes = SummarizationTypes.DEFAULT,
    webpage: AnyHttpUrl | None = Query(None),
    file: UploadFile | None = File(None),
    bus: Bus = Depends(get_bus),
) -> StreamingResponse:
    """
    Endpoint to handle chat.

    Args:
        text: The text to be processed.
        model: The model to be used.
        summarization_type: The summarization type.
        webpage: The webpage to be used.
        file: The file to be processed.
        bus: The bus instance.

    Returns:
        StreamingResponse: The streaming response.
    """
    # Variables
    context_variable_name, multiple = "input", False
    file = (await file.read()).decode("utf-8") if file else None
    run_collector = RunCollectorCallbackHandler()
    runnable_config = RunnableConfig(callbacks=[run_collector])
    run_collector.traced_runs = []

    # Instances
    service = TeacherService()
    callback = AsyncIteratorCallbackHandler()
    llm_params = bus.parameters.get("llm_params", {})
    llm_params.update({"model": model,})
    chain = service.create_summarization_chain(
        llm_params=llm_params,
        summarization_type=summarization_type,
        summarization_webpage=webpage,
        summarization_file=file,
    )
    if summarization_type == SummarizationTypes.WEBPAGE and webpage:
        text = service.document_loader(urls=[str(webpage)])
        context_variable_name = "input_documents"
    elif summarization_type == SummarizationTypes.DOCUMENT and file:
        text = service.document_loader(text=file)
        context_variable_name = "input_documents"
        multiple = True

    _stream = service.stream(
        input=text,
        chain=chain,
        callback=callback,
        config=runnable_config,
        context_variable_name=context_variable_name,
        multiple=multiple,
    )

    return StreamingResponse(_stream, media_type="text/event-stream")


@router.post("/")
async def stream(
    text: str,
    model: OpenAIModels = OpenAIModels.GPT3,
    action: TeacherActions = TeacherActions.GRAMMAR,
    summarization_type: SummarizationTypes = SummarizationTypes.DEFAULT,
    style_type: StyleTypes = StyleTypes.FREE,
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
        model: The model to be used.
        action: The action to be performed.
        summarization_type: The summarization type.
        style_type: The style type.
        style_context: The style context.
        style_rules: The style rules.
        webpage: The webpage to be used.
        file: The file to be processed.
        bus: The bus instance.

    Returns:
        StreamingResponse: The streaming response.
    """
    # Variables
    context_variable_name, multiple = "input", False
    file = (await file.read()).decode("utf-8") if file else None
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
                style_webpage=webpage,
                style_file=file,
            )
            multiple = True
        case TeacherActions.SUMMARIZATION:
            chain = service.create_summarization_chain(
                llm_params=llm_params,
                summarization_type=summarization_type,
                summarization_webpage=webpage,
                summarization_file=file,
            )
            if summarization_type == SummarizationTypes.WEBPAGE and webpage:
                text = service.document_loader(urls=[str(webpage)])
                context_variable_name = "input_documents"
            elif summarization_type == SummarizationTypes.DOCUMENT and file:
                text = service.document_loader(text=file)
                context_variable_name = "input_documents"
                multiple = True

    _stream = service.stream(
        input=text,
        chain=chain,
        callback=callback,
        config=runnable_config,
        context_variable_name=context_variable_name,
        multiple=multiple,
    )

    return StreamingResponse(_stream, media_type="text/event-stream")
