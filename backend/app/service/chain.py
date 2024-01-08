import asyncio
from typing import AsyncIterable

from langchain.chains.base import Chain
from langchain.chains import LLMChain, SimpleSequentialChain
from langchain.llms import OpenAI
from langchain.callbacks import AsyncIteratorCallbackHandler
from langchain_core.language_models.llms import BaseLLM
from langchain.prompts import BasePromptTemplate, BaseChatPromptTemplate, PromptTemplate
from langchain.schema.runnable import Runnable
from langchain.schema.output_parser import BaseOutputParser, StrOutputParser
from langchain.callbacks.base import BaseCallbackHandler


class ChainService:
    """Class for chain and llm services."""

    @staticmethod
    def create_llm(
        stream: bool = False,
        callbacks: list[BaseCallbackHandler] | None = None,
        **parameters
    ) -> BaseLLM:
        """
        Creates an instance of the LLM.

        Args:
            stream: Whether to stream the LLM response.
            callbacks: The callback handlers (optional).
            **parameters: Additional parameters for the LLM model.

        Returns:
            BaseLLM: The instantiated LLM.
        """
        if stream:
            parameters.update(
                {
                    "streaming": True,
                    "callbacks": callbacks,
                }
            )
        return OpenAI(**parameters)

    @staticmethod
    def create_prompt_template(
        resources: str | list[tuple[str, str]],
        prompt_template: BaseChatPromptTemplate | BasePromptTemplate = PromptTemplate,
    ) -> BasePromptTemplate | BaseChatPromptTemplate:
        """
        Creates an instance of the prompt template.

        Args:
            resources: The prompt template resources.
            prompt_template: The prompt template object.

        Returns:
            BasePromptTemplate: The instantiated prompt template.
        """
        if isinstance(resources, str):
            return prompt_template.from_template(resources)
        return prompt_template.from_messages(resources)

    @staticmethod
    def create_chain(
        llm: BaseLLM,
        prompt: BasePromptTemplate,
        output_parser: BaseOutputParser = StrOutputParser,
        chain: bool = False,
    ) -> Runnable | LLMChain:
        """
        Creates a chain based on provided llm.

        Args:
            llm: The instantiated LLM model.
            prompt: The instantiated prompt template.
            output_parser: The output parser object.
            chain: The chain object (optional).

        Returns:
            Runnable: A chained object.
        """
        if not chain:
            runnable = prompt | llm | output_parser
        else:
            runnable = LLMChain(llm=llm, prompt=prompt, output_parser=output_parser)
        return runnable

    @staticmethod
    async def stream(
        input: str,
        chain: Chain,
        callback: AsyncIteratorCallbackHandler,
        config: any
    ) -> AsyncIterable[str]:
        task = asyncio.create_task(
            chain.arun(input=input, config=config)
        )

        async def stream_runner(input: str, chunk_size=25):
            text = await chain.arun({"input": input})
            for i in range(0, len(text), chunk_size):
                yield text[i:i + chunk_size]
                await asyncio.sleep(0.1)
        try:
            iterator = stream_runner(input) if isinstance(chain, SimpleSequentialChain) else callback.aiter()
            async for token in iterator:
                yield token
        finally:
            callback.done.set()
        await task
