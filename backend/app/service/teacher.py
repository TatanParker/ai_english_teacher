from langchain.prompts import PromptTemplate
from langchain.schema.output_parser import StrOutputParser
from langchain.chains.base import Chain
from langchain.chains import SimpleSequentialChain

from app.core.enums import StyleTypes, SummarizationTypes
from app.core.prompts_templates import (
    grammar_template,
    style_template,
    summarization_template_default,
    summarization_template_basic,
)
from app.service.chain import ChainService


class TeacherService(ChainService):

    @staticmethod
    def inject_context(context: str, template: str) -> str:
        assert context, "Context must be provided"
        return template.replace("{context}", context)

    @staticmethod
    def build_rules(rules: list[str]) -> str:
        header = "In the following text look follow the following rules: \n"
        rules = "\n".join([f"- {rule}" for rule in rules])
        return header + rules

    def create_grammar_chain(
        self,
        llm_params: dict[str, any],
        chain: bool = False,
    ) -> Chain:
        llm = self.create_llm(**llm_params)
        prompt = self.create_prompt_template(
            resources=grammar_template,
            prompt_template=PromptTemplate,
        )
        chain = self.create_chain(
            llm=llm,
            prompt=prompt,
            output_parser=StrOutputParser(),
            chain=chain
        )
        return chain

    def create_style_chain(
        self,
        llm_params: dict[str, any],
        style_type: StyleTypes,
        style_context: str | None = None,
        style_rules: list[str] | None = None,
    ) -> Chain:
        grammar_chain = self.create_grammar_chain(llm_params=llm_params, chain=True)
        llm = self.create_llm(**llm_params)
        template = style_template
        match style_type:
            case StyleTypes.FREE:
                template = self.inject_context(
                    context="Let yourself get inspired by the randomness of the AI.",
                    template=style_template,
                )
            case StyleTypes.CONCRETE:
                template = self.inject_context(
                    context=style_context,
                    template=style_template,
                )
            case StyleTypes.RULES:
                template = self.inject_context(
                    context=self.build_rules(style_rules),
                    template=style_template,
                )
        prompt = self.create_prompt_template(
            resources=template,
            prompt_template=PromptTemplate,
        )
        style_chain = self.create_chain(
            llm=llm,
            prompt=prompt,
            output_parser=StrOutputParser(),
            chain=True,
        )
        chain = SimpleSequentialChain(
            chains=[grammar_chain, style_chain],
        )
        return chain

    def create_summarization_chain(
        self,
        llm_params: dict[str, any],
        summarization_type: SummarizationTypes = SummarizationTypes.DEFAULT,
        chain: bool = False,
    ) -> Chain:
        llm = self.create_llm(**llm_params)
        template = summarization_template_default
        match summarization_type:
            case SummarizationTypes.BASIC:
                template = summarization_template_basic
            case SummarizationTypes.DEFAULT:
                template = summarization_template_default
        prompt = self.create_prompt_template(
            resources=template,
            prompt_template=PromptTemplate,
        )
        chain = self.create_chain(
            llm=llm,
            prompt=prompt,
            output_parser=StrOutputParser(),
            chain=chain
        )
        return chain
