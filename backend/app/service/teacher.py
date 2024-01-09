from langchain.prompts import PromptTemplate
from langchain.schema.output_parser import StrOutputParser
from langchain.chains.base import Chain
from langchain.chains import SimpleSequentialChain
from pydantic import AnyHttpUrl

from app.core.enums import StyleTypes, SummarizationTypes, SummarizationChainTypes
import app.core.prompts_templates as tmp
from app.service.chain import ChainService
from app.service.rag import RagService


class TeacherService(RagService, ChainService):
    """Class for teacher services."""

    @staticmethod
    def build_rules(rules: list[str]) -> str:
        """
        Builds the rules for the style chain.

        Args:
            rules: The rules to be built.

        Returns:
            str: The built rules subprompt.
        """
        header = "In the following text look follow the following rules: \n"
        rules = "\n".join([f"- {rule}" for rule in rules])
        return header + rules

    def create_grammar_chain(
        self,
        llm_params: dict[str, any],
        chain: bool = False,
    ) -> Chain:
        """
        Creates a grammar chain.

        Args:
            llm_params: The llm parameters.
            chain: Whether to chain the grammar chain.

        Returns:
            Chain: The grammar chain.
        """
        llm = self.create_llm(**llm_params)
        prompt = self.create_prompt_template(
            resources=tmp.grammar_template,
            prompt_template=PromptTemplate,
        )
        chain = self.create_chain(
            llm=llm, prompt=prompt, output_parser=StrOutputParser(), chain=chain
        )
        return chain

    def create_style_chain(
        self,
        llm_params: dict[str, any],
        style_type: StyleTypes,
        style_context: str | None = None,
        style_rules: list[str] | None = None,
        style_webpage: str | None = None,
        style_file: str | bytes | None = None,
    ) -> Chain:
        """
        Creates a style chain.

        Args:
            llm_params: The llm parameters.
            style_type: The style type.
            style_context: The style context.
            style_rules: The style rules.
            style_webpage: The style webpage.
            style_file: The style file.

        Returns:
            Chain: The style chain.
        """
        grammar_chain = self.create_grammar_chain(llm_params=llm_params, chain=True)
        llm = self.create_llm(**llm_params)
        template = tmp.style_template
        vectorize = False
        chain_type = SummarizationChainTypes.STUFF
        match style_type:
            case StyleTypes.FREE:
                template = self.inject_context(
                    context="Let yourself get inspired by the randomness of the AI.",
                    template=tmp.style_template,
                )
            case StyleTypes.CONCRETE:
                template = self.inject_context(
                    context=style_context,
                    template=tmp.style_template,
                )
            case StyleTypes.RULES:
                template = self.inject_context(
                    context=self.build_rules(style_rules),
                    template=tmp.style_template,
                )
            case StyleTypes.WEBPAGE:
                assert style_webpage, "Webpage must be provided."
                docs = self.document_loader(urls=[str(style_webpage)])
                prompt = self.create_prompt_template(
                    resources=tmp.style_webpage_template
                )
                tokens = self.num_tokens(str(style_webpage))
                if 4400 < tokens < 10000:
                    chain_type = SummarizationChainTypes.MAP_REDUCE
                elif tokens > 10000:
                    vectorize = True
                webpage_style = self.summarize_large_text(
                    docs=docs,
                    llm=llm,
                    chain_type=chain_type,
                    vectorize=vectorize,
                    prompt=prompt,
                    document_variable_name="input",
                )
                template = self.inject_context(
                    context=webpage_style,
                    template=tmp.style_template,
                )
            case StyleTypes.DOCUMENT:
                assert style_file, "File must be provided."
                docs = self.document_loader(
                    text=style_file
                    if isinstance(style_file, str)
                    else style_file.decode("utf-8")
                )
                prompt = self.create_prompt_template(
                    resources=tmp.style_document_template
                )
                tokens = self.num_tokens(str(style_file))
                if 4400 < tokens < 10000:
                    chain_type = SummarizationChainTypes.MAP_REDUCE
                elif tokens > 10000:
                    vectorize = True
                document_style = self.summarize_large_text(
                    docs=docs,
                    llm=self.create_llm(),
                    chain_type=chain_type,
                    vectorize=vectorize,
                    prompt=prompt,
                )
                template = self.inject_context(
                    context=document_style,
                    template=tmp.style_template,
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
        summarization_webpage: AnyHttpUrl | None = None,
        summarization_file: str | None = None,
    ) -> Chain:
        llm = self.create_llm(**llm_params)
        chain_type = SummarizationChainTypes.STUFF
        match summarization_type:
            case SummarizationTypes.BASIC:
                template = tmp.summarization_basic_template
                prompt = self.create_prompt_template(
                    resources=template,
                )
                return self.create_chain(
                    llm=llm,
                    prompt=prompt,
                    output_parser=StrOutputParser(),
                )
            case SummarizationTypes.DEFAULT:
                template = tmp.summarization_default_template
                prompt = self.create_prompt_template(
                    resources=template,
                )
                return self.create_chain(
                    llm=llm,
                    prompt=prompt,
                    output_parser=StrOutputParser(),
                )
            case SummarizationTypes.WEBPAGE:
                assert summarization_webpage, "Webpage must be provided."
                tokens = self.num_tokens(str(summarization_webpage))
                self.session_docs = self.document_loader(
                    urls=[str(summarization_webpage)]
                )
                if 4400 < tokens < 10000:
                    chain_type = SummarizationChainTypes.MAP_REDUCE
                template = tmp.summarization_webpage_template
                prompt = self.create_prompt_template(
                    resources=template,
                )
                return self.create_rag_summarization_chain(
                    llm=llm,
                    chain_type=chain_type,
                    prompt=prompt,
                    document_variable_name="input",
                )
            case SummarizationTypes.DOCUMENT:
                assert summarization_file, "File must be provided."
                self.session_docs = self.document_loader(text=summarization_file)
                tokens = self.num_tokens(self.session_docs)
                summarization_combine_template = tmp.summarization_document_template
                if 4000 < tokens < 10000:
                    chain_type = SummarizationChainTypes.MAP_REDUCE
                elif tokens > 9500:
                    self.session_docs = self.vectorizer(self.session_docs)
                    summarization_combine_template = (
                        tmp.summarization_large_combine_prompt
                    )
                return self.create_rag_summarization_chain(
                    llm=self.create_llm(),
                    chain_type=chain_type,
                    summarization_combine_template=summarization_combine_template,
                )
