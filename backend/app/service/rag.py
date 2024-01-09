import pathlib
from typing import Literal
import io

from pypdf import PdfReader
from pydantic import AnyHttpUrl
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.document_loaders import WebBaseLoader, PyPDFLoader
from langchain_community.document_loaders.blob_loaders import Blob
from langchain.document_loaders.parsers import PyPDFParser
from langchain_core.prompts import PromptTemplate
from langchain.schema import Document
from langchain.llms.base import BaseLLM
from langchain import embeddings
from langchain.chains.summarize import load_summarize_chain
from langchain.chains.base import Chain
from langchain.llms import OpenAI
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
import numpy as np

from app.core.enums import SummarizationChainTypes
from app.core.settings import settings
import app.core.prompts_templates as tmp


class RagService:
    """Class for Retrieval Augmented Generation services."""

    pdf_parser: PyPDFParser = PyPDFParser()
    text_splitter: RecursiveCharacterTextSplitter = RecursiveCharacterTextSplitter(
        separators=["\n\n", "\n"], chunk_size=10000, chunk_overlap=500
    )
    text_large_splitter: RecursiveCharacterTextSplitter = (
        RecursiveCharacterTextSplitter(
            separators=["\n\n", "\n", "\t"], chunk_size=10000, chunk_overlap=3000
        )
    )
    session_docs: list[Document | None] = []

    @staticmethod
    def inject_context(context: str, template: str) -> str:
        """
        Injects the context into the template manually.

        Args:
            context: The context to be injected.
            template: The template to be injected into.

        Returns:
            str: The injected template.
        """
        assert context, "Context must be provided"
        return template.replace("{context}", context)

    @staticmethod
    def num_tokens(text: str | list[Document]) -> int:
        """
        Returns the number of tokens in the text.

        Args:
            text: The text to be counted.

        Returns:
            int: The number of tokens.
        """
        if isinstance(text, list):
            return sum([len(x.page_content.split()) for x in text])
        return OpenAI().get_num_tokens(text)

    @classmethod
    def document_loader(
        cls,
        text: str | bytes | None = None,
        urls: AnyHttpUrl | list[AnyHttpUrl] | None = None,
        file_path: str | None = None,
        pdf_path: str | None = None,
    ) -> list[Document | None]:
        """
        Loads the documents from the given text, urls, file_path or pdf_path.

        Args:
            text: The text to be loaded.
            urls: The urls to be loaded.
            file_path: The file path to be loaded.
            pdf_path: The pdf path to be loaded.

        Returns:
            list[Document | None]: The loaded documents.
        """
        if text:
            try:
                _ = PdfReader(io.BytesIO(text))
                return list(cls.pdf_parser.parse(Blob.from_data(text)))
            except UnicodeError:
                return cls.text_splitter.create_documents(
                    [text if isinstance(text, str) else text.decode("utf-8")]
                )
        elif urls:
            loader = WebBaseLoader(web_path=urls)
            return loader.load()
        elif file_path:
            assert pathlib.Path(file_path).is_file(), "Is not a valid file"
            with open(file_path, "r") as file:
                text = file.read()
            return cls.text_splitter.create_documents([text])
        elif pdf_path:
            assert pathlib.Path(file_path).is_file(), "Is not a valid file"
            loader = PyPDFLoader(pdf_path)
            return loader.load()
        return []

    @staticmethod
    def optimal_clusters(
        vectors: list[list[float]], max_clusters: int = 12
    ) -> tuple[int, float]:
        """
        Returns the optimal number of clusters based on the silhouette score.

        Args:
            vectors: The vectors to be clustered.
            max_clusters: The maximum number of clusters.

        Returns:
            tuple[int, float]: The optimal number of clusters and the silhouette score.
        """
        if len(vectors) < 2:
            raise ValueError(
                "The number of vectors should be at least 2 for clustering."
            )

        best_score = -1
        best_k = 0

        for k in range(2, min(len(vectors), max_clusters) + 1):
            kmeans = KMeans(
                n_clusters=k, init="k-means++", max_iter=300, n_init=10, random_state=0
            )
            preds = kmeans.fit_predict(vectors)
            score = silhouette_score(vectors, preds)
            if score > best_score:
                best_score = score
                best_k = k
        return best_k, best_score

    @staticmethod
    def closest_index(
        kmeans: object, n_clusters: int, vectors: list[list[float]]
    ) -> list[int]:
        """
        Returns the closest index to the cluster center.

        Args:
            kmeans: The kmeans object.
            n_clusters: The number of clusters.
            vectors: The vectors to be clustered.

        Returns:
            list[int]: The closest indices.
        """
        closest_indices = []
        for i in range(n_clusters):
            distances = np.linalg.norm(vectors - kmeans.cluster_centers_[i], axis=1)
            closest_index = np.argmin(distances)
            closest_indices.append(closest_index)
        selected_indices = sorted(closest_indices)
        return selected_indices

    @classmethod
    def vectorizer(
        cls,
        documents: list[Document],
        embed_model: Literal["openai", "faiss", "chroma"] = "openai",
        predict_clusters: bool = True,
    ):
        """
        Vectorizes the documents using the given embed_model.

        Args:
            documents: The documents to be vectorized.
            embed_model: The embed model to be used.
            predict_clusters: Whether to predict the number of clusters.

        Returns:
            list[list[float]]: The vectors.
        """
        vectors = []
        match embed_model:
            case "openai":
                embed = embeddings.OpenAIEmbeddings(
                    openai_api_key=settings.OPENAI_API_KEY
                )
                vectors = embed.embed_documents([x.page_content for x in documents])

        n_clusters = cls.optimal_clusters(vectors)[0] if predict_clusters else 10
        kmeans = KMeans(n_clusters=n_clusters, random_state=42).fit(vectors)
        selected_indices = cls.closest_index(kmeans, n_clusters, vectors)
        return [documents[i] for i in selected_indices]

    @staticmethod
    def create_rag_summarization_chain(
        llm: BaseLLM,
        chain_type: SummarizationChainTypes = SummarizationChainTypes.STUFF,
        context_variable_name: str = "input",
        summarization_map_template: str = tmp.summarization_map_template,
        summarization_combine_template: str = tmp.summarization_combine_template,
        **kwargs: any,
    ) -> Chain:
        """
        Creates a summarization chain.

        Args:
            llm: The instantiated LLM model.
            chain_type: The chain type.
            context_variable_name: The context variable name.

        Returns:
            Chain: The summarization chain.
        """
        match chain_type:
            case SummarizationChainTypes.STUFF:
                return load_summarize_chain(
                    llm=llm, chain_type=str(chain_type), **kwargs
                )
            case SummarizationChainTypes.MAP_REDUCE:
                map_prompt_template = PromptTemplate.from_template(
                    summarization_map_template
                )
                combine_prompt_template = PromptTemplate.from_template(
                    summarization_combine_template
                )
                return load_summarize_chain(
                    llm=llm,
                    chain_type=str(chain_type),
                    map_prompt=map_prompt_template,
                    combine_prompt=combine_prompt_template,
                    combine_document_variable_name=context_variable_name,
                    map_reduce_document_variable_name=context_variable_name,
                )
            case SummarizationChainTypes.REFINE:
                return load_summarize_chain(
                    llm=llm,
                    chain_type=str(chain_type),
                    refine_prompt=PromptTemplate.from_template(
                        tmp.summarization_refine_template
                    ),
                )

    @classmethod
    def summarize_large_text(
        cls,
        docs: list[Document],
        llm: BaseLLM,
        chain_type: SummarizationChainTypes = SummarizationChainTypes.STUFF,
        vectorize: bool = False,
        **kwargs: any,
    ):
        if vectorize:
            docs = cls.vectorizer(docs)
        chain = cls.create_rag_summarization_chain(
            llm=llm, chain_type=chain_type, **kwargs
        )
        return chain.run(docs)
