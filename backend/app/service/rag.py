import pathlib
from typing import Literal

from pydantic import AnyHttpUrl
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.document_loaders import WebBaseLoader, PyPDFLoader
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

from app.core.settings import settings
from app.core.prompts_templates import summarization_map_template, summarization_combine_template, summarization_refine_template


class RagService:

    text_splitter: RecursiveCharacterTextSplitter = RecursiveCharacterTextSplitter(
        separators=["\n\n", "\n", "\t"],
        chunk_size=10000,
        chunk_overlap=3000
    )

    @staticmethod
    def num_tokens(text: str | list[Document]) -> int:
        return OpenAI.get_num_tokens(text)

    @classmethod
    def document_loader(
        cls,
        text: str | bytes | None = None,
        urls: AnyHttpUrl | list[AnyHttpUrl] | None = None,
        file_path: str | None = None,
        pdf_path: str | None = None,
    ) -> list[Document | None]:
        if text:
            return cls.text_splitter.create_documents([
                text if isinstance(text, str) else text.decode()
            ])
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
    def optimal_clusters(vectors: list[list[float]], max_clusters: int = 12) -> tuple[int, float]:
        best_score = -1
        best_k = 0
        for k in range(2, max_clusters + 1):
            kmeans = KMeans(n_clusters=k, init='k-means++', max_iter=300, n_init=10, random_state=0)
            preds = kmeans.fit_predict(vectors)
            score = silhouette_score(vectors, preds)
            if score > best_score:
                best_score = score
                best_k = k
        return best_k, best_score

    @staticmethod
    def closest_index(kmeans: object, n_clusters: int, vectors: list[list[float]]) -> list[int]:
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
        predict_clusters: bool = False
    ):
        vectors = []
        match embed_model:
            case "openai":
                embed = embeddings.OpenAIEmbeddings(openai_api_key=settings.OPENAI_API_KEY)
                vectors = embed.embed_documents([x.page_content for x in documents])
        n_clusters = cls.optimal_clusters(vectors)[0] if predict_clusters else 10
        kmeans = KMeans(n_clusters=n_clusters, random_state=42).fit(vectors)
        selected_indices = cls.closest_index(kmeans, n_clusters, vectors)
        return [documents[i] for i in selected_indices]

    @staticmethod
    def create_summary_chain(
        llm: BaseLLM,
        chain_type: Literal["stuff", "map_reduce", "refine"] = "map_reduce",
        context_variable_name: str = "input",
    ) -> Chain:
        match chain_type:
            case "stuff":
                return load_summarize_chain(llm=llm, chain_type=chain_type)
            case "map_reduce":
                map_prompt_template = PromptTemplate(
                    template=summarization_map_template,
                    input_variables=[context_variable_name]
                )
                combine_prompt_template = PromptTemplate(
                    template=summarization_combine_template,
                    input_variables=[context_variable_name]
                )
                return load_summarize_chain(
                    llm=llm,
                    chain_type=chain_type,
                    map_prompt=map_prompt_template,
                    combine_prompt=combine_prompt_template,
                )
            case "refine":
                return load_summarize_chain(
                    llm=llm,
                    chain_type=chain_type,
                    refine_prompt=PromptTemplate.from_template(summarization_refine_template)
                )

    @classmethod
    def summarize_large_text(
        cls,
        docs: list[Document],
        llm: BaseLLM,
        chain_type: Literal["stuff", "map_reduce", "refine"] = "stuff",
        vectorize: bool = True
    ):
        if vectorize:
            docs = cls.vectorizer(docs)
        chain = cls.create_summary_chain(llm=llm, chain_type=chain_type)
        return chain.summarize(docs)


