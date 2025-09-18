import json
import random

from openai import OpenAI
from fastapi import APIRouter
from haystack_integrations.components.retrievers.qdrant import QdrantHybridRetriever
from haystack import Document, Pipeline
from haystack.components.writers import DocumentWriter
from haystack.components.embedders import (
    SentenceTransformersDocumentEmbedder,
    SentenceTransformersTextEmbedder,
)
from haystack_integrations.document_stores.qdrant import QdrantDocumentStore
from haystack.components.preprocessors import DocumentCleaner
from haystack.components.preprocessors import DocumentSplitter
from haystack.components.converters import PyPDFToDocument
from haystack.core.component import component
from haystack import Document

# New imports for sparse embeddings
from haystack_integrations.components.embedders.fastembed import (
    FastembedSparseDocumentEmbedder,
    FastembedSparseTextEmbedder,
)

from ...models import ChatSession, ChatMessage


PAGE_OFFSET = 2


@component
class PageNumberCorrector:
    """
    A component to adjust the page number metadata of documents by a fixed offset.
    """

    def __init__(self, offset: int):
        if not isinstance(offset, int):
            raise TypeError("Offset must be an integer.")
        self.offset = offset

    @component.output_types(documents=list[Document])
    def run(self, documents: list[Document]):
        for doc in documents:
            if "page_number" in doc.meta and doc.meta["page_number"] is not None:
                # Only adjust if the resulting page number is positive
                if doc.meta["page_number"] > self.offset:
                    doc.meta["page_number"] -= self.offset
        return {"documents": documents}


router = APIRouter(prefix="/api/day3", tags=["day3"])

# LANGUAGE_MODEL_NAME = "mistralai/mistral-7b-instruct:free"
LANGUAGE_MODEL_NAME = "openai/gpt-oss-20b:free"
# EMBEDDING_MODEL_NAME = "intfloat/multilingual-e5-small"
EMBEDDING_MODEL_NAME = "intfloat/multilingual-e5-large-instruct"
document_embedder = SentenceTransformersDocumentEmbedder(
    model=EMBEDDING_MODEL_NAME, prefix="passage"
)
document_embedder.warm_up()

sparse_document_embedder = FastembedSparseDocumentEmbedder(
    model="prithivida/Splade_PP_en_v1"
)
sparse_document_embedder.warm_up()

document_store = QdrantDocumentStore(
    url="localhost:6333",
    # recreate_index=True,
    embedding_dim=1024,
    use_sparse_embeddings=True,  # Added for sparse embeddings
    return_embedding=True,
    wait_result_from_api=True,
    similarity="cosine",
)

indexing_pipeline = Pipeline()
indexing_pipeline.add_component("converter", PyPDFToDocument())
indexing_pipeline.add_component("corrector", PageNumberCorrector(offset=PAGE_OFFSET))
indexing_pipeline.add_component("cleaner", DocumentCleaner())
splitter = DocumentSplitter(
    split_by="word", language="de", split_length=200, split_overlap=20
)
indexing_pipeline.add_component("splitter", splitter)
indexing_pipeline.add_component("embedder", document_embedder)
indexing_pipeline.add_component(
    "sparse_embedder", sparse_document_embedder
)  # Added sparse embedder
indexing_pipeline.add_component("writer", DocumentWriter(document_store=document_store))

# Adjust the connections to include the new component
indexing_pipeline.connect("converter", "corrector")
indexing_pipeline.connect("corrector", "cleaner")
indexing_pipeline.connect("cleaner", "splitter")
indexing_pipeline.connect("splitter", "embedder")
indexing_pipeline.connect(
    "embedder", "sparse_embedder"
)  # Connect dense to sparse embedder
indexing_pipeline.connect(
    "sparse_embedder", "writer"
)  # Connect sparse embedder to writer

if document_store.count_documents() == 0:
    print("Document store is empty. Starting indexing...")
    result = indexing_pipeline.run(data={"sources": ["data/Owners_Manual_tesla.pdf"]})
    print("Indexing finished.")
    print(f"Documents written: {result['writer']['documents_written']}")
else:
    print("Document store already contains documents. Skipping indexing.")


def get_top_k_documents(
    query: str,
    max_top_k: int = 5,
    similarity_threshold=0.85,
) -> list[tuple[str | None, float | None, int | None]]:
    query_pipeline = Pipeline()
    text_embedder = SentenceTransformersTextEmbedder(model=EMBEDDING_MODEL_NAME)
    text_embedder.warm_up()
    sparse_text_embedder = FastembedSparseTextEmbedder(
        model="prithivida/Splade_PP_en_v1"
    )  # Added sparse text embedder
    sparse_text_embedder.warm_up()

    query_pipeline.add_component("text_embedder", text_embedder)
    query_pipeline.add_component(
        "sparse_text_embedder", sparse_text_embedder
    )  # Add sparse text embedder to pipeline

    retriever = QdrantHybridRetriever(
        document_store=document_store, top_k=max_top_k
    )  # Using Hybrid Retriever
    query_pipeline.add_component("retriever", retriever)

    query_pipeline.connect("text_embedder.embedding", "retriever.query_embedding")
    query_pipeline.connect(
        "sparse_text_embedder.sparse_embedding", "retriever.query_sparse_embedding"
    )  # Connect sparse embedder

    results = query_pipeline.run(
        {"text_embedder": {"text": query}, "sparse_text_embedder": {"text": query}}
    )  # Add sparse embedder input
    docs: list[Document] = results["retriever"]["documents"]

    return [
        (doc.content, doc.score, doc.meta.get("page_number"))
        for doc in docs
        if doc.score is not None and doc.score > similarity_threshold
    ]


def generate_hyde_document(query: str, VERBOSE: bool = True) -> str:
    client = OpenAI(
        base_url="https://openrouter.ai/api/v1",
    )

    system_prompt = (
        "You are a helpful assistant. Generate a document that could potentially answer the following user query. "
        "The document should be detailed and comprehensive. "
        "Do not include any preambles like 'Here is a document that could answer your question'."
    )
    completion = client.chat.completions.create(
        model=LANGUAGE_MODEL_NAME,
        messages=[
            {
                "role": "system",
                "content": system_prompt,
            },
            {
                "role": "user",
                "content": query,
            },
        ],
    )
    response = completion.choices[0].message.content
    if VERBOSE:
        print("---- Hypothetical Document ----")
        print(response)
        print("-----------------------------")

    if response is None:
        response = ""
    return response


def get_hyde_documents(
    query: str,
    max_top_k: int = 3,
    similarity_threshold=0.80,
    VERBOSE: bool = False,
) -> list[tuple[str | None, float | None, int | None]]:
    """
    Retrieves documents using the HyDE technique.
    """
    hypothetical_document = generate_hyde_document(query, VERBOSE=VERBOSE)

    # If the hypothetical document is empty, fall back to the original query
    if not hypothetical_document.strip():
        search_text = query
    else:
        search_text = hypothetical_document

    query_pipeline = Pipeline()
    text_embedder = SentenceTransformersTextEmbedder(model=EMBEDDING_MODEL_NAME)
    text_embedder.warm_up()
    sparse_text_embedder = FastembedSparseTextEmbedder(
        model="prithivida/Splade_PP_en_v1"
    )
    sparse_text_embedder.warm_up()

    query_pipeline.add_component("text_embedder", text_embedder)
    query_pipeline.add_component("sparse_text_embedder", sparse_text_embedder)

    retriever = QdrantHybridRetriever(document_store=document_store, top_k=max_top_k)
    query_pipeline.add_component("retriever", retriever)

    query_pipeline.connect("text_embedder.embedding", "retriever.query_embedding")
    query_pipeline.connect(
        "sparse_text_embedder.sparse_embedding", "retriever.query_sparse_embedding"
    )

    results = query_pipeline.run(
        {
            "text_embedder": {"text": search_text},
            "sparse_text_embedder": {"text": search_text},
        }
    )
    docs: list[Document] = results["retriever"]["documents"]

    return [
        (doc.content, doc.score, doc.meta.get("page_number"))
        for doc in docs
        if doc.score is not None and doc.score > similarity_threshold
    ]


def ask_llm(
    query: str,
    max_tokens: int | None = None,
    temperature: float | None = None,
    VERBOSE: bool = False,
) -> str:
    if VERBOSE:
        print(query)

    client = OpenAI(
        base_url="https://openrouter.ai/api/v1",
    )

    system_prompt = (
        "You are a helpful assistant who answers user questions concisely using only the provided context. "
        "Answer in the users language. Think internally about your answer befor you answer. "
        "Answer in 1 to 3 sentences, be precise and short. "
        "Always cite the page numbers where the information was found *at the end of your answer*, using the format (Seite X, Y, Z). "
        "If you cannot find the answer in the context, say so politely. Do not invent page numbers. "
        "You shall not output any unreadable tokens like <s> or [/s]. Only use standard ASCII tokens."
    )
    completion = client.chat.completions.create(
        model=LANGUAGE_MODEL_NAME,
        messages=[
            {
                "role": "system",
                "content": system_prompt,
            },
            {
                "role": "user",
                "content": query,
            },
        ],
        max_tokens=max_tokens,
        temperature=temperature,
    )
    response = completion.choices[0].message.content
    if VERBOSE:
        print(response)

    if response is None:
        response = "Something went wrong."
    return response


def get_response(
    session,
    max_top_k: int = 3,
    similarity_threshold: float = 0.0,
    USE_HYDE: bool = False,
    VERBOSE: bool = False,
) -> str:
    if USE_HYDE:
        documents = get_hyde_documents(
            session.messages[-1].content,
            max_top_k=max_top_k,
            similarity_threshold=similarity_threshold,
        )
    else:
        documents = get_top_k_documents(
            session.messages[-1].content,
            max_top_k=max_top_k,
            similarity_threshold=similarity_threshold,
        )
    if VERBOSE:
        print(documents)
    documents_str = ""
    for document_content, document_score, document_page in documents:
        documents_str += f"[Seite {document_page}]: {document_content}\n\n"
    query_context = f"# CONTEXT\n{documents_str}"

    history_str = ""
    for message in session.messages:
        history_str += f"{message.role}:\n{message.content}\n\n"
    query_history = f"# CHAT HISTORY\n{history_str}"

    query_prompt = f"# INPUT DATA\n{session.messages[-1].content}"

    full_query = query_history + query_context + query_prompt
    reply = ask_llm(full_query, VERBOSE=VERBOSE)
    if len(reply.strip()) == 0:
        reply = "Entschuldigung, hier ist etwas schief gegangen."
    asciify_table = {
        "‑": "-",
        "–": "-",
        " ": " ",
    }
    for old, new in asciify_table.items():
        reply = reply.replace(old, new)
    return reply


if __name__ == "__main__":
    # Run with:
    # python -m backend.routers.day3.tools
    import numpy as np

    VERBOSE = True

    def cosine_similarity(a: list[float], b: list[float]) -> float:
        a_array = np.array(a)
        b_array = np.array(b)
        return float(
            np.dot(a_array, b_array)
            / (np.linalg.norm(a_array) * np.linalg.norm(b_array))
        )

    scores = []
    responses = []
    test_data = json.load(open("data/test_data.json", "r"))
    # random.seed(0)
    # random.shuffle(test_data)
    counter = 0

    text_embedder = SentenceTransformersTextEmbedder(model=EMBEDDING_MODEL_NAME)
    text_embedder.warm_up()

    for data in test_data:
        # if counter > 3:
        #     break

        response = get_response(
            ChatSession(messages=[ChatMessage(role="user", content=data["query"])]),
            VERBOSE=VERBOSE,
        )

        resp_emb = text_embedder.run(response)["embedding"]
        gold_emb = text_embedder.run(data["answer"])["embedding"]

        similarity = cosine_similarity(resp_emb, gold_emb)

        responses.append(
            {
                "query": data["query"],
                "answer": data["answer"],
                "page": data["page"],
                "result": response,
            }
        )
        scores.append(similarity)
        counter += 1

    print(
        f"Finished {counter} queries. Average similarity score is {np.mean(scores):.3f}"
    )

    with open("backend/routers/day4/evaluation_data_day_4.json", "w") as f:
        json.dump(responses, f, indent=2, ensure_ascii=False)
