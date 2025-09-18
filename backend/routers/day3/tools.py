import json
import random
from haystack import Document, Pipeline
from haystack.components.writers import DocumentWriter
from openai import OpenAI
from haystack_integrations.components.retrievers.qdrant import QdrantEmbeddingRetriever
from fastapi import APIRouter
from typing import Any
from haystack.components.embedders import (
    SentenceTransformersDocumentEmbedder,
    SentenceTransformersTextEmbedder,
)
from haystack_integrations.document_stores.qdrant import QdrantDocumentStore
from haystack.components.preprocessors import DocumentCleaner
from haystack.components.preprocessors import DocumentSplitter
from haystack.components.converters import PyPDFToDocument
from unidecode import unidecode

from ...models import ChatSession, ChatMessage


router = APIRouter(prefix="/api/day3", tags=["day3"])

# LANGUAGE_MODEL_NAME = "mistralai/mistral-7b-instruct:free"
LANGUAGE_MODEL_NAME = "openai/gpt-oss-20b:free"
EMBEDDING_MODEL_NAME = "intfloat/multilingual-e5-small"
document_embedder = SentenceTransformersDocumentEmbedder(
    model=EMBEDDING_MODEL_NAME, prefix="passage"
)
document_embedder.warm_up()

document_store = QdrantDocumentStore(
    url="localhost:6333",
    # recreate_index=True,
    return_embedding=True,
    embedding_dim=384,
    wait_result_from_api=True,
    similarity="cosine",
)

indexing_pipeline = Pipeline()
indexing_pipeline.add_component("converter", PyPDFToDocument())
indexing_pipeline.add_component("cleaner", DocumentCleaner())
indexing_pipeline.add_component("splitter", DocumentSplitter(split_by="word"))
indexing_pipeline.add_component("embedder", document_embedder)
indexing_pipeline.add_component("writer", DocumentWriter(document_store=document_store))
indexing_pipeline.connect("converter", "cleaner")
indexing_pipeline.connect("cleaner", "splitter")
indexing_pipeline.connect("splitter", "embedder")
indexing_pipeline.connect("embedder", "writer")

if document_store.count_documents() == 0:
    print("Document store is empty. Starting indexing...")
    result = indexing_pipeline.run(data={"sources": ["data/Owners_Manual_tesla.pdf"]})
    print("Indexing finished.")
    print(f"Documents written: {result['writer']['documents_written']}")
else:
    print("Document store already contains documents. Skipping indexing.")


def get_top_k_documents(
    query: str,
    max_top_k: int = 3,
    similarity_threshold=0.85,
) -> list[tuple[str | None, float | None]]:
    query_pipeline = Pipeline()
    text_embedder = SentenceTransformersTextEmbedder(model=EMBEDDING_MODEL_NAME)
    text_embedder.warm_up()
    query_pipeline.add_component("text_embedder", text_embedder)
    retriever = QdrantEmbeddingRetriever(document_store=document_store, top_k=max_top_k)
    query_pipeline.add_component("retriever", retriever)
    query_pipeline.connect("text_embedder.embedding", "retriever.query_embedding")

    results = query_pipeline.run({"text_embedder": {"text": query}})
    docs: list[Document] = results["retriever"]["documents"]

    return [
        (doc.content, doc.score)
        for doc in docs
        if doc.score is not None and doc.score > similarity_threshold
    ]


def ask_llm(query: str, VERBOSE: bool = False) -> str:
    if VERBOSE:
        print(query)

    client = OpenAI(
        base_url="https://openrouter.ai/api/v1",
    )

    system_prompt = (
        "You are a helpful assistant who will answer user questions in their language with only the provided context.\n"
        "First, reason internally (but don't output). Then, provide the user with only a concise 1-3 sentence answer.\n"
        "If you do not know an answer, tell the user kindly. Never hallucinate an answer.\n"
        "You shall not output any unreadable tokens like <s> or [/s]"
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
        print(response)

    if response is None:
        response = "Something went wrong."
    return response


def get_response(session, VERBOSE: bool = False) -> str:
    documents = get_top_k_documents(session.messages[-1].content)
    if VERBOSE:
        print(documents)
    documents_str = ""
    for document_content, document_score in documents:
        documents_str += f"{document_content}\n\n"
    query_context = f"# CONTEXT\n{documents_str}"

    history_str = ""
    for message in session.messages:
        history_str += f"{message.role}:\n{message.content}\n\n"
    query_history = f"# CHAT HISTORY\n{history_str}"

    query_prompt = f"# INPUT DATA\n{session.messages[-1].content}"

    full_query = query_history + query_context + query_prompt
    reply = ask_llm(full_query)
    if len(reply.strip()) == 0:
        reply = "Entschuldigung, hier ist etwas schief gegangen."
    return unidecode(reply)


if __name__ == "__main__":
    # Run with:
    # python -m backend.routers.day3.tools
    import numpy as np

    def cosine_similarity(a: list[float], b: list[float]) -> float:
        a_array = np.array(a)
        b_array = np.array(b)
        return float(
            np.dot(a_array, b_array)
            / (np.linalg.norm(a_array) * np.linalg.norm(b_array))
        )

    responses = []
    test_data = json.load(open("data/test_data.json", "r"))
    random.seed(0)
    random.shuffle(test_data)
    counter = 0

    text_embedder = SentenceTransformersTextEmbedder(model=EMBEDDING_MODEL_NAME)
    text_embedder.warm_up()

    for data in test_data:
        if counter > 5:
            break

        response = get_response(
            ChatSession(messages=[ChatMessage(role="user", content=data["query"])])
        )

        # embed both response and ground truth
        resp_emb = text_embedder.run(response)["embedding"]
        gold_emb = text_embedder.run(data["answer"])["embedding"]

        similarity = cosine_similarity(resp_emb, gold_emb)

        responses.append(
            {
                "query": data["query"],
                "expected": data["answer"],
                "response": response,
                "similarity": similarity,
            }
        )
        counter += 1

    # you could also dump this to a JSON file for later inspection
    with open("results.json", "w") as f:
        json.dump(responses, f, indent=2, ensure_ascii=False)
