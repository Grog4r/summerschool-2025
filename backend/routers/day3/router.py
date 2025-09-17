from haystack import Pipeline
from haystack.components.writers import DocumentWriter
from haystack.components.converters import MarkdownToDocument
from openai import OpenAI
from haystack_integrations.components.retrievers.qdrant import QdrantEmbeddingRetriever
import asyncio
from typing import Any, AsyncIterator

from fastapi import APIRouter
from fastapi.responses import StreamingResponse
from haystack.components.embedders import (
    SentenceTransformersDocumentEmbedder,
    SentenceTransformersTextEmbedder,
)
from haystack_integrations.document_stores.qdrant import QdrantDocumentStore
from haystack.components.preprocessors import DocumentCleaner
from haystack.components.preprocessors import DocumentSplitter

from ...models import ChatMessage, ChatSession

router = APIRouter(prefix="/api/day3", tags=["day3"])

LANGUAGE_MODEL_NAME = "mistralai/mistral-7b-instruct:free"
EMBEDDING_MODEL_NAME = "intfloat/multilingual-e5-small"
embedder = SentenceTransformersDocumentEmbedder(
    model=EMBEDDING_MODEL_NAME, prefix="passage"
)

document_store = QdrantDocumentStore(
    url="localhost:6333",
    # recreate_index=True,
    return_embedding=True,
    embedding_dim=384,
    wait_result_from_api=True,
    similarity="cosine",
)

indexing_pipeline = Pipeline()
indexing_pipeline.add_component("converter", MarkdownToDocument())
indexing_pipeline.add_component("cleaner", DocumentCleaner())
indexing_pipeline.add_component(
    "splitter", DocumentSplitter(split_by="sentence", split_length=5, )
)
indexing_pipeline.add_component("embedder", embedder)
indexing_pipeline.add_component("writer", DocumentWriter(document_store=document_store))
indexing_pipeline.connect("converter", "cleaner")
indexing_pipeline.connect("cleaner", "splitter")
indexing_pipeline.connect("splitter", "embedder")
indexing_pipeline.connect("embedder", "writer")

if document_store.count_documents() == 0:
    print("Document store is empty. Starting indexing...")
    result = indexing_pipeline.run(data={"sources": ["data/Owners_Manual_tesla.md"]})
    print("Indexing finished.")
    print(f"Documents written: {result['writer']['documents_written']}")
else:
    print("Document store already contains documents. Skipping indexing.")


def get_top_k_documents(query: str, top_k: int = 3) -> list[dict[str, Any]]:
    query_pipeline = Pipeline()
    query_pipeline.add_component(
        "text_embedder", SentenceTransformersTextEmbedder(model=EMBEDDING_MODEL_NAME)
    )
    retriever = QdrantEmbeddingRetriever(document_store=document_store, top_k=top_k)
    query_pipeline.add_component("retriever", retriever)
    query_pipeline.connect("text_embedder.embedding", "retriever.query_embedding")

    results = query_pipeline.run({"text_embedder": {"text": query}})
    return results["retriever"]["documents"]


def ask_llm(query: str) -> str:
    print(query)
    client = OpenAI(
        base_url="https://openrouter.ai/api/v1",
    )

    system_prompt = (
        "You are a helpful assistant who will answer user questions in their language with only the provided context. Try to be concise and precise.\n"
        "You shall not output any tokens like <s> or [/s]"
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
    print(response)
    if response is None:
        response = "Something went wrong."
    return response


async def _stream_reply(reply: str) -> AsyncIterator[str]:
    for token in reply.split(" "):
        yield f"{token} "
        await asyncio.sleep(0)


@router.post("/chat")
async def chat(session: ChatSession) -> StreamingResponse:
    documents = get_top_k_documents(session.messages[-1].content)
    documents_str = ""
    for document in documents:
        documents_str += f"{document.content}\n\n"
    query_context = f"""
# CONTEXT
{documents_str}
    """

    history_str = ""
    for message in session.messages:
        history_str += f"{message.role}:\n{message.content}\n\n"
    query_history = f"""
# CHAT HISTORY
{history_str}
    """

    query_prompt = f"# INPUT DATA\n{session.messages[-1].content}"

    full_query = query_history + query_context + query_prompt
    reply = ask_llm(full_query)
    print(f"Reply: '{reply}'")
    session.messages.append(ChatMessage(role="assistant", content=reply))

    stream = _stream_reply(reply)
    return StreamingResponse(stream, media_type="text/markdown")


@router.get("/health")
def health():
    return {"ok": True}
