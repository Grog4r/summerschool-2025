from haystack import Pipeline
from haystack.components.preprocessors import DocumentPreprocessor
from haystack.components.writers import DocumentWriter
from haystack.components.converters import MarkdownToDocument
from haystack_integrations.components.retrievers.qdrant import QdrantEmbeddingRetriever
import asyncio
from typing import AsyncIterator

from fastapi import APIRouter
from fastapi.responses import StreamingResponse
from haystack.components.embedders import (
    SentenceTransformersDocumentEmbedder,
    SentenceTransformersTextEmbedder,
)
from haystack_integrations.document_stores.qdrant import QdrantDocumentStore
from haystack.components.preprocessors import DocumentCleaner
from haystack.components.preprocessors import DocumentSplitter
from haystack.components.retrievers.in_memory import InMemoryEmbeddingRetriever

from ...models import ChatMessage, ChatSession

router = APIRouter(prefix="/api/day3", tags=["day3"])


MODEL_NAME = "intfloat/multilingual-e5-small"
embedder = SentenceTransformersDocumentEmbedder(model=MODEL_NAME, prefix="passage")

document_store = QdrantDocumentStore(
    url="localhost:6333",
    # recreate_index=True,
    return_embedding=True,
    embedding_dim=384,
    wait_result_from_api=True,
    similarity="cosine",
)

retriever = QdrantEmbeddingRetriever(document_store=document_store)


preprocessor = DocumentPreprocessor(split_by="passage")

indexing_pipeline = Pipeline()
indexing_pipeline.add_component("converter", MarkdownToDocument())
indexing_pipeline.add_component("cleaner", DocumentCleaner())
indexing_pipeline.add_component(
    "splitter", DocumentSplitter(split_by="sentence", split_length=5)
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

query_pipeline = Pipeline()
query_pipeline.add_component(
    "text_embedder", SentenceTransformersTextEmbedder(model=MODEL_NAME)
)
query_pipeline.add_component("retriever", retriever)
query_pipeline.connect("text_embedder.embedding", "retriever.query_embedding")


def get_query_embedding(query: str):
    result = query_pipeline.run({"text_embedder": {"text": query}})
    return result


def _build_reply_text(messages: list[ChatMessage]) -> str:
    for msg in reversed(messages):
        if msg.role == "user":
            return f"Chat (day 3) streaming: {msg.content}"
    return "Chat (day 3) ready when you are."


async def _stream_reply(messages: list[ChatMessage]) -> AsyncIterator[str]:
    reply = _build_reply_text(messages)
    for token in reply.split():
        yield f"{token} "
        await asyncio.sleep(0)


@router.post("/chat")
async def chat(session: ChatSession) -> StreamingResponse:

    query_embedding = get_query_embedding(session.messages[-1].content)
    print(query_embedding)
    stream = _stream_reply(session.messages)
    return StreamingResponse(stream, media_type="text/plain")


@router.get("/health")
def health():
    return {"ok": True}
