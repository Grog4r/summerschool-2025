from haystack import Pipeline
from haystack.components.preprocessors import DocumentPreprocessor
from haystack.components.writers import DocumentWriter
from haystack.components.converters import MarkdownToDocument
import asyncio
from typing import AsyncIterator

from fastapi import APIRouter
from fastapi.responses import StreamingResponse
from haystack.components.embedders import SentenceTransformersDocumentEmbedder
from haystack_integrations.document_stores.qdrant import QdrantDocumentStore
from haystack.components.preprocessors import DocumentCleaner
from haystack.components.preprocessors import DocumentSplitter

from ...models import ChatMessage, ChatSession

router = APIRouter(prefix="/api/day3", tags=["day3"])

embedder = SentenceTransformersDocumentEmbedder(
    model="intfloat/multilingual-e5-small", prefix="passage"
)

document_store = QdrantDocumentStore(
    url="localhost:6333",
    recreate_index=True,
    return_embedding=True,
    wait_result_from_api=True,
)

preprocessor = DocumentPreprocessor(split_by="passage")

pipeline = Pipeline()
pipeline.add_component("converter", MarkdownToDocument())
pipeline.add_component("cleaner", DocumentCleaner())
pipeline.add_component("splitter", DocumentSplitter(split_by="sentence", split_length=5))
pipeline.add_component("writer", DocumentWriter(document_store=document_store))
pipeline.connect("converter", "cleaner")
pipeline.connect("cleaner", "splitter")
pipeline.connect("splitter", "writer")

result = pipeline.run(data={"sources": ["data/Owners_Manual_tesla.md"]})

print("Indexing finished.")
print(f"Documents written: {result['writer']['documents_written']}")

# # --- Add this code to list all documents ---

# all_documents = document_store.filter_documents()
# print(f"\nTotal documents retrieved from store: {len(all_documents)}")
# # You can also inspect the content of the first document
# if all_documents:
#     print("\n--- Content of the first document: ---")
#     print(all_documents[0].content)


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
    stream = _stream_reply(session.messages)
    return StreamingResponse(stream, media_type="text/plain")


@router.get("/health")
def health():
    return {"ok": True}
