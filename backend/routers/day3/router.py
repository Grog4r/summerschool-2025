import asyncio
from typing import  AsyncIterator

from fastapi import APIRouter
from fastapi.responses import StreamingResponse

from .tools import get_response
from ...models import ChatMessage, ChatSession

router = APIRouter(prefix="/api/day3", tags=["day3"])


async def _stream_reply(reply: str) -> AsyncIterator[str]:
    for token in reply.split(" "):
        yield f"{token} "
        await asyncio.sleep(0)


@router.post("/chat")
async def chat(session: ChatSession) -> StreamingResponse:
    reply = get_response(session)
    print(f"Reply: '{reply}'")
    session.messages.append(ChatMessage(role="assistant", content=reply))

    stream = _stream_reply(reply)
    return StreamingResponse(stream, media_type="text/markdown")


@router.get("/health")
def health():
    return {"ok": True}
