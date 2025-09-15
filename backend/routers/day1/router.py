from fastapi import APIRouter
from ...models import ChatRequest, ChatResponse
from openai import OpenAI
import os

router = APIRouter(prefix="/api/day1", tags=["day1"])

lang_to_language = {"de": "german", "en": "english", "fr": "french"}


@router.post("/echo", response_model=ChatResponse)
def echo(request: ChatRequest) -> ChatResponse:

    request_content = f"""
    # ROLE
    You are a sophisticated movie expert.
    
    # INSTRUCTIONS
    You write short reviews for movie titles (4-6 sentences).
    The input data is in the format "<MOVIE TITLE>, <LANGUAGE_ABBR|LANGUAGE>".
    If you do not know the language or it's abbreviation, please tell the user.
    If you do not know the movie in question, please tell the user kindly in the specified language and tell them your knowledge cutoff date, which is "June 2024", in case this is the issue. Do not hallucinate movies.

    # INPUT DATA
    {request.message}
    """

    client = OpenAI(
        base_url="https://openrouter.ai/api/v1",
        api_key=os.getenv(""),
    )

    completion = client.chat.completions.create(
        extra_body={},
        model="openai/gpt-oss-20b:free",
        messages=[{"role": "user", "content": request_content}],
    )

    response = completion.choices[0].message.content
    if response is None:
        response = "Something went wrong."

    return ChatResponse(reply=response)


@router.get("/health")
def health():
    return {"ok": True}
