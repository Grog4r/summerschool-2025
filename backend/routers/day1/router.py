from fastapi import APIRouter
from ...models import ChatRequest, ChatResponse
from openai import OpenAI
import os

router = APIRouter(prefix="/api/day1", tags=["day1"])

lang_to_language = {"de": "german", "en": "english", "fr": "french"}


@router.post("/echo", response_model=ChatResponse)
def echo(request: ChatRequest) -> ChatResponse:
    lang = request.message.rsplit(",")[-1].strip()
    try:
        language = lang_to_language[lang]
    except KeyError:
        response = f"The language abbreviation '{lang}'' is not valid. It has to be in {list(lang_to_language.keys())}"
        return ChatResponse(reply=response)

    title = request.message.rstrip(lang).rstrip().rstrip(",")

    request_content = f"""
    # ROLE
    You are a sophisticated movie expert.
    
    # INSTRUCTIONS
    You write short reviews for movie titles (4-6 sentences). You answer in {language}. 

    # INPUT DATA
    The movie in question is "{title}."
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
