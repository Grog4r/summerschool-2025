from decimal import DecimalException
from fastapi import APIRouter
from ...models import ChatRequest, ChatResponse
from openai import OpenAI
import os
import json

router = APIRouter(prefix="/api/day1", tags=["day1"])


@router.post("/echo", response_model=ChatResponse)
def echo(request: ChatRequest) -> ChatResponse:

    review_prompt = f"""
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
        messages=[{"role": "user", "content": review_prompt}],
    )
    review = completion.choices[0].message.content
    if review is None:
        return ChatResponse(reply="The review was None.")

    sentiment_prompt = f"""
    # ROLE
    You are a sentiment analysis machine.

    # INSTRUCTIONS
    It is your job to take in a movie review and to rate the movie on a few different aspects according to the review:
    - acting
    - story
    - visual and visual effects
    You have to give each category a rating of "positive", "negative" or "mixed".
    You have to give the answer back as a JSON in the following format:
    {{
        "acting": <RATING>,
        "story": <RATING>,
        "visual": <RATING>
    }}

    # INPUT DATA
    {review}
    """

    completion = client.chat.completions.create(
        extra_body={},
        model="openai/gpt-oss-20b:free",
        messages=[{"role": "user", "content": sentiment_prompt}],
        response_format={"type": "json_object"},
    )
    sentiment_json = completion.choices[0].message.content
    if sentiment_json is None:
        return ChatResponse(reply="The sentiment_json was None.")
    
    try:
        json.loads(sentiment_json)
    except json.JSONDecodeError:
        print("Invalid JSON. Letting the LLM retry.")
        repair_prompt = f"""
        # ROLE
        You are a sentiment analysis machine.

        # INSTRUCTIONS
        It is your job to take in a movie review and to rate the movie on a few different aspects according to the review:
        - acting
        - story
        - visual and visual effects
        You have to give each category a rating of "positive", "negative" or "mixed".
        You have to give the answer back as a JSON in the following format:
        {{
            "acting": <RATING>,
            "story": <RATING>,
            "visual": <RATING>
        }}

        # CONTEXT
        Your last answer was not valid JSON. You answered with "{sentiment_json}". Please redo your task properly.
        Do not apologize.

        # INPUT DATA
        {review}
        """
        completion = client.chat.completions.create(
            extra_body={},
            model="openai/gpt-oss-20b:free",
            messages=[{"role": "user", "content": repair_prompt}],
            response_format={"type": "json_object"},
        )
        sentiment_json = completion.choices[0].message.content
        if sentiment_json is None:
            return ChatResponse(reply="The sentiment_json was None after repairing.")
        try:
            json.loads(sentiment_json)
        except json.JSONDecodeError:
            return ChatResponse(reply="The JSON could not be repaired.")
        sentiment_json += "\n(This was done after repairing.)"

    response_string = review + "\n\n" + sentiment_json

    return ChatResponse(reply=response_string)


@router.get("/health")
def health():
    return {"ok": True}
