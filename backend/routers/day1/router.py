from fastapi import APIRouter
from ...models import ChatRequest, ChatResponse
from openai import OpenAI
import json
import json

router = APIRouter(prefix="/api/day1", tags=["day1"])
model = "nvidia/nemotron-nano-9b-v2:free"



@router.post("/core_task_1", response_model=ChatResponse)
def core_task_1(request: ChatRequest) -> ChatResponse:

    review_prompt = f"""
# ROLE
You are a sophisticated movie expert.

# INSTRUCTIONS
You write short reviews for movie titles (4-6 sentences).
The input data is in the format "<MOVIE TITLE>, <LANGUAGE_ABBR|LANGUAGE>".
If you do not know the language or it's abbreviation, please tell the user.
If you do not know the movie in question, please tell the user kindly in the specified language and tell them your knowledge cutoff date, in case this is the issue. Do not hallucinate movies.

# INPUT DATA
{request.message}
    """

    client = OpenAI(
        base_url="https://openrouter.ai/api/v1",
    )

    completion = client.chat.completions.create(
        extra_body={},
        model=model,
        messages=[{"role": "user", "content": review_prompt}],
    )
    review = completion.choices[0].message.content
    if review is None:
        return ChatResponse(reply="The review was None.")

    return ChatResponse(reply=review)


@router.post("/core_task_2", response_model=ChatResponse)
def core_task_2(request: ChatRequest) -> ChatResponse:

    review_prompt = f"""
# ROLE
You are a sophisticated movie expert.

# INSTRUCTIONS
You write short reviews for movie titles (4-6 sentences).
The input data is in the format "<MOVIE TITLE>, <LANGUAGE_ABBR|LANGUAGE>".
If you do not know the language or it's abbreviation, please tell the user.
If you do not know the movie in question, please tell the user kindly in the specified language and tell them your knowledge cutoff date, in case this is the issue. Do not hallucinate movies.

# INPUT DATA
{request.message}
    """

    client = OpenAI(
        base_url="https://openrouter.ai/api/v1",
    )

    completion = client.chat.completions.create(
        extra_body={},
        model=model,
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

# OUTPUT FORMAT
You must return only a valid JSON object. Do not include any other text, code fences, or commentary.
The JSON must look exactly like this (with your filled-in values):
{{
"acting": "positive|negative|mixed",
"story": "positive|negative|mixed",
"visual": "positive|negative|mixed"
}}

# INPUT DATA
{review}
    """

    completion = client.chat.completions.create(
        extra_body={},
        model=model,
        messages=[{"role": "user", "content": sentiment_prompt}],
        response_format={"type": "json_object"},
    )
    sentiment_json = completion.choices[0].message.content
    if sentiment_json is None:
        return ChatResponse(reply="The sentiment_json was None.")

    response_string = review + "\n\n" + sentiment_json

    return ChatResponse(reply=response_string)


@router.post("/core_task_3", response_model=ChatResponse)
def core_task_3(request: ChatRequest) -> ChatResponse:

    review_prompt = f"""
# ROLE
You are a sophisticated movie expert.

# INSTRUCTIONS
You write short reviews for movie titles (4-6 sentences).
The input data is in the format "<MOVIE TITLE>, <LANGUAGE_ABBR|LANGUAGE>".
If you do not know the language or it's abbreviation, please tell the user.
If you do not know the movie in question, please tell the user kindly in the specified language and tell them your knowledge cutoff date, in case this is the issue. Do not hallucinate movies.

# INPUT DATA
    {request.message}
    """

    client = OpenAI(
        base_url="https://openrouter.ai/api/v1",
    )

    completion = client.chat.completions.create(
        extra_body={},
        model=model,
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

# OUTPUT FORMAT
You must return only a valid JSON object. Do not include any other text, code fences, or commentary. NO COMMENTARY!!!
The JSON must look exactly like this (with your filled-in values):
{{
"acting": "positive|negative|mixed",
"story": "positive|negative|mixed",
"visual": "positive|negative|mixed"
}}


# INPUT DATA
{review}
    """

    completion = client.chat.completions.create(
        extra_body={},
        model=model,
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

# OUTPUT FORMAT
You must return only a valid JSON object. Do not include any other text, code fences, or commentary.
The JSON must look exactly like this (with your filled-in values):
{{
"acting": "positive|negative|mixed",
"story": "positive|negative|mixed",
"visual": "positive|negative|mixed"
}}

# CONTEXT
Your last answer was not valid JSON. You answered with "{sentiment_json}". Please redo your task properly.
Do not apologize.

# INPUT DATA
    {review}
        """
        completion = client.chat.completions.create(
            extra_body={},
            model=model,
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

    response_string = f""""
{review}

```json
{sentiment_json}
```
    """
    print(response_string)

    return ChatResponse(reply=response_string)


@router.get("/health")
def health():
    return {"ok": True}
