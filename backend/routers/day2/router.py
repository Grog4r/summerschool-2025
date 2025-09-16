from fastapi import APIRouter
from ...models import ChatRequest, ChatResponse
from openai import OpenAI

router = APIRouter(prefix="/api/day2", tags=["day2"])
model = "mistralai/mistral-7b-instruct:free"


@router.post("/echo", response_model=ChatResponse)
def echo(request: ChatRequest) -> ChatResponse:
    return ChatResponse(reply=f"Echo (day 2): {request.message}")


def solve_with_cot(problem_description: str) -> str:
    system_prompt = (
        "You are a helpful assistant. "
        "For any question that involves reasoning, show your chain of thought first, step by step, before giving the final answer. "
        "You will answer in the language the user prompt was given in."
    )
    client = OpenAI(
        base_url="https://openrouter.ai/api/v1",
    )

    completion = client.chat.completions.create(
        model=model,
        messages=[
            {
                "role": "system",
                "content": system_prompt,
            },
            {
                "role": "user",
                "content": problem_description,
            },
        ],
        temperature=0.6,
    )
    response = completion.choices[0].message.content
    if response is None:
        response = "Something went wroing in cot."
    return response


def solve_with_self_consistency(problem_description: str, count: int = 5) -> str:
    responses = []
    for _ in range(count):
        responses.append(solve_with_cot(problem_description))
    return ""


@router.post("/core_task_1", response_model=ChatResponse)
def core_task_1(request: ChatRequest) -> ChatResponse:
    reply = solve_with_cot(request.message)
    print(reply)
    return ChatResponse(reply=reply)


@router.get("/health")
def health():
    return {"ok": True}
