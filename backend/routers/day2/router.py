import re
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
        "You will give your final answer in the last line."
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


def normalize_answer(ans: str) -> str:
    ans = ans.lower()
    ans = re.sub(r"[^a-z0-9\s]", "", ans)
    ans = " ".join(ans.split())  # collapse whitespace
    return ans


def find_final_answer(response: str) -> str:
    return response.split("\n")[-1].strip()


def solve_with_self_consistency(problem_description: str, count: int = 5) -> str:
    prompt = f"""# OUTPUT FORMAT
Give your final answer as one compact sentence in the final line.

# INPUT
{problem_description}
    """
    responses = []
    for _ in range(count):
        responses.append(solve_with_cot(prompt))
    final_answers = [
        normalize_answer(find_final_answer(response)) for response in responses
    ]
    answer_counts = {}
    for answer in final_answers:
        answer_counts[answer] = answer_counts.get(answer, 0) + 1

    answers_sorted = dict(
        sorted(answer_counts.items(), key=lambda item: item[1], reverse=True)
    )
    print(answers_sorted)
    best_answer = list(answers_sorted.keys())[0]
    answer_count = answer_counts[best_answer]
    return f"[{answer_count}]: {best_answer}"


def solve_with_self_consistency_aggregate_with_llm(
    problem_description: str, count: int = 5
) -> str:
    prompt = f"""# OUTPUT FORMAT
Give your final answer as one compact sentence in the final line.

# INPUT
{problem_description}
    """
    responses = []
    for _ in range(count):
        responses.append(solve_with_cot(prompt))
    final_answers = [find_final_answer(response) for response in responses]
    print(final_answers)
    final_answers_str = "\n".join(final_answers)
    client = OpenAI(
        base_url="https://openrouter.ai/api/v1",
    )

    completion = client.chat.completions.create(
        model=model,
        messages=[
            {
                "role": "system",
                "content": "You are a helpful assistant that aggregates multiple answers to the same quesion into just one answer. ",
            },
            {
                "role": "user",
                "content": f"""# OUTPUT FORMAT:
                One final answer in one line, short and precise. Do not format the answer in any way, just return plain text.

                # INPUT
                The request prompt was: {problem_description}

                These were the given answers, find the common answer:
                {final_answers_str}""",
            },
        ],
    )
    response = completion.choices[0].message.content
    if response is None:
        response = "Something went wroing in aggregation."
    return response


@router.post("/core_task_1", response_model=ChatResponse)
def core_task_1(request: ChatRequest) -> ChatResponse:
    reply = solve_with_cot(request.message)
    print(reply)
    return ChatResponse(reply=reply)


@router.post("/core_task_2", response_model=ChatResponse)
def core_task_2(request: ChatRequest) -> ChatResponse:
    reply = solve_with_self_consistency(request.message)
    print(reply)
    return ChatResponse(reply=reply)


@router.post("/core_task_2_bonus", response_model=ChatResponse)
def core_task_2_bonus(request: ChatRequest) -> ChatResponse:
    reply = solve_with_self_consistency_aggregate_with_llm(request.message)
    print(reply)
    return ChatResponse(reply=reply)


@router.get("/health")
def health():
    return {"ok": True}
