#!/usr/bin/env python3
"""
Day 4 Evaluation Script

This script evaluates RAG system outputs using an LLM judge through OpenRouter.
It compares system-generated answers against ground truth using GPT-4.
"""

import json
import os
import sys
import time
from dotenv import load_dotenv

from openai import OpenAI
from pydantic import ValidationError

load_dotenv(override=True)

try:
    from .models import JudgeResponse, EvaluationResult, EvaluationData
except ImportError:
    from models import JudgeResponse, EvaluationResult, EvaluationData


def load_evaluation_data(file_path: str) -> EvaluationData:
    """Load and validate evaluation data from JSON file"""
    try:
        with open(file_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        # Convert list to EvaluationData format if needed
        if isinstance(data, list):
            data = {"items": data}

        return EvaluationData(**data)

    except FileNotFoundError:
        raise FileNotFoundError(f"Evaluation data file not found: {file_path}")
    except json.JSONDecodeError as e:
        raise ValueError(f"Invalid JSON in evaluation data file: {e}")
    except ValidationError as e:
        raise ValueError(f"Invalid evaluation data format: {e}")


def create_system_prompt() -> str:
    """Create the system prompt for the LLM judge"""
    return """System
You are a strict, domain-aware grader for Tesla vehicle documentation. Evaluate a SYSTEM_ANSWER only against the REFERENCE_ANSWER from the official manual. Output valid JSON only.

Task
Decide if the SYSTEM_ANSWER is factually correct and complete relative to the REFERENCE_ANSWER.

Scoring
- 1 (CORRECT): All essential facts in REFERENCE_ANSWER are present and accurate in SYSTEM_ANSWER. Minor wording differences allowed.
- 0 (INCORRECT): Any essential fact missing, contradicted, or replaced by irrelevant content. If uncertain, choose 0.

Rules
1) Compare content, not style. Prefer factual accuracy over phrasing.
2) Numbers, procedures, and specifications must match. Unit conversions must be equivalent.
3) Do not reward correct-but-different procedures if they omit essentials in REFERENCE_ANSWER.
4) Ignore extra harmless details unless they contradict or obscure essentials.
5) If REFERENCE_ANSWER lists steps, all critical steps must appear. Missing critical steps → 0.
6) Treat ambiguous or generic answers as 0 if essentials are not clearly met.
7) Do not add new facts. Do not infer beyond REFERENCE_ANSWER.
8) If REFERENCE_ANSWER is empty or unclear, return 0 unless SYSTEM_ANSWER exactly states that the reference is unavailable.
9) Do NOT Think, output the result immediately
Input Format (JSON provided to you)
{
  "question": "<string>",
  "reference_answer": "<string>",
  "system_answer": "<string>",
  "metadata": { "page": "<string or number>", "notes": "<optional string>" }
}

Output Format (must be a single JSON object; no markdown, no prose, no trailing commas)
{
  "correct": 0 or 1,
  "reasoning": "<brief justification in <= 40 words. Cite which essential element matched or was missing. No new facts.>"
}

Validation Requirements
- Keys in this order: correct, reasoning.
- Double quotes only. No comments. No extra keys. One line preferred.
- If you cannot produce valid JSON, output {"correct":0,"reasoning":"invalid-json-would-have-been-produced"}

Decision Checklist (apply silently)
- Identify essential elements in REFERENCE_ANSWER.
- Verify each element is present and accurate in SYSTEM_ANSWER.
- Check numbers/units/steps.
- Penalize contradictions or omissions.
- Resolve ties toward 0.

Few-Shot Examples

Example 1 — CORRECT
Input:
{
  "question": "How do I power off the vehicle from the touchscreen?",
  "reference_answer": "Open Controls > Safety > Power Off. Wait 2 minutes without interacting, then press the brake to wake.",
  "system_answer": "Tap Controls, open Safety, choose Power Off. Do not touch anything for about 2 minutes, then press the brake to turn it back on.",
  "metadata": { "page": 142 }
}
Output:
{"correct":1,"reasoning":"All essential steps present: Controls > Safety > Power Off, wait 2 minutes no interaction, press brake to wake."}

Example 2 — INCORRECT (missing critical step)
Input:
{
  "question": "How to reboot the touchscreen?",
  "reference_answer": "Press and hold both steering wheel scroll buttons until the screen turns black.",
  "system_answer": "Hold the right scroll button until the screen turns off.",
  "metadata": { "page": 103 }
}
Output:
{"correct":0,"reasoning":"Missing essential detail: both scroll buttons are required, not only right."}

Example 3 — INCORRECT (numeric mismatch)
Input:
{
  "question": "What is the recommended cold tire pressure?",
  "reference_answer": "42 psi (cold).",
  "system_answer": "40 psi is recommended.",
  "metadata": { "page": 220 }
}
Output:
{"correct":0,"reasoning":"Numeric value does not match the reference 42 psi."}

Example 4 — CORRECT (wording differs, semantics match)
Input:
{
  "question": "How to enable child lock?",
  "reference_answer": "Go to Controls > Locks and turn Child Lock ON.",
  "system_answer": "Open Controls, then Locks, toggle Child Lock to On.",
  "metadata": { "page": 155 }
}
Output:
{"correct":1,"reasoning":"Essential path and action match: Controls > Locks, toggle Child Lock On."}

Example 5 — INCORRECT (irrelevant procedure)
Input:
{
  "question": "How to open the charge port from the touchscreen?",
  "reference_answer": "On the touchscreen, tap Controls > Charging > Open Charge Port.",
  "system_answer": "Press the button on the charging connector to open the port.",
  "metadata": { "page": 198 }
}
Output:
{"correct":0,"reasoning":"Provides an alternative method, but missing the required touchscreen steps from the reference."}
"""

def create_user_prompt(query: str, ground_truth: str, system_answer: str) -> str:
    """Create the user prompt with the specific evaluation data"""
    return f"""Please evaluate the following:

**Question:** {query}

**Reference Answer (Ground Truth):** {ground_truth}

**System Answer (To Evaluate):** {system_answer}

Provide your evaluation in JSON format."""


def call_llm_judge(
    client: OpenAI,
    query: str,
    ground_truth: str,
    system_answer: str,
    max_retries: int = 3,
    use_structured_output: bool = True,
    debug: bool = False,
) -> JudgeResponse:
    """Call the LLM judge to evaluate a single Q&A pair"""
    system_prompt = create_system_prompt()
    user_prompt = create_user_prompt(query, ground_truth, system_answer)

    for attempt in range(max_retries):
        try:
            # Choose response format based on structured output support
            if use_structured_output:
                response_format = JudgeResponse.get_openrouter_response_format()
            else:
                response_format = {"type": "json_object"}

            if debug:
                print(f"Attempt {attempt + 1}: Using {'structured' if use_structured_output else 'basic'} output mode")
                print(f"Response format: {response_format}")

            response = client.chat.completions.create(
                model="openrouter/sonoma-dusk-alpha",
                messages=[
                    {
                        "role": "system",
                        "content": system_prompt,
                    },
                    {"role": "user", "content": user_prompt},
                ],
                response_format=response_format,
                temperature=0.1,
            )

            # Safely extract response content
            if not response.choices or not response.choices[0].message:
                print(f"Warning: Empty response on attempt {attempt + 1}")
                print(f"Full response object: {response}")
                if attempt == max_retries - 1:
                    return JudgeResponse(correct=0, reasoning="Empty response from API")
                continue

            result_text = response.choices[0].message.content
            if not result_text:
                print(f"Warning: Empty content on attempt {attempt + 1}")
                print(f"Message object: {response.choices[0].message}")
                if attempt == max_retries - 1:
                    return JudgeResponse(correct=0, reasoning="Empty content from API")
                continue

            result_text = result_text.strip()
            if not result_text:
                print(f"Warning: Empty content after strip on attempt {attempt + 1}")
                print(f"Original content before strip: {repr(response.choices[0].message.content)}")
                if attempt == max_retries - 1:
                    return JudgeResponse(correct=0, reasoning="Empty content after processing")
                continue

            # Parse JSON based on mode
            try:
                if debug:
                    print(f"Parsing response: {repr(result_text[:100])}")

                result_json = json.loads(result_text)

                if debug:
                    print(f"Parsed JSON: {result_json}")

                # Create and validate the response
                judge_response = JudgeResponse(**result_json)

                # Always show successful responses for transparency
                print(f"✅ Judge decision: {judge_response.correct} ({'CORRECT' if judge_response.correct == 1 else 'INCORRECT'})")
                print(f"📝 Reasoning: {judge_response.reasoning}")

                return judge_response

            except json.JSONDecodeError as e:
                print(f"\n{'='*60}")
                print(f"🚨 JSON PARSING FAILED - ATTEMPT {attempt + 1}")
                print(f"{'='*60}")
                print(f"Error: {e}")
                print(f"Using structured output: {use_structured_output}")
                print(f"Response length: {len(result_text)} characters")
                print(f"\n📋 EVALUATION CONTEXT:")
                print(f"Query: {query[:100]}{'...' if len(query) > 100 else ''}")
                print(f"Ground truth: {ground_truth[:100]}{'...' if len(ground_truth) > 100 else ''}")
                print(f"System answer: {system_answer[:100]}{'...' if len(system_answer) > 100 else ''}")
                print(f"\n📤 RAW API RESPONSE:")
                print(f"{'─'*60}")
                print(repr(result_text[:500]))
                print(f"{'─'*60}")
                if len(result_text) > 500:
                    print(f"Last 200 characters of response:")
                    print(f"{'─'*60}")
                    print(repr(result_text[-200:]))
                    print(f"{'─'*60}")
                print(f"\n📄 FULL RESPONSE (if under 1000 chars):")
                if len(result_text) <= 1000:
                    print(result_text)
                else:
                    print(f"[Response too long - {len(result_text)} chars total]")
                print(f"{'='*60}\n")

                if use_structured_output:
                    # In structured output mode, this shouldn't happen - fall back
                    print("🔄 Structured output failed, attempting fallback to basic JSON mode")
                    return call_llm_judge(client, query, ground_truth, system_answer, max_retries, use_structured_output=False, debug=debug)
                else:
                    # In fallback mode, retry or fail
                    if attempt == max_retries - 1:
                        return JudgeResponse(correct=0, reasoning=f"JSON parsing failed: {e}")
                    continue

            except Exception as e:
                print(f"\n{'='*60}")
                print(f"🚨 PYDANTIC VALIDATION FAILED - ATTEMPT {attempt + 1}")
                print(f"{'='*60}")
                print(f"Error: {e}")
                print(f"Raw response that was successfully parsed as JSON:")
                print(f"{'─'*60}")
                print(f"Parsed JSON: {result_json if 'result_json' in locals() else 'Failed to parse'}")
                print(f"{'─'*60}")
                print(f"Original response text:")
                print(result_text)
                print(f"{'='*60}\n")
                if attempt == max_retries - 1:
                    return JudgeResponse(correct=0, reasoning=f"Validation failed: {e}")
                continue

        except Exception as e:
            error_msg = str(e)

            # Check if this is a structured output compatibility issue
            if use_structured_output and ("json_schema" in error_msg.lower() or "schema" in error_msg.lower() or "structured" in error_msg.lower()):
                print(f"\n🔄 Structured output not supported, falling back to basic JSON mode")
                print(f"Original error: {e}")
                return call_llm_judge(client, query, ground_truth, system_answer, max_retries, use_structured_output=False, debug=debug)

            # Handle other API errors (network, auth, etc.)
            print(f"\n{'='*60}")
            print(f"🚨 API CALL FAILED - ATTEMPT {attempt + 1}")
            print(f"{'='*60}")
            print(f"Error type: {type(e).__name__}")
            print(f"Error message: {e}")
            print(f"Using structured output: {use_structured_output}")
            print(f"Model: openai/gpt-oss-20b")
            if hasattr(e, 'response'):
                print(f"HTTP status: {getattr(e.response, 'status_code', 'Unknown')}")
                print(f"Response headers: {getattr(e.response, 'headers', 'Unknown')}")
                try:
                    print(f"Response body: {e.response.text if hasattr(e.response, 'text') else 'No body'}")
                except:
                    print("Could not read response body")
            print(f"{'='*60}\n")

            if attempt == max_retries - 1:
                return JudgeResponse(correct=0, reasoning=f"API call failed: {e}")
            time.sleep(2**attempt)  # Exponential backoff

    return JudgeResponse(correct=0, reasoning="Max retries exceeded")


def evaluate_system(evaluation_data: EvaluationData, api_key: str, debug: bool = False) -> EvaluationResult:
    """Evaluate the entire system using LLM judge"""
    client = OpenAI(
        base_url="https://openrouter.ai/api/v1",
        api_key=api_key,
    )

    detailed_results = []
    correct_count = 0
    total_count = len(evaluation_data.items)

    print(f"Starting evaluation of {total_count} questions...")
    if debug:
        print("Debug mode enabled - detailed logging will be shown")

    for i, item in enumerate(evaluation_data.items, 1):
        print(f"Evaluating question {i}/{total_count}...")

        # Call LLM judge
        judge_result = call_llm_judge(client, item.query, item.answer, item.result, debug=debug)

        # Track results
        if judge_result.correct == 1:
            correct_count += 1

        detailed_results.append(
            {
                "question_id": i,
                "query": item.query,
                "ground_truth": item.answer,
                "system_answer": item.result,
                "page": item.page,
                "correct": judge_result.correct,
                "reasoning": judge_result.reasoning,
            }
        )

        # Brief delay to avoid rate limiting
        time.sleep(0.5)

    accuracy = correct_count / total_count if total_count > 0 else 0.0

    return EvaluationResult(
        total_questions=total_count,
        correct_answers=correct_count,
        accuracy=accuracy,
        detailed_results=detailed_results,
    )


def print_results(results: EvaluationResult):
    """Print evaluation results in a formatted way"""
    print("\n" + "=" * 60)
    print("EVALUATION RESULTS")
    print("=" * 60)
    print(f"Total Questions: {results.total_questions}")
    print(f"Correct Answers: {results.correct_answers}")
    print(f"Accuracy: {results.accuracy:.2%}")
    print("=" * 60)

    print("\nDETAILED RESULTS:")
    print("-" * 60)

    for result in results.detailed_results:
        status = "✓ CORRECT" if result["correct"] == 1 else "✗ INCORRECT"
        print(f"\nQuestion {result['question_id']}: {status}")
        print(f"Query: {result['query']}")
        print(f"Ground Truth: {result['ground_truth']}")
        print(f"System Answer: {result['system_answer']}")
        print(f"Judge Reasoning: {result['reasoning']}")
        print(f"Page Reference: {result['page']}")
        print("-" * 60)


def save_results(results: EvaluationResult, output_path: str):
    """Save detailed results to JSON file"""
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(results.model_dump(), f, indent=2, ensure_ascii=False)
    print(f"\nDetailed results saved to: {output_path}")


def main():
    """Main evaluation function"""
    # Default paths
    default_data_path = "evaluation_data.json"
    default_output_path = "evaluation_results.json"

    # Check for command line arguments
    data_path = sys.argv[1] if len(sys.argv) > 1 else default_data_path
    output_path = sys.argv[2] if len(sys.argv) > 2 else default_output_path

    # Get API key from environment
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        print("Error: OPENAI_API_KEY environment variable not set")
        print("Please add your OpenRouter API key to your .env file")
        sys.exit(1)

    try:
        # Load evaluation data
        print(f"Loading evaluation data from: {data_path}")
        evaluation_data = load_evaluation_data(data_path)
        print(f"Loaded {len(evaluation_data.items)} evaluation items")

        # Check for debug mode
        debug_mode = os.getenv("EVALUATION_DEBUG", "false").lower() == "true"

        # Run evaluation
        results = evaluate_system(evaluation_data, api_key, debug=debug_mode)

        # Print results
        print_results(results)

        # Save detailed results
        save_results(results, output_path)

    except Exception as e:
        print(f"Error during evaluation: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
