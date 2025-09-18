from typing import List
from pydantic import BaseModel, Field, field_validator


class EvaluationItem(BaseModel):
    """Individual test case for evaluation"""

    query: str = Field(..., description="The user question")
    answer: str = Field(..., description="Ground truth answer")
    page: str = Field(..., description="Page reference from manual")
    result: str = Field(..., description="System-generated answer to evaluate")


class JudgeResponse(BaseModel):
    """Structured response from LLM judge"""

    correct: int = Field(
        ...,
        description="Binary correctness score: 1 for correct, 0 for incorrect",
        ge=0,  # Greater than or equal to 0
        le=1   # Less than or equal to 1
    )
    reasoning: str = Field(
        ...,
        description="Explanation of the scoring decision"
    )

    @field_validator('reasoning')
    @classmethod
    def truncate_reasoning(cls, v: str) -> str:
        """Truncates the reasoning string to 200 characters."""
        if len(v) > 200:
            return v[:200]
        return v

    @classmethod
    def get_json_schema_for_openrouter(cls) -> dict:
        """Generate JSON Schema for OpenRouter structured outputs"""
        schema = cls.model_json_schema()

        # Enhance schema for OpenRouter structured outputs
        enhanced_schema = {
            "type": "object",
            "properties": {
                "correct": {
                    "type": "integer",
                    "minimum": 0,
                    "maximum": 1,
                    "description": "Binary correctness score: 1 for correct, 0 for incorrect"
                },
                "reasoning": {
                    "type": "string",
                    "maxLength": 200,
                    "description": "Brief explanation of the scoring decision citing which essential elements matched or were missing"
                }
            },
            "required": ["correct", "reasoning"],
            "additionalProperties": False
        }

        return enhanced_schema

    @classmethod
    def get_openrouter_response_format(cls) -> dict:
        """Get the complete response_format parameter for OpenRouter API calls"""
        return {
            "type": "json_schema",
            "json_schema": {
                "name": "judge_response",
                "strict": True,
                "schema": cls.get_json_schema_for_openrouter()
            }
        }


class EvaluationResult(BaseModel):
    """Summary of evaluation results"""

    total_questions: int = Field(..., description="Total number of questions evaluated")
    correct_answers: int = Field(..., description="Number of correct answers")
    accuracy: float = Field(..., description="Overall accuracy score (0-1)")
    detailed_results: List[dict] = Field(
        ..., description="Per-question results with judge reasoning"
    )


class EvaluationData(BaseModel):
    """Container for all evaluation items"""

    items: List[EvaluationItem] = Field(
        ..., description="List of evaluation test cases"
    )
