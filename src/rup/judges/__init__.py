from .base import SafetyJudge
from .llm_judge import LLMJudge, KeywordJudge


def get_judge(judge_type: str, model=None) -> SafetyJudge:
    """
    Instantiate a judge by type.

    Args:
        judge_type: 'llm' | 'keyword'
        model:      BaseModel instance (required for 'llm')
    """
    if judge_type == "llm":
        if model is None:
            raise ValueError("An LLM judge requires a model instance.")
        return LLMJudge(model)
    elif judge_type == "keyword":
        return KeywordJudge()
    else:
        raise ValueError(f"Unknown judge type: {judge_type!r}. Use 'llm' or 'keyword'.")


__all__ = ["SafetyJudge", "LLMJudge", "KeywordJudge", "get_judge"]
