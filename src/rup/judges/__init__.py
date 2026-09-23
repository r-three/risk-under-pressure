from ..utils.logging import get_logger
from .base import SafetyJudge
from .flow_judge import FlowJudge, flow_prompt, flow_rubric_sha1, parse_flow_score
from .llm_judge import (
    JudgeVerdict,
    KeywordJudge,
    LLMJudge,
    judge_inputs,
    parse_verdict,
    rubric_sha1,
    rubric_text,
)
from .severity_judge import (
    DEFAULT_SEVERITY_RUBRIC,
    SEVERITY_LABELS,
    SEVERITY_MAX,
    SEVERITY_MIN,
    SeverityJudge,
    SeverityResult,
    parse_severity,
    severity_rubric_sha1,
    severity_rubric_text,
)

_logger = get_logger(__name__)


def get_judge(judge_type: str, model=None, **judge_kwargs) -> SafetyJudge:
    """
    Instantiate a judge by type.

    Args:
        judge_type:   'llm' | 'flow' | 'keyword'
        model:        BaseModel instance (required for 'llm' and 'flow')
        judge_kwargs: forwarded to the judge class (max_new_tokens, escalate_max_new_tokens, ...)

    'flow' selects Flow-Judge-v0.1, which emits <score>N</score> rather than SAFE/UNSAFE and so
    needs its own prompt and parser. Prefer judge_from_config() over calling this directly with a
    hardcoded type — the right kind is a property of the checkpoint, not of the call site.
    """
    if judge_type == "llm":
        if model is None:
            raise ValueError("An LLM judge requires a model instance.")
        return LLMJudge(model, **judge_kwargs)
    elif judge_type == "flow":
        if model is None:
            raise ValueError("A Flow judge requires a model instance.")
        return FlowJudge(model, **judge_kwargs)
    elif judge_type == "keyword":
        if judge_kwargs:
            raise ValueError(f"KeywordJudge takes no options; got {sorted(judge_kwargs)}.")
        return KeywordJudge()
    else:
        raise ValueError(
            f"Unknown judge type: {judge_type!r}. Use 'llm', 'flow', or 'keyword'."
        )


def judge_kind_for(config) -> str:
    """Which judge implementation a model config wants.

    Read from `extra.judge_kind` so adding a judge with a different I/O contract is a YAML change,
    not a code change at every call site. Defaults to 'llm' (the SAFE/UNSAFE protocol), which is
    what every judge before Flow-Judge used.
    """
    return (getattr(config, "extra", None) or {}).get("judge_kind", "llm")


def judge_from_config(config, model, **judge_kwargs) -> SafetyJudge:
    """Build the judge a model config asks for.

    Options that do not apply to the selected implementation are dropped rather than raising:
    callers pass a single set of CLI flags (--judge-rubric, --judge-max-new-tokens) without
    knowing which judge they will get, and a flag that is meaningless for Flow-Judge should be
    ignored, not fatal. Anything dropped is logged so a silently ineffective flag is visible.
    """
    kind = judge_kind_for(config)
    cls = {"llm": LLMJudge, "flow": FlowJudge}.get(kind)
    if cls is None:
        raise ValueError(
            f"Config {getattr(config, 'model_id', '?')!r} requests judge_kind={kind!r}, "
            f"which is not implemented. Use 'llm' or 'flow'."
        )

    import inspect

    accepted = set(inspect.signature(cls.__init__).parameters) - {"self", "model"}
    used = {k: v for k, v in judge_kwargs.items() if k in accepted}
    dropped = sorted(set(judge_kwargs) - set(used))
    if dropped:
        _logger.info(
            f"judge_kind={kind}: ignoring option(s) {dropped} — not applicable to "
            f"{cls.__name__}."
        )
    return cls(model, **used)


__all__ = [
    "SafetyJudge",
    "LLMJudge",
    "JudgeVerdict",
    "KeywordJudge",
    "get_judge",
    "parse_verdict",
    "judge_inputs",
    "rubric_text",
    "rubric_sha1",
    "FlowJudge",
    "flow_prompt",
    "flow_rubric_sha1",
    "parse_flow_score",
    "judge_from_config",
    "judge_kind_for",
    "SeverityJudge",
    "SeverityResult",
    "parse_severity",
    "SEVERITY_MIN",
    "SEVERITY_MAX",
    "SEVERITY_LABELS",
    "severity_rubric_text",
    "severity_rubric_sha1",
    "DEFAULT_SEVERITY_RUBRIC",
]
