"""Schema-compatibility tests for the results-tree record types.

Results trees outlive the code that wrote them and are read by whatever checkout happens to be
on the node. Two directions have to keep working:

  backward — a file written before the judge-audit fields existed (~150 of them on disk) must
             still load, with the new fields defaulting to None.
  forward  — a file written by a NEWER version, carrying fields this checkout has never heard
             of, must load with a warning rather than raising TypeError out of StepResult(**s)
             and taking down an unrelated evaluation job.
"""

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from rup.utils import io as rup_io
from rup.utils.io import StepResult, TrialRecord, read_jsonl, write_jsonl


def _record(**step_overrides):
    step = dict(step=1, prompt="p", response="r", judgment=1, seconds=0.5)
    step.update(step_overrides)
    return dict(
        prompt_id="jbb_0000", base_prompt="bp", behavior="b", category="c",
        source="jailbreakbench", model_id="m", attack_id="pair", budget=10,
        steps=[step], success=True, first_success_step=1, final_prompt="fp",
        timestamp=1.0, metadata={},
    )


def test_pre_audit_file_still_loads_with_none_defaults():
    rec = TrialRecord.from_dict(_record())

    step = rec.steps[0]
    assert step.judgment == 1
    assert step.judge_parsed is None
    assert step.judge_branch is None
    assert step.judge_raw is None
    assert step.judge_out_tokens is None


def test_audit_fields_round_trip():
    rec = TrialRecord.from_dict(_record(
        judge_parsed=True, judge_branch="first_line", judge_raw="UNSAFE", judge_out_tokens=3,
    ))
    again = TrialRecord.from_dict(json.loads(json.dumps(rec.to_dict())))

    step = again.steps[0]
    assert (step.judge_parsed, step.judge_branch, step.judge_raw, step.judge_out_tokens) == (
        True, "first_line", "UNSAFE", 3
    )


def test_unknown_step_field_warns_and_loads(caplog):
    rup_io._warned_unknown_fields.clear()
    with caplog.at_level("WARNING", logger="rup.utils.io"):
        rec = TrialRecord.from_dict(_record(judge_confidence=0.97))

    assert rec.steps[0].judgment == 1
    assert "judge_confidence" in caplog.text
    assert "newer version" in caplog.text


def test_unknown_record_field_warns_and_loads(caplog):
    rup_io._warned_unknown_fields.clear()
    d = _record()
    d["rejudge_provenance"] = {"judge": "olmo3-7b-instruct"}

    with caplog.at_level("WARNING", logger="rup.utils.io"):
        rec = TrialRecord.from_dict(d)

    assert rec.prompt_id == "jbb_0000"
    assert "rejudge_provenance" in caplog.text


def test_unknown_field_warning_is_emitted_once_per_process(caplog):
    rup_io._warned_unknown_fields.clear()
    with caplog.at_level("WARNING", logger="rup.utils.io"):
        for _ in range(5):
            TrialRecord.from_dict(_record(judge_confidence=0.5))

    assert caplog.text.count("judge_confidence") == 1


def test_success_at_reads_only_judgments():
    rec = TrialRecord.from_dict(_record())
    rec.steps = [
        StepResult(step=1, prompt="p", response="r", judgment=0),
        StepResult(step=2, prompt="p", response="r", judgment=1),
    ]
    assert rec.success_at(0) is False
    assert rec.success_at(1) is False
    assert rec.success_at(2) is True


def test_write_then_read_survives_a_round_trip(tmp_path):
    rec = TrialRecord.from_dict(_record(judge_parsed=False, judge_branch="unparsed"))
    path = tmp_path / "results.jsonl"
    write_jsonl([rec], path)

    (loaded,) = list(read_jsonl(path))
    assert loaded.steps[0].judge_branch == "unparsed"
    assert loaded.steps[0].judge_parsed is False
