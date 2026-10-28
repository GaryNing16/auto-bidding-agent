"""Apply a documented wording correction without changing frozen experiment code.

Run after harmony_baselines.report. This clarifies incomplete wall-clock
accounting, an empty failure log, and reference-normalized low-tail results in
the generated Markdown; scores, figures and model selection stay intact.
"""
from pathlib import Path
import argparse
import csv
import hashlib
import json

ROOT = Path(__file__).resolve().parents[1]


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_json(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=ROOT / "outputs/baseline_comparison_20261028")
    out = parser.parse_args().root.resolve()
    accounting = json.loads((out / "runtime_accounting.json").read_text(encoding="utf-8"))
    status_path = out / "report.status.json"
    status = json.loads(status_path.read_text(encoding="utf-8"))
    assert status["status"] == "completed", "generate and verify the report first"
    report = Path(status["report"])
    before = digest(report)
    assert before == status["report_sha256"], "report changed outside its recorded generation"
    assert digest(out / "episode_results.csv") == status["source_csv_sha256"]
    if accounting["coverage_complete"]:
        print(json.dumps({"annotation_needed": False}))
        return
    text = report.read_text(encoding="utf-8")
    marker = "有记录的评测执行段合计约"
    if marker in text:
        assert "document_postprocessing" in status
        print(json.dumps({"already_applied": True, "report_sha256": before}))
        return
    old = "各实验阶段墙钟合计约"
    assert text.count(old) == 1
    first = text.replace(old, marker)
    old_resource = "CPU运行，具体配置及资源见"
    assert first.count(old_resource) == 1
    note = ("CPU运行。最初576局验证及早期中断等待的耗时未完整保存，"
            "上述评测时间不代表端到端总耗时；详见[时间记录范围]("
            + (out / "runtime_accounting.json").as_posix() + ")。具体配置及资源见")
    corrected = first.replace(old_resource, note)
    command = "运行`.venv-baselines/Scripts/python.exe -m harmony_baselines.report`可重建本报告。"
    assert corrected.count(command) == 1
    corrected = corrected.replace(command,
        "依次运行`.venv-baselines/Scripts/python.exe -m harmony_baselines.report`与"
        "`.venv-baselines/Scripts/python.exe scripts/annotate_baseline_report_runtime.py`"
        "可重建本报告及文档说明。")
    changes = ["execution-time label", "missing-duration explanation and evidence link", "reproduction command"]
    failure_log = json.loads((out / "evaluation_failures.json").read_text(encoding="utf-8"))
    if failure_log == []:
        completion = json.loads((out / "pipeline_completion_audit.json").read_text(encoding="utf-8"))
        assert completion["status"] == "passed" and completion["failure_episodes"] == 0
        old_failure = "保留未完成角色及seed。"
        assert corrected.count(old_failure) == 1
        corrected = corrected.replace(old_failure,
            f"本轮为空：最终测试{completion['successful_episodes']:,}局全部完成。")
        changes.append("empty final-test failure log clarified")
    with (out / "robustness.csv").open(encoding="utf-8", newline="") as handle:
        robust = {row["algorithm"]: row for row in csv.DictReader(handle) if float(row["budget_scale"]) == 1.0}
    neural_names = {"BC", "IQL", "BCQ", "TD3BC", "CQL", "DT"}
    low_tail = lambda name: float(robust[name]["mean_role_p10"])
    nonreference = max((n for n in robust if n != "LocalPID"), key=low_tail)
    neural = max((n for n in robust if n in neural_names), key=low_tail)
    average = max(robust, key=lambda n: float(robust[n]["scenario_mean"]))
    labels = {"FixedAlpha": "Fixed α", "TD3BC": "TD3+BC", "OnlineLP": "Online LP"}
    old_tail = ("平均角色P10最高的是 **LocalPID**（1.000），可作为角色间稳健性的一个指标；"
                "所有角色均有训练数据，这不是“未见过角色”的泛化实验。")
    assert corrected.count(old_tail) == 1
    new_tail = ("LocalPID的R及P10恒为1，来自自归一化。其余策略中，**"
        f"{labels.get(nonreference, nonreference)}**的跨场景平均角色P10最高（{low_tail(nonreference):.3f}）；"
        f"神经策略以**{labels.get(neural, neural)}**最高（{low_tail(neural):.3f}）。"
        f"平均成绩冠军{labels.get(average, average)}的角色P10为{low_tail(average):.3f}，"
        "其平均优势没有覆盖所有角色。所有角色均有训练数据，本实验不能证明未见角色泛化。")
    corrected = corrected.replace(old_tail, new_tail)
    changes.append("reference normalization and non-reference role-P10 comparison clarified from robustness.csv")
    original = out / "report_before_runtime_annotation.md"
    if original.exists():
        assert digest(original) == before, "preserve a different prior report rather than overwrite it"
    else:
        original.write_bytes(report.read_bytes())
    report.write_text(corrected, encoding="utf-8")
    audit_path = out / "report_runtime_annotation.json"
    audit = {
        "purpose": "clarify runtime coverage, empty test failures, and reference-normalized role tails; no metric or figure changes",
        "frozen_report_generator_modified": False,
        "report": str(report), "original_report": str(original),
        "original_report_sha256": before, "annotated_report_sha256": digest(report),
        "accounting_sha256": digest(out / "runtime_accounting.json"),
        "script": str(Path(__file__).resolve()), "script_sha256": digest(Path(__file__).resolve()),
        "source_csv_sha256": status["source_csv_sha256"],
        "robustness_csv_sha256": digest(out / "robustness.csv"),
        "changes": changes,
    }
    write_json(audit_path, audit)
    status["report_sha256"] = digest(report)
    status["document_postprocessing"] = {"audit": str(audit_path), "sha256": digest(audit_path)}
    write_json(status_path, status)
    print(json.dumps({"annotation_applied": True, "report_sha256": digest(report)}, ensure_ascii=False))


if __name__ == "__main__":
    main()
