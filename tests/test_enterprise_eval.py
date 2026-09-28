import json
from pathlib import Path

from atlasnav.benchmarks.enterprise_eval import evaluate_frozen


def test_frozen_enterprise_evaluation_preserves_answer_metrics(tmp_path: Path) -> None:
    questions = tmp_path / "questions.jsonl"
    answers = tmp_path / "answers.jsonl"
    results = tmp_path / "results.json"
    questions.write_text(json.dumps({"question_id": "q1", "expected_doc_ids": ["d1"], "valid_doc_ids": ["d1"]}) + "\n")
    answers.write_text(json.dumps({"question_id": "q1", "answer": "x", "document_ids": ["d1"]}) + "\n")
    results.write_text(json.dumps({"questions": [{"question_id": "q1", "answer_correct": True, "completeness_pct": 100.0}]}))
    report = evaluate_frozen(questions=questions, answers=answers, results=results, output=tmp_path / "out")
    assert report["aggregate"] == {
        "questions": 1, "correctness_percent": 100.0, "completeness_percent": 100.0,
        "overall": 100.0, "document_recall_percent": 100.0, "invalid_extra_documents": 0.0,
    }
