"""CLI integration tests for prediction feedback ledger generation."""

import hashlib
import json
from pathlib import Path
import tempfile
import unittest

from match_analysis.interfaces.cli.prediction_feedback_ledger import main


class TestPredictionFeedbackLedgerCLI(unittest.TestCase):
    def setUp(self) -> None:
        self.p15c_snapshot = Path("report/p15c_admitted_prediction_observation_snapshot/admitted_observations.jsonl")
        self.p15c_summary = Path("report/p15c_admitted_prediction_observation_snapshot/summary.json")
        self.p16a_attachments = Path("report/p16a_final_result_attachment/attachments.jsonl")
        self.p16a_summary = Path("report/p16a_final_result_attachment/summary.json")
        self.p16b_evaluations = Path("report/p16b_prediction_evaluation_scorecard/evaluations.jsonl")
        self.p16b_summary = Path("report/p16b_prediction_evaluation_scorecard/summary.json")

    def _run_cli(self, out_dir: Path, **overrides: Path) -> int:
        args_dict = {
            "prediction_snapshot": self.p15c_snapshot,
            "prediction_summary": self.p15c_summary,
            "result_attachments": self.p16a_attachments,
            "result_summary": self.p16a_summary,
            "evaluations": self.p16b_evaluations,
            "evaluation_summary": self.p16b_summary,
        }
        args_dict.update(overrides)

        argv = [
            "--prediction-snapshot", str(args_dict["prediction_snapshot"]),
            "--prediction-summary", str(args_dict["prediction_summary"]),
            "--result-attachments", str(args_dict["result_attachments"]),
            "--result-summary", str(args_dict["result_summary"]),
            "--evaluations", str(args_dict["evaluations"]),
            "--evaluation-summary", str(args_dict["evaluation_summary"]),
            "--output-dir", str(out_dir),
        ]
        return main(argv)

    def test_cli_valid_complete_lineage(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            out_dir = Path(tmp_dir) / "output"
            code = self._run_cli(out_dir)
            self.assertEqual(code, 0)
            self.assertTrue((out_dir / "feedback.jsonl").exists())
            self.assertTrue((out_dir / "summary.json").exists())
            self.assertTrue((out_dir / "report.md").exists())

    def test_cli_deterministic_repeated_execution(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            out_dir1 = Path(tmp_dir) / "output1"
            out_dir2 = Path(tmp_dir) / "output2"

            code1 = self._run_cli(out_dir1)
            code2 = self._run_cli(out_dir2)
            self.assertEqual(code1, 0)
            self.assertEqual(code2, 0)

            self.assertEqual(
                (out_dir1 / "feedback.jsonl").read_bytes(),
                (out_dir2 / "feedback.jsonl").read_bytes(),
            )
            self.assertEqual(
                (out_dir1 / "summary.json").read_bytes(),
                (out_dir2 / "summary.json").read_bytes(),
            )
            self.assertEqual(
                (out_dir1 / "report.md").read_bytes(),
                (out_dir2 / "report.md").read_bytes(),
            )

    def test_cli_malformed_json_returns_error(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            bad_file = Path(tmp_dir) / "bad.jsonl"
            bad_file.write_text("{bad json}\n", encoding="utf-8")
            out_dir = Path(tmp_dir) / "output"

            code = self._run_cli(out_dir, evaluations=bad_file)
            self.assertEqual(code, 1)
            self.assertFalse(out_dir.exists())

    def test_cli_duplicate_json_key_returns_error(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            bad_file = Path(tmp_dir) / "dup_key.jsonl"
            bad_file.write_text('{"a": 1, "a": 2}\n', encoding="utf-8")
            out_dir = Path(tmp_dir) / "output"

            code = self._run_cli(out_dir, evaluations=bad_file)
            self.assertEqual(code, 1)
            self.assertFalse(out_dir.exists())

    def test_cli_duplicate_prediction_id_returns_error(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            content = self.p15c_snapshot.read_bytes()
            lines = content.strip().split(b"\n")
            dup_content = content + b"\n" + lines[0] + b"\n"
            dup_file = Path(tmp_dir) / "dup_snap.jsonl"
            dup_file.write_text(dup_content.decode("utf-8"), encoding="utf-8")

            # Update summary SHA
            sum_dict = json.loads(self.p15c_summary.read_text(encoding="utf-8"))
            sum_dict["admitted_observations_jsonl_sha256"] = hashlib.sha256(dup_content).hexdigest()
            sum_file = Path(tmp_dir) / "snap_sum.json"
            sum_file.write_text(json.dumps(sum_dict, indent=2), encoding="utf-8")

            # Update P16A summary expectations
            att_sum_dict = json.loads(self.p16a_summary.read_text(encoding="utf-8"))
            att_sum_dict["source_snapshot_sha256"] = hashlib.sha256(dup_content).hexdigest()
            att_sum_dict["source_snapshot_summary_sha256"] = hashlib.sha256(sum_file.read_bytes()).hexdigest()
            att_sum_file = Path(tmp_dir) / "att_sum.json"
            att_sum_file.write_text(json.dumps(att_sum_dict, indent=2), encoding="utf-8")

            out_dir = Path(tmp_dir) / "output"
            code = self._run_cli(
                out_dir,
                prediction_snapshot=dup_file,
                prediction_summary=sum_file,
                result_summary=att_sum_file,
            )
            self.assertEqual(code, 1)
            self.assertFalse(out_dir.exists())

    def test_cli_summary_fingerprint_mismatch_returns_error(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            att_sum_dict = json.loads(self.p16a_summary.read_text(encoding="utf-8"))
            att_sum_dict["source_snapshot_fingerprint"] = "0000000000000000000000000000000000000000000000000000000000000000"
            bad_att_sum = Path(tmp_dir) / "bad_att_sum.json"
            bad_att_sum.write_text(json.dumps(att_sum_dict, indent=2), encoding="utf-8")

            # Update P16B summary to match the bad P16A summary hash
            eval_sum_dict = json.loads(self.p16b_summary.read_text(encoding="utf-8"))
            eval_sum_dict["source_summary_sha256"] = hashlib.sha256(bad_att_sum.read_bytes()).hexdigest()
            bad_eval_sum = Path(tmp_dir) / "bad_eval_sum.json"
            bad_eval_sum.write_text(json.dumps(eval_sum_dict, indent=2), encoding="utf-8")

            out_dir = Path(tmp_dir) / "output"
            code = self._run_cli(
                out_dir,
                result_summary=bad_att_sum,
                evaluation_summary=bad_eval_sum,
            )
            self.assertEqual(code, 1)
            self.assertFalse(out_dir.exists())

    def test_cli_missing_file_returns_error(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            out_dir = Path(tmp_dir) / "output"
            code = self._run_cli(
                out_dir,
                prediction_snapshot=Path(tmp_dir) / "non_existent.jsonl",
            )
            self.assertEqual(code, 1)
            self.assertFalse(out_dir.exists())


if __name__ == "__main__":
    unittest.main()
