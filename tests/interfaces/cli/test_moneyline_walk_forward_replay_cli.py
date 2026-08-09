"""CLI tests for bounded P20A replay."""

from pathlib import Path
import sys
import tempfile
import unittest


REPOSITORY_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPOSITORY_ROOT / "src"))

from match_analysis.interfaces.cli.moneyline_walk_forward_replay import main


FIXTURE = (
    REPOSITORY_ROOT
    / "data"
    / "fixtures"
    / "p20a_p13_walk_forward"
    / "fold_wf_001.json"
)
MODEL_STATE = (
    REPOSITORY_ROOT
    / "report"
    / "p20a_p13_walk_forward_reconstruction"
    / "reconstructed_model.json"
)


class MoneylineWalkForwardReplayCliTests(unittest.TestCase):
    def test_cli_replays_verified_model_state_deterministically(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            first = root / "first"
            second = root / "second"
            base_args = [
                "--fold-fixture",
                str(FIXTURE),
                "--reconstructed-model",
                str(MODEL_STATE),
            ]
            self.assertEqual(
                main([*base_args, "--output-dir", str(first)]),
                0,
            )
            self.assertEqual(
                main([*base_args, "--output-dir", str(second)]),
                0,
            )
            for path in first.iterdir():
                self.assertEqual(path.read_bytes(), (second / path.name).read_bytes())

    def test_missing_fixture_does_not_create_output(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            output = root / "output"
            self.assertEqual(
                main(
                    [
                        "--fold-fixture",
                        str(root / "missing.json"),
                        "--output-dir",
                        str(output),
                    ]
                ),
                1,
            )
            self.assertFalse(output.exists())


if __name__ == "__main__":
    unittest.main()
