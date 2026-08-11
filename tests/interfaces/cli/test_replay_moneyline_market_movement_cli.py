from __future__ import annotations

from pathlib import Path
import unittest

from match_analysis.interfaces.cli.replay_moneyline_market_movement import main


REPOSITORY_ROOT = Path(__file__).resolve().parents[3]
RUNTIME_ROOT = Path("/tmp/matchanalysis-p29a-market-movement/cli-test")


class ReplayMoneylineMarketMovementCliTests(unittest.TestCase):
    def test_offline_replay_writes_exact_three_artifacts(self) -> None:
        output_dir = RUNTIME_ROOT / "one"
        status = main(
            [
                "--repository-root",
                str(REPOSITORY_ROOT),
                "--offline",
                "--output-dir",
                str(output_dir),
            ]
        )
        self.assertEqual(status, 0)
        self.assertEqual(
            sorted(path.name for path in output_dir.iterdir()),
            ["closing_prices.jsonl", "market_movement.jsonl", "summary.json"],
        )

    def test_network_mode_is_not_available(self) -> None:
        self.assertEqual(main(["--repository-root", str(REPOSITORY_ROOT)]), 1)


if __name__ == "__main__":
    unittest.main()
