from copy import deepcopy
from hashlib import sha256
import json
from pathlib import Path
import unittest

from match_analysis.application.use_cases.adapt_legacy_bullpen_evidence import adapt_legacy_bullpen_evidence
from match_analysis.application.use_cases.convert_official_bullpen_capture import baseball_innings_to_outs, convert_official_bullpen_capture


ROOT = Path(__file__).resolve().parents[2]
CAPTURE = ROOT / "examples/method_loop_v1/official_capture"


class OfficialBullpenCaptureTests(unittest.TestCase):
    def setUp(self):
        self.raw = (CAPTURE/"game_823218.json").read_bytes()
        self.meta = json.loads((CAPTURE/"capture.json").read_bytes())
        self.schedule = json.loads((CAPTURE/"source_schedule.json").read_bytes())

    def changed(self, mutate):
        feed = json.loads(self.raw)
        mutate(feed)
        raw = json.dumps(feed).encode()
        meta = {**self.meta, "sha256": sha256(raw).hexdigest(), "bytes": len(raw)}
        return convert_official_bullpen_capture(raw, meta, self.schedule)

    def test_real_official_game_outs_pitches_and_local_usage_day(self):
        result = convert_official_bullpen_capture(self.raw, self.meta, self.schedule)
        self.assertEqual(result["diagnostics"]["team_totals"], {
            "home": {"pitchers": 3, "relief_outs": 3, "relief_pitches": 25},
            "away": {"pitchers": 5, "relief_outs": 24, "relief_pitches": 118}})
        self.assertEqual(result["diagnostics"]["usage_date"], "2026-06-08")
        self.assertEqual(result["capture_evidence"]["coverage"], [])
        cached = json.loads((ROOT/"data/fixtures/p23f2_official_2026_history/raw/boxscores/823218.json").read_bytes())
        for row in map(json.loads, result["source_bytes"].splitlines()):
            side = "home" if row["team"] == self.schedule["home_team"]["name"] else "away"
            stats = cached["teams"][side]["players"][f"ID{row['pitcher_id']}"]["stats"]["pitching"]
            self.assertEqual(row["outs_recorded"], baseball_innings_to_outs(stats["inningsPitched"]))
            self.assertEqual(row["pitches_thrown"], stats["numberOfPitches"])
            self.assertIsNone(row["high_leverage"])

    def test_today_capture_cannot_become_historical_pregame_feature(self):
        result = convert_official_bullpen_capture(self.raw, self.meta, self.schedule)
        target = json.loads((CAPTURE/"historical_target.json").read_bytes())
        admission = adapt_legacy_bullpen_evidence(result["source_bytes"], source_format="phase63-jsonl",
                                                  target=target, evidence=result["capture_evidence"])
        self.assertEqual(admission["selected_relief_appearance_count"], 0)
        self.assertEqual(admission["decision_counts"], {
            "FIRST_PITCHER_OR_OPENER_NOT_BULLPEN": 2, "NOT_AVAILABLE_BEFORE_CUTOFF": 6})
        self.assertEqual(admission, json.loads((CAPTURE/"historical_admission.json").read_bytes()))

    def test_baseball_fraction_is_not_decimal_and_no_float_reconstruction(self):
        self.assertEqual(baseball_innings_to_outs("1.2"), 5)
        self.assertEqual(baseball_innings_to_outs("0.1"), 1)
        for value in (1.2, "1.3", "1.67", "-1.0", "1", None):
            with self.assertRaises(ValueError): baseball_innings_to_outs(value)

    def test_wrong_hash_wrong_identity_and_nonfinal_rejected(self):
        with self.assertRaisesRegex(ValueError, "hash"):
            convert_official_bullpen_capture(self.raw+b" ", self.meta, self.schedule)
        with self.assertRaisesRegex(ValueError, "team identity"):
            self.changed(lambda f: f["gameData"]["teams"]["home"].update(id=147))
        with self.assertRaisesRegex(ValueError, "final regular"):
            self.changed(lambda f: f["gameData"]["status"].update(detailedState="Suspended"))

    def test_multi_day_game_and_missing_time_do_not_invent_usage_date(self):
        with self.assertRaisesRegex(ValueError, "multi-day"):
            self.changed(lambda f: f["liveData"]["plays"]["allPlays"][-1]["about"].update(
                startTime="2026-06-10T01:00:00Z", endTime="2026-06-10T01:01:00Z"))
        with self.assertRaises(KeyError):
            self.changed(lambda f: f["liveData"]["plays"]["allPlays"][0]["about"].pop("startTime"))

    def test_season_statistics_never_substitute_and_starter_conflict_rejected(self):
        def change(f):
            box = f["liveData"]["boxscore"]["teams"]["home"]
            box["players"][f"ID{box['pitchers'][1]}"]["seasonStats"] = {"pitching": {"inningsPitched": "999.0"}}
        result = self.changed(change)
        self.assertEqual(result["diagnostics"]["team_totals"]["home"]["relief_outs"], 3)
        def conflict(f):
            box = f["liveData"]["boxscore"]["teams"]["home"]
            box["players"][f"ID{box['pitchers'][1]}"]["stats"]["pitching"]["gamesStarted"] = 1
        with self.assertRaisesRegex(ValueError, "starter"):
            self.changed(conflict)


if __name__ == "__main__":
    unittest.main()
