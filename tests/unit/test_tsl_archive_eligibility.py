from copy import deepcopy
from hashlib import sha256
import json
from pathlib import Path
import tempfile
import unittest

from match_analysis.application.use_cases.audit_tsl_archive_eligibility import audit_tsl_archive_eligibility


def game(pk=101, start="2026-06-14T20:00:00Z", kind="R", final=True):
    return {"gamePk":pk,"gameNumber":1,"season":"2026","officialDate":start[:10],"gameDate":start,
            "gameType":kind,"status":{"abstractGameState":"Final" if final else "Preview",
                                        "detailedState":"Final" if final else "Scheduled"},
            "teams":{"home":{"team":{"id":147,"name":"New York Yankees","abbreviation":"NYY"},"score":99},
                     "away":{"team":{"id":111,"name":"Boston Red Sox","abbreviation":"BOS"},"score":0}}}


def quote(start="2026-06-14T20:00:00Z", stamp="2026-06-14T19:00:00Z", source_id="tsl-1", price="1.80"):
    return {"source":"TSL_BLOB3RD","sport_league":"MLB","is_pregame":True,"game_time":start,"fetched_at":stamp,
            "match_id":source_id,"home_team_name":"紐約洋基","away_team_name":"波士頓紅襪","home_code":"NYY","away_code":"BOS",
            "markets":[{"marketCode":"MNL","outcomes":[{"outcomeName":"紐約洋基","odds":price,"specialBetValue":None},
                                                          {"outcomeName":"波士頓紅襪","odds":"2.10","specialBetValue":None}]}]}


class ArchiveEligibilityTests(unittest.TestCase):
    def audit(self, rows, games=None):
        schedule=json.dumps({"dates":[{"games":games or [game()]}]}).encode()
        raw="".join(json.dumps(row,ensure_ascii=False)+"\n" for row in rows).encode()
        with tempfile.TemporaryDirectory() as directory:
            path=Path(directory)/"source.jsonl";path.write_bytes(raw)
            return audit_tsl_archive_eligibility(path,expected_source_sha256=sha256(raw).hexdigest(),
                       schedule_raw=schedule,expected_schedule_sha256=sha256(schedule).hexdigest(),prior_game_ids={"101"})

    def test_all_rows_latest_quote_and_three_way_not_confused_with_two_way(self):
        earlier=quote(stamp="2026-06-14T17:00:00Z")
        latest=quote();tie=deepcopy(latest["markets"][0]);tie["outcomes"].append({"outcomeName":"和局","odds":"3.0"})
        latest["markets"].append(tie)
        result=self.audit([earlier,latest,deepcopy(latest)])
        self.assertEqual(result["summary"]["source_rows"],3)
        self.assertEqual(len(result["games"]),1)
        selected=result["games"][0]
        self.assertEqual(selected["provider_game_id"],"101")
        self.assertEqual(selected["game_number"],1)
        self.assertEqual(selected["home_team"]["id"],147)
        self.assertEqual(selected["market_observed_at_utc"],latest["fetched_at"])
        self.assertEqual(selected["home_decimal_price"],"1.80")
        self.assertEqual(selected["quote_age_seconds_at_official_start"],3600)
        self.assertNotIn("score",selected["home_team"])
        self.assertTrue(selected["prior_p37_game"])

    def test_winner_scores_never_change_eligibility(self):
        original=game();modified=deepcopy(original)
        modified["teams"]["home"]["score"]=0;modified["teams"]["away"]["score"]=20
        modified["teams"]["away"]["isWinner"]=True
        self.assertEqual(self.audit([quote()],[original])["games"],self.audit([quote()],[modified])["games"])

    def test_wrong_identity_missing_price_and_post_official_start(self):
        wrong=quote();wrong["provider_game_id"]="999"
        missing=quote();missing["markets"][0]["outcomes"][1]["odds"]=None
        late=quote(start="2026-06-14T20:30:00Z",stamp="2026-06-14T20:01:00Z")
        result=self.audit([wrong,missing,late])
        self.assertEqual([d["reason"] for d in result["row_decisions"]],
                         ["SOURCE_EXPLICIT_GAME_IDENTITY_CONFLICT","MALFORMED_OR_INCOMPLETE_PRICE","POST_OFFICIAL_START"])
        self.assertEqual(len(result["games"]),0)

    def test_nonregular_and_nonfinal_not_scored(self):
        for g,reason in [(game(kind="S"),"OFFICIAL_NOT_REGULAR_SEASON"),(game(final=False),"OFFICIAL_NOT_FINAL")]:
            self.assertEqual(self.audit([quote()],[g])["row_decisions"][0]["reason"],reason)

    def test_doubleheader_ambiguity_not_resolved_by_nearest_or_game_order(self):
        second=game(pk=102,start="2026-06-14T20:10:00Z");second["gameNumber"]=2
        result=self.audit([quote()],[game(),second])
        self.assertEqual(result["row_decisions"][0]["reason"],"AMBIGUOUS_OFFICIAL_GAME_WITHIN_TIME_WINDOW")
        self.assertFalse(result["games"])

    def test_schedule_revision_and_same_timestamp_price_conflict_rejected(self):
        revision=game(start="2026-06-15T20:00:00Z")
        result=self.audit([quote()],[game(),revision])
        self.assertEqual(result["row_decisions"][0]["reason"],"OFFICIAL_CAPTURE_GAME_REVISION_AMBIGUITY")
        result=self.audit([quote(),quote(price="1.90")])
        self.assertEqual(result["summary"]["reason_counts"],{"GAME_SOURCE_ID_OR_SAME_TIME_PRICE_CONFLICT":2})
        self.assertFalse(result["games"])

    def test_source_match_id_cannot_map_to_multiple_games(self):
        second=game(pk=102,start="2026-06-15T20:00:00Z")
        result=self.audit([quote(),quote(start=second["gameDate"],stamp="2026-06-15T19:00:00Z")],[game(),second])
        self.assertFalse(result["games"])
        self.assertEqual(result["summary"]["reason_counts"],{"GAME_SOURCE_ID_OR_SAME_TIME_PRICE_CONFLICT":2})

    def test_source_hash_revision_fails_instead_of_silently_repairing(self):
        with tempfile.TemporaryDirectory() as directory:
            path=Path(directory)/"source.jsonl";path.write_text(json.dumps(quote())+"\n")
            schedule=json.dumps({"dates":[{"games":[game()]}]}).encode()
            with self.assertRaisesRegex(ValueError,"source hash mismatch"):
                audit_tsl_archive_eligibility(path,expected_source_sha256="0"*64,schedule_raw=schedule,
                                             expected_schedule_sha256=sha256(schedule).hexdigest())


if __name__ == "__main__":unittest.main()
