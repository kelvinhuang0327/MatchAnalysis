"""Synthetic proof of availability, actual decisions and immutable bounded capture."""
from datetime import UTC, datetime, timedelta
from types import SimpleNamespace
from unittest.mock import patch
import json
from pathlib import Path
import tempfile
import unittest
from urllib.request import Request

from test_p50c_market_comparison import _prediction, _quote
from test_p52a_daily_moneyline_prediction_freeze import _build_mock_schedule_payload, MockMLBOpener
from match_analysis.application.use_cases.prospective_pairing_capture import (
    SourceCaptures, assess_pair, collect_once, replay_capture,
)
from match_analysis.application.use_cases.p50c_market_comparison import MarketComparisonPolicy

POLICY = MarketComparisonPolicy(min_net_ev='0.02', max_quote_age_seconds=120, paper_cost_per_unit='0.02')
REPO = Path(__file__).resolve().parents[2]


def evidence():
    return [{"url": "https://statsapi.mlb.com/api/v1/schedule", "purpose": "MODEL_INPUT",
             "status": "SUCCESS", "received_at_utc": "2026-10-01T15:59:00Z",
             "persisted_at_utc": "2026-10-01T15:59:01Z"},
            {"url": "https://blob3rd.sportslottery.com.tw/apidata/Pre/Games.zh.json",
             "purpose": "QUOTE_CAPTURE", "status": "SUCCESS",
             "received_at_utc": "2026-10-01T19:29:58Z", "persisted_at_utc": "2026-10-01T19:29:59Z"}]


class PairEligibilityTests(unittest.TestCase):
    def assess(self, *, prediction=None, quote=None, sources=None, decision='2026-10-01T19:30:01Z', persisted='2026-10-01T16:00:01Z'):
        return assess_pair(prediction or _prediction(1), quote,
            source_records=evidence() if sources is None else sources,
            freeze_persisted_at_utc=persisted, decision_at_utc=decision, policy=POLICY)

    def test_valid_pair_is_pending_not_final_and_no_vig(self):
        row = self.assess(quote=_quote(_prediction(1)))
        self.assertTrue(row['paired_pregame_eligible'])
        self.assertFalse(row['final_sample_eligible'])
        self.assertEqual(row['paper_selection'], 'BET')
        self.assertAlmostEqual(float(row['market_home_probability']), 2 / 3.9)
        self.assertIsNone(row['clv'])

    def test_actual_decision_expiration_is_not_inferred_max(self):
        row = self.assess(quote=_quote(_prediction(1)), decision='2026-10-01T19:33:00Z')
        self.assertEqual(row['status'], 'QUOTE_STALE_BEFORE_START')

    def test_missing_wrong_identity_and_two_sided_price(self):
        self.assertEqual(self.assess()['status'], 'QUOTE_MISSING')
        self.assertEqual(self.assess(quote=_quote(_prediction(2)))['status'], 'QUOTE_IDENTITY_MISMATCH')
        self.assertEqual(self.assess(quote=_quote(_prediction(1), away_odds=None))['status'], 'TWO_SIDED_PRICE_MISSING')

    def test_post_start_and_future_quote_are_excluded(self):
        row = self.assess(quote=_quote(_prediction(1)), decision='2026-10-01T20:00:00Z')
        self.assertEqual(row['status'], 'FREEZE_OR_DECISION_NOT_PREGAME')
        row = self.assess(quote=_quote(_prediction(1), observed_at='2026-10-01T19:31:00Z'))
        self.assertEqual(row['status'], 'QUOTE_NOT_AVAILABLE_AT_DECISION')

    def test_model_capture_after_cutoff_is_not_backdated(self):
        sources = evidence()
        sources[0]['persisted_at_utc'] = '2026-10-01T16:01:00Z'
        self.assertEqual(self.assess(quote=_quote(_prediction(1)), sources=sources)['status'],
                         'MODEL_SOURCE_NOT_AVAILABLE_AT_FREEZE')

    def test_late_persist_and_missing_quote_capture(self):
        self.assertEqual(self.assess(quote=_quote(_prediction(1)), persisted='2026-10-01T20:00:00Z',
                                    decision='2026-10-01T20:00:01Z')['status'], 'FREEZE_OR_DECISION_NOT_PREGAME')
        self.assertEqual(self.assess(quote=_quote(_prediction(1)), sources=evidence()[:1])['status'],
                         'QUOTE_CAPTURE_EVIDENCE_MISSING')

    def test_low_edge_is_valid_pair_with_no_bet(self):
        row = self.assess(quote=_quote(_prediction(1), home_odds='1.4', away_odds='2.9'))
        self.assertTrue(row['paired_pregame_eligible'])
        self.assertEqual(row['paper_selection'], 'NO_BET')
        self.assertEqual(row['paper_stake_units'], '0')


class CaptureTests(unittest.TestCase):
    def test_capture_receipt_persist_hash_and_no_arbitrary_cache(self):
        with tempfile.TemporaryDirectory() as temp:
            times = iter([datetime(2026,10,8,7,0,n,tzinfo=UTC) for n in range(3)])
            capture = SourceCaptures(Path(temp), clock=lambda: next(times), transport=lambda req, timeout: b'{}')
            request = Request('https://statsapi.mlb.com/api/v1/schedule')
            capture.fetch(request)
            self.assertEqual(capture.records[0]['received_at_utc'], '2026-10-08T07:00:01Z')
            self.assertEqual(capture.records[0]['persisted_at_utc'], '2026-10-08T07:00:02Z')
            self.assertIsNone(capture.records[0]['provider_published_at_utc'])
            with self.assertRaisesRegex(ValueError, 'DUPLICATE_SOURCE_FETCH'):
                capture.fetch(request)
            (Path(temp)/capture.records[0]['path']).write_text('[]')
            with self.assertRaisesRegex(ValueError, 'HASH_MISMATCH'):
                capture.replay(request)
            with self.assertRaisesRegex(ValueError, 'NOT_ALLOWLISTED'):
                capture.fetch(Request('https://example.com/paid'))

    def test_missing_prices_still_preserve_genuine_model_inputs_and_offline_replay(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)/'capture'
            mock = MockMLBOpener(_build_mock_schedule_payload())
            def transport(request, timeout):
                if 'statsapi.mlb.com' in request.full_url:
                    return mock(request, timeout)
                if request.full_url.endswith('Sports.zh.json'):
                    return b'[{"abb":"BSB","id":"1"}]'
                return b'[]'
            ticks = iter(range(120))
            def clock():
                return datetime(2026,8,18,13,0,next(ticks),tzinfo=UTC)
            result = collect_once(repository_root=REPO, output_root=root, target_date='2026-08-18',
                                  game_type='R', game_pk=999001, clock=clock, transport=transport)
            self.assertEqual(result['status'], 'CAPTURE_COMPLETED_PENDING_FINAL', result)
            self.assertEqual(result['freeze_result']['eligible_predictions_count'], 1)
            intake = json.loads(Path(result['freeze_result']['pregame_input_path']).read_bytes())
            self.assertEqual(intake['source_manifest']['fetched_at_utc_semantics'], 'INFERENCE_CUTOFF_NOT_RAW_RECEIPT')
            self.assertEqual(result['pair_rows'][0]['status'], 'QUOTE_MISSING')
            self.assertEqual(result['settled_same_sample_count'], 0)
            self.assertIsNone(result['paper_roi'])
            self.assertEqual(replay_capture(root)['pair_rows'], result['pair_rows'])
            raw = root/result['source_captures'][0]['path']
            raw.write_text('{}')
            with self.assertRaisesRegex(ValueError, 'HASH_MISMATCH'):
                replay_capture(root)

    def test_one_game_price_and_champion_pair_replays_without_final_scoring(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)/'capture'
            mock = MockMLBOpener(_build_mock_schedule_payload())
            game = {"id": "synthetic-tsl-1", "an": "洛杉磯道奇", "hn": "紐約洋基",
                    "kt": "2026-08-18T23:05:00Z", "san": "BSB", "ms": [{"name": "不讓分", "cs": [
                        {"name": "洛杉磯道奇", "pd": "1", "pu": "1"},
                        {"name": "紐約洋基", "pd": "10", "pu": "9"}]}]}
            def transport(request, timeout):
                if 'statsapi.mlb.com' in request.full_url:
                    return mock(request, timeout)
                if request.full_url.endswith('Sports.zh.json'):
                    return b'[{"abb":"BSB","id":"1"}]'
                if '/Pre/' in request.full_url:
                    return json.dumps([game], ensure_ascii=False).encode()
                return b'[]'
            ticks = iter(range(120))
            clock = lambda: datetime(2026,8,18,13,tzinfo=UTC) + timedelta(seconds=next(ticks))
            result = collect_once(repository_root=REPO, output_root=root, target_date='2026-08-18',
                                  game_type='R', game_pk=999001, clock=clock, transport=transport)
            self.assertEqual(result['paired_pending_count'], 1, result)
            self.assertEqual(result['quote_crosswalk_present_count'], 1)
            self.assertEqual(result['pair_rows'][0]['status'], 'PAIRED_PENDING_FINAL')
            self.assertEqual(replay_capture(root)['pair_rows'], result['pair_rows'])
            self.assertEqual(result['settled_same_sample_count'], 0)

    def test_unequally_near_official_games_are_still_ambiguous(self):
        from match_analysis.application.use_cases.p32a_frozen_moneyline_quotes import quotes_from_acquisition
        prediction = _prediction(1)
        identity = prediction['game_identity']
        official = {"game_pk": identity['game_pk'], "provider_game_id": identity['provider_game_id'],
            "game_type": "R", "game_number": 1, "official_date": identity['official_date'],
            "scheduled_start_utc": identity['scheduled_start_utc'],
            "home_team": {"name": identity['home_team'], "abbreviation": identity['home_team_code']},
            "away_team": {"name": identity['away_team'], "abbreviation": identity['away_team_code']}}
        second = {**official, "game_pk": 999, "provider_game_id": "999",
                  "scheduled_start_utc": "2026-10-01T20:00:30Z"}
        observation = SimpleNamespace(official=official, canonical_home_code=identity['home_team_code'],
            canonical_away_code=identity['away_team_code'], row={"game_time": identity['scheduled_start_utc']})
        acquisition = SimpleNamespace(history=SimpleNamespace(rows=[]), target_schedule_rows=[official,second],
                                      target_date='2026-10-01')
        run = SimpleNamespace(predictions=[prediction], game_type='R')
        with patch('match_analysis.application.use_cases.p32a_frozen_moneyline_quotes._crosswalk', return_value=[observation]):
            quotes, issues = quotes_from_acquisition(acquisition, run, reject_ambiguous=True)
        self.assertEqual(quotes, ())
        self.assertEqual(issues[identity['game_pk']], 'QUOTE_OFFICIAL_CROSSWALK_AMBIGUOUS')

    def test_output_history_and_live_date_are_bounded(self):
        with self.assertRaisesRegex(ValueError, 'ISOLATED_OUTPUT'):
            collect_once(repository_root=REPO, output_root=REPO, target_date='2026-10-08', game_type='R', game_pk=1)
        with self.assertRaisesRegex(ValueError, 'HISTORY_WINDOW_LIMIT'):
            collect_once(repository_root=REPO, output_root=Path('/tmp/unused-capture'), target_date='2026-10-08',
                         game_type='R', game_pk=1, history_days=100)
        with self.assertRaisesRegex(ValueError, 'LIVE_TARGET_DATE_OUT_OF_RANGE'):
            collect_once(repository_root=REPO, output_root=Path('/tmp/unused-capture'), target_date='2020-01-01',
                         game_type='R', game_pk=1)
