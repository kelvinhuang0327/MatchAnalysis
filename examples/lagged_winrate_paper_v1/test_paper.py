"""Only adapter time gates, missing inputs and deterministic lifecycle replay."""
from datetime import UTC, datetime, timedelta
from hashlib import sha256
from pathlib import Path
from unittest.mock import patch
import copy
import json
import tempfile
import unittest
import paper


class PaperTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(prefix='match-candidate-synthetic-')
        self.root = Path(self.temp.name)
        self.input = self.root / 'inputs'; self.input.mkdir()
        self.output = self.root / 'paper'
        self.time = datetime(2026, 10, 11, 15, tzinfo=UTC)
        self.target = self.game(999999, '2026-10-11', final=False)
        self.target['gameDate'] = '2026-10-11T23:00:00Z'
        self.histories = [self.game(1000 + i, f'2026-07-{i+1:02d}', final=True) for i in range(20)]
        self.source('target_schedule', [self.target], '2026-10-11', '2026-10-11')
        self.source('season_history', self.histories, '2026-01-01', '2026-10-04')
        self.manifest = {'schema_version': 'lagged_winrate.paper_input.v1', 'synthetic': True,
                         'target_game_pk': 999999, 'source_captures': self.records.copy(),
                         'prior_annual': {'year': 2025}}
        for key in ['csv', 'gamelog']:
            raw = ('SYNTHETIC_ANNUAL_' + key).encode(); path = key + '.txt'
            (self.input / path).write_bytes(raw)
            self.manifest['prior_annual'][key] = {'path': path, 'sha256': sha256(raw).hexdigest()}
        self.write_manifest()
        annual_info = {'last_completion_date': '2025-09-28', 'synthetic': True}
        self.annual_patch = patch.object(paper.research.source_gate.annual, 'season',
            return_value=({147: {'strength': .8}, 119: {'strength': -.2}}, [], annual_info))
        self.annual_patch.start()

    def tearDown(self):
        self.annual_patch.stop()
        self.temp.cleanup()

    def game(self, pk, day, final):
        return {'gamePk': pk, 'gameType': 'R', 'gameNumber': 1, 'officialDate': day,
                'gameDate': day + 'T19:00:00Z', 'status': {'abstractGameState': 'Final' if final else 'Preview'},
                'gameInfo': {'firstPitch': day + 'T19:05:00Z', 'gameDurationMinutes': 180} if final else {},
                'teams': {'home': {'team': {'id': 147, 'name': 'New York Yankees', 'abbreviation': 'NYY'}, 'score': 5 if final else 0},
                          'away': {'team': {'id': 119, 'name': 'Los Angeles Dodgers', 'abbreviation': 'LAD'}, 'score': 2 if final else 0}}}

    def source(self, name, games, start, end):
        payload = {'totalGames': len(games), 'dates': [{'date': start, 'games': games}]}
        path = name + '.json'; raw = paper.encoded(payload); (self.input / path).write_bytes(raw)
        if not hasattr(self, 'records'):
            self.records = []
        record = {'name': name, 'path': path, 'url': f'https://statsapi.mlb.com/api/v1/schedule?sportId=1&gameType=R&startDate={start}&endDate={end}&hydrate=gameInfo',
                  'status': 'SUCCESS', 'purpose': 'MODEL_INPUT', 'sha256': sha256(raw).hexdigest(),
                  'received_at_utc': '2026-10-11T14:58:00Z', 'persisted_at_utc': '2026-10-11T14:59:00Z'}
        self.records = [r for r in self.records if r['name'] != name] + [record]
        if hasattr(self, 'manifest'):
            self.manifest['source_captures'] = self.records.copy(); self.write_manifest()

    def write_manifest(self):
        (self.input / 'input_manifest.json').write_bytes(paper.encoded(self.manifest))

    def freeze(self):
        return paper.freeze(self.input, self.output, clock=lambda: self.time)

    def test_saved_probability_replays_and_synthetic_never_forward(self):
        report = self.freeze()
        self.assertEqual(report['run_classification'], paper.REHEARSAL)
        self.assertEqual(report['pair']['status'], 'QUOTE_MISSING')
        self.assertFalse(report['pair']['final_sample_eligible'])
        self.assertEqual(report['new_fits'], 0)
        self.assertEqual(paper.replay(self.output), report)
        run = paper.load_frozen_prediction_run(self.output / report['run_relative'], data_mode='FIXTURE')
        self.assertEqual(run.predictions[0]['model_identity']['model_role'], 'RESEARCH_CANDIDATE')
        self.assertNotIn('home_decimal_odds', run.predictions[0])
        self.assertNotIn('actual_winner', run.predictions[0])
        (self.output / 'season_history.json').write_text('{}')
        with self.assertRaisesRegex(ValueError, 'HASH_MISMATCH'):
            paper.replay(self.output)

    def test_after_start_and_late_source_block_before_output(self):
        self.time = datetime(2026, 10, 11, 23, tzinfo=UTC)
        with self.assertRaisesRegex(ValueError, 'TARGET_NOT_FUTURE'):
            self.freeze()
        self.assertFalse(self.output.exists())
        self.time = datetime(2026, 10, 11, 15, tzinfo=UTC)
        self.manifest['source_captures'][0]['persisted_at_utc'] = '2026-10-11T15:01:00Z'
        self.write_manifest()
        with self.assertRaisesRegex(ValueError, 'SOURCE_NOT_AVAILABLE'):
            self.freeze()
        self.assertFalse(self.output.exists())

    def test_insufficient_or_unknown_completion_history_blocks(self):
        self.source('season_history', self.histories[:19], '2026-01-01', '2026-10-04')
        with self.assertRaisesRegex(ValueError, 'INSUFFICIENT_LAGGED_HISTORY'):
            self.freeze()
        self.assertFalse(self.output.exists())
        unknown = copy.deepcopy(self.histories)
        for g in unknown:
            g['gameInfo'] = {}
        self.source('season_history', unknown, '2026-01-01', '2026-10-04')
        with self.assertRaisesRegex(ValueError, 'INSUFFICIENT_LAGGED_HISTORY'):
            self.freeze()

    def test_seven_day_cutoff_excludes_too_recent_and_resumed(self):
        extra = [self.game(2000, '2026-10-05', True), self.game(2001, '2026-07-30', True)]
        extra[1]['resumedFrom'] = '2026-07-29'
        self.source('season_history', self.histories + extra, '2026-01-01', '2026-10-04')
        report = self.freeze()
        evidence = json.loads((self.output / 'input_evidence.json').read_bytes())
        self.assertEqual(len(evidence['accepted_history']), 20)
        self.assertEqual(evidence['history_reason_counts'], {'HISTORY_AMBIGUOUS_OR_RESUMED': 1, 'HISTORY_NOT_FINISHED_BY_SEVEN_DAY_BOUNDARY': 1})
        self.assertEqual(paper.replay(self.output), report)

    def test_partial_history_and_wrong_prior_year_block(self):
        self.source('season_history', self.histories, '2026-07-01', '2026-10-04')
        with self.assertRaisesRegex(ValueError, 'FULL_CURRENT_SEASON'):
            self.freeze()
        self.source('season_history', self.histories, '2026-01-01', '2026-10-04')
        self.manifest['prior_annual']['year'] = 2024; self.write_manifest()
        with self.assertRaisesRegex(ValueError, 'PREVIOUS_COMPLETE_SEASON'):
            self.freeze()

    def test_only_official_final_settles_and_frozen_replays_unchanged(self):
        report = self.freeze()
        result_root = self.root / 'results'; result_root.mkdir()
        game = copy.deepcopy(self.target); game['status']['abstractGameState'] = 'Live'
        def result_manifest(g):
            raw = paper.encoded({'dates': [{'games': [g]}]})
            (result_root / 'final.json').write_bytes(raw)
            record = {'path': 'final.json', 'url': 'https://statsapi.mlb.com/api/v1/schedule?gamePk=999999',
                      'status': 'SUCCESS', 'received_at_utc': '2026-10-12T02:00:00Z',
                      'persisted_at_utc': '2026-10-12T02:00:01Z', 'sha256': sha256(raw).hexdigest()}
            (result_root / 'input_manifest.json').write_bytes(paper.encoded({'source_captures': [record]}))
        result_manifest(game)
        later = lambda: datetime(2026, 10, 12, 3, tzinfo=UTC)
        self.assertEqual(paper.settle(self.output, result_root, clock=later)['status'], 'PENDING_OFFICIAL_FINAL')
        frozen = self.output / report['run_relative'] / 'frozen_predictions.jsonl'
        before = frozen.read_bytes()
        game['status']['abstractGameState'] = 'Final'; game['teams']['home']['score'] = 5; game['teams']['away']['score'] = 2
        result_manifest(game)
        result = paper.settle(self.output, result_root, clock=later)
        self.assertEqual(result['newly_settled_count'], 1)
        self.assertEqual(frozen.read_bytes(), before)
        self.assertEqual(paper.replay(self.output), report)
        self.assertEqual(paper.settle(self.output, result_root, clock=later)['newly_settled_count'], 0)
        self.assertFalse((self.output / 'ledger/prediction_forward_ledger.jsonl').exists())


if __name__ == '__main__':
    unittest.main()
