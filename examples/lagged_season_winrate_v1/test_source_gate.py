"""Focused mathematical/time gate tests use synthetic isolated dictionaries only."""
from datetime import date, timedelta
import unittest
from source_gate import event_reasons, lagged_history, split_date


class GateTests(unittest.TestCase):
    def test_whole_boundary_day_and_label_independent_split(self):
        games = [{'official_date': d, 'y': i % 2} for i, d in enumerate(
            ['2025-01-01', '2025-01-02', '2025-01-02', '2025-01-02', '2025-01-03'])]
        self.assertEqual(split_date(games), ('2025-01-02', 4, 1))
        for g in games:
            g['y'] = 1 - g['y']
        self.assertEqual(split_date(games), ('2025-01-02', 4, 1))

    def test_blank_completion_does_not_imply_never_postponed(self):
        g = {'complete_game_source': True, 'independent_final': True, 'later_completion': False}
        self.assertEqual(event_reasons(g, None), ['POSTPONEMENT_STATUS_UNKNOWN'])
        self.assertEqual(event_reasons(g, True), ['POSTPONED_OR_RESCHEDULED'])
        self.assertEqual(event_reasons(g, False), [])

    def test_suspension_or_incomplete_evidence_never_admitted(self):
        self.assertIn('LATER_DATE_COMPLETION', event_reasons(
            {'complete_game_source': True, 'independent_final': True, 'later_completion': True}, False))
        self.assertIn('COMPLETION_SOURCE_UNKNOWN', event_reasons({}, False))

    def test_seven_day_boundary_and_future_targets_do_not_leak(self):
        target = {'official_date': '2025-05-20', 'home_id': 1, 'away_id': 2, 'y': 1}
        history = []
        for i in range(20):
            d = (date(2025, 5, 13) - timedelta(days=i)).isoformat()
            history.append({'key': str(i), 'completion_date': d, 'home_id': 1, 'away_id': 2, 'y': 1})
        history += [{'key': 'too_recent', 'completion_date': '2025-05-14', 'home_id': 1, 'away_id': 2, 'y': 0},
                    {'key': 'future', 'completion_date': '2025-05-21', 'home_id': 1, 'away_id': 2, 'y': 0}]
        x, counts, reasons = lagged_history(target, history)
        self.assertEqual((x, reasons), (1.0, []))
        self.assertEqual(counts[1]['games'], 20)
        self.assertNotIn('too_recent', counts[1]['keys'])
        target['y'] = 0
        self.assertEqual(lagged_history(target, history)[0], x)

    def test_minimum_excludes_both_models_and_never_imputes(self):
        x, counts, reasons = lagged_history({'official_date': '2025-05-20', 'home_id': 1, 'away_id': 2}, [])
        self.assertIsNone(x)
        self.assertEqual(set(reasons), {'HOME_HISTORY_BELOW_MINIMUM', 'AWAY_HISTORY_BELOW_MINIMUM'})


if __name__ == '__main__':
    unittest.main()
