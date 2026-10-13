"""Focused identity/completion/lag tests only; no fits or external data writes."""
from copy import deepcopy
from datetime import UTC, datetime
import unittest
import evaluate as e


def game():
    return {'gamePk': 1, 'gameType': 'R', 'gameNumber': 2, 'officialDate': '2026-04-03',
            'gameDate': '2026-04-03T18:10:00Z', 'status': {'abstractGameState': 'Final'},
            'teams': {'home': {'team': {'id': 147}, 'score': 5}, 'away': {'team': {'id': 119}, 'score': 2}},
            'gameInfo': {'firstPitch': '2026-04-03T18:10:00Z', 'gameDurationMinutes': 177}}


class ReplayTests(unittest.TestCase):
    def test_explicit_reschedule_keeps_actual_identity_and_counts_once(self):
        actual = game(); actual['rescheduledFrom'] = '2026-04-02T20:10:00Z'
        earlier = deepcopy(actual); earlier['gameDate'] = actual['rescheduledFrom']; earlier['gameNumber'] = 1
        earlier.pop('rescheduledFrom')
        selected, reason = e.canonical_game([earlier, actual])
        self.assertIsNone(reason)
        self.assertEqual(selected, actual)
        self.assertEqual(selected['gameNumber'], 2)
        earlier['teams']['home']['score'] = None
        earlier['teams']['away']['score'] = None
        self.assertEqual(e.canonical_game([earlier, actual]), (actual, None))
        earlier['teams']['home']['score'] = 6
        self.assertEqual(e.canonical_game([earlier, actual])[1], 'RESCHEDULE_SCORE_TEAM_OR_COMPLETION_CONFLICT')

    def test_resumption_and_unlinked_identity_conflict_are_rejected(self):
        original = game(); resumed = deepcopy(original); resumed['resumeDate'] = '2026-04-04'
        self.assertEqual(e.canonical_game([original, resumed])[1], 'RESUMPTION_COMPLETION_UNSUPPORTED')
        other = deepcopy(original); other['gameDate'] = '2026-04-04T18:10:00Z'
        self.assertEqual(e.canonical_game([original, other])[1], 'PHYSICAL_GAME_IDENTITY_AMBIGUOUS')

    def test_final_without_completion_and_future_completion_are_blocked(self):
        g = game(); received = datetime(2026, 10, 13, tzinfo=UTC)
        end, reason = e.completed(g, received)
        self.assertIsNone(reason)
        self.assertEqual(end, datetime(2026, 4, 3, 21, 8, tzinfo=UTC))
        g['gameInfo'].pop('firstPitch')
        self.assertEqual(e.completed(g, received)[1], 'COMPLETION_FIRST_PITCH_OR_DURATION_MISSING')
        self.assertEqual(e.completed(game(), datetime(2026, 4, 3, 20, tzinfo=UTC))[1], 'COMPLETION_TIME_INCONSISTENT_OR_NOT_YET_COMPLETE')

    def test_seven_day_history_excludes_self_future_and_too_recent(self):
        target = {'official_date': '2026-05-20', 'home_id': 1, 'away_id': 2}
        history = [{'key': str(i), 'completion_date': '2026-05-13', 'home_id': 1, 'away_id': 2, 'y': 1} for i in range(20)]
        history += [{'key': 'self', 'completion_date': '2026-05-20', 'home_id': 1, 'away_id': 2, 'y': 0},
                    {'key': 'future', 'completion_date': '2026-05-21', 'home_id': 1, 'away_id': 2, 'y': 0},
                    {'key': 'recent', 'completion_date': '2026-05-14', 'home_id': 1, 'away_id': 2, 'y': 0}]
        delta, totals, errors = e.paper.research.source_gate.lagged_history(target, history)
        self.assertEqual((delta, errors), (1.0, []))
        self.assertEqual(totals[1]['games'], 20)
        self.assertNotIn('self', totals[1]['keys'])

    def test_exact_mcnemar_matches_saved_39_discordance_case(self):
        rows = [{'y': 1, 'candidate_p_home': .6, 'baseline_p_home': .4} for _ in range(21)]
        rows += [{'y': 1, 'candidate_p_home': .4, 'baseline_p_home': .6} for _ in range(18)]
        result = e.exact_paired(rows, 'candidate_p_home', 'baseline_p_home')
        self.assertAlmostEqual(result['exact_mcnemar_two_sided_p'], .7492586247608415)
        self.assertEqual(result['discordant_pairs'], 39)


if __name__ == '__main__':
    unittest.main()
