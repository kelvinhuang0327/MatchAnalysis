"""Focused synthetic checks: no derived-time substitution, no boundary leakage."""
import unittest
from data_gate import strict_completion_reason,audit_year

def record(**changes):
    row={'first_pitch_utc':'2025-06-01T17:00:00Z','completed_at_utc':'2025-06-01T20:00:00Z',
        'timing_is_derived':False,'completion_source_kind':'OFFICIAL_ACTUAL_FINAL_EVENT',
        'raw_sha256':'1'*64,'final':True,'game_type':'R','suspended_resumed':False}
    row.update(changes);return row

class CompletionGateTests(unittest.TestCase):
    def test_actual_final_before_cutoff_is_usable(self):
        self.assertIsNone(strict_completion_reason(record(),'2025-06-01T21:00:00Z'))

    def test_duration_estimate_and_date_only_cannot_supply_actual_completion(self):
        self.assertEqual(strict_completion_reason(record(timing_is_derived=True,
            completion_source_kind='FIRST_PITCH_PLUS_DURATION'),'2025-06-01T21:00:00Z'),
            'ACTUAL_COMPLETION_EVENT_PROVENANCE_REQUIRED')
        self.assertEqual(strict_completion_reason(record(completed_at_utc='2025-06-01'),
            '2025-06-01T21:00:00Z'),'UTC_INSTANT_REQUIRED')

    def test_completion_equal_to_or_after_cutoff_rejected(self):
        for at in ['2025-06-01T20:00:00Z','2025-06-01T19:00:00Z']:
            self.assertEqual(strict_completion_reason(record(),at),
                'NOT_COMPLETED_STRICTLY_BEFORE_FEATURE_CUTOFF')

    def test_unresolved_resumption_cannot_use_original_clock(self):
        self.assertEqual(strict_completion_reason(record(suspended_resumed=True),
            '2025-06-01T21:00:00Z'),'RESUMPTION_FINAL_COMPLETION_UNRESOLVED')

    def test_final_and_raw_hash_are_required(self):
        self.assertEqual(strict_completion_reason(record(final=False),'2025-06-01T21:00:00Z'),
            'FINAL_REGULAR_GAME_REQUIRED')
        self.assertEqual(strict_completion_reason(record(raw_sha256=''),'2025-06-01T21:00:00Z'),
            'RAW_RESPONSE_HASH_REQUIRED')

    def test_wrong_score_label_and_doubleheader_ambiguity_are_reported(self):
        rows=[{'date':'2025-06-01','away_team':'A','home_team':'B','away_score':'3','home_score':'4','home_win':'0'},
              {'date':'2025-06-01','away_team':'A','home_team':'B','away_score':'2','home_score':'1','home_win':'0'}]
        logs=[]
        for number in ['1','2']:
            gl=['']*161;gl[0]='20250601';gl[1]=number;gl[3]='A';gl[6]='B';gl[18]='180';logs.append(gl)
        result=audit_year(2025,rows,logs)
        self.assertEqual(result['valid_non_tie_score_labels'],1)
        self.assertEqual(result['invalid_or_conflicting_labels'],1)
        self.assertEqual(result['duplicate_csv_date_team_excess_rows'],1)
        self.assertEqual(result['unique_retrosheet_date_number_away_home_keys'],2)
        self.assertEqual(result['strict_verified_completion_rows'],0)

if __name__=='__main__':unittest.main()
