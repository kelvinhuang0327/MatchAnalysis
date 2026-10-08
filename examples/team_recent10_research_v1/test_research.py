"""Focused chronology and no-lookahead checks on isolated synthetic copies."""
from hashlib import sha256
from pathlib import Path
from tempfile import TemporaryDirectory
import json
import shutil
import unittest
from evaluate import evaluate, parse, direction

ROOT=Path(__file__).resolve().parent

class ResearchBoundaryTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.result=evaluate(ROOT)

    def test_ties_abstain_without_probability(self):
        self.assertIsNone(direction(5,5))
        ties=[r for r in self.result['rows'] if r['included'] and r['recent10_direction'] is None]
        self.assertEqual(len(ties),18)
        self.assertIsNone(self.result['probability_mapping'])
        self.assertIsNone(self.result['brier'])
        self.assertIsNone(self.result['log_loss'])

    def test_all_used_histories_completed_before_target_and_unique(self):
        for row in self.result['rows']:
            if not row['included']:continue
            for side,history in row['histories'].items():
                self.assertEqual(len(history['games']),10)
                self.assertEqual(len({g['game_pk'] for g in history['games']}),10)
                for game in history['games']:
                    self.assertNotEqual(game['game_pk'],row['game_pk'])
                    self.assertLess(parse(game['derived_completion_upper_bound_utc']),parse(row['scheduled_start_utc']))

    def test_resumed_game_never_uses_original_start_plus_duration_as_completion(self):
        excluded=[r for r in self.result['rows'] if not r['included']]
        self.assertEqual(len(excluded),7)
        for row in excluded:
            errors=[e for h in row['histories'].values() for e in h['errors']]
            self.assertTrue(any(e['game_pk']==824912 and 'SUSPENDED_RESUMED' in e['reason'] for e in errors))

    def isolated(self,tmp):
        repo=Path(tmp)/'repo'
        capsule=repo/'examples'/ROOT.name
        shutil.copytree(ROOT,capsule,ignore=shutil.ignore_patterns('__pycache__'))
        target=repo/'data/fixtures/p23f2_official_2026_history/raw/schedule_2026-03-25_2026-06-30.json'
        target.parent.mkdir(parents=True)
        shutil.copyfile(ROOT.parent.parent/'data/fixtures/p23f2_official_2026_history/raw'/target.name,target)
        return capsule,target

    def test_missing_latest_completion_invalidates_instead_of_silent_backfill(self):
        base=next(r for r in self.result['rows'] if r['included'])
        game_pk=base['histories']['home']['games'][0]['game_pk']
        with TemporaryDirectory(prefix='recent10-missing-synthetic-') as tmp:
            capsule,_=self.isolated(tmp)
            path=capsule/'completion_source.json';data=json.loads(path.read_bytes())
            for date in data['dates']:
                for game in date['games']:
                    if game['gamePk']==game_pk:game['gameInfo']['firstPitch']=None
            raw=(json.dumps(data,sort_keys=True)+'\n').encode();path.write_bytes(raw)
            captures=json.loads((capsule/'source_captures.json').read_bytes())
            for capture in captures:
                if capture['path']==path.name:capture.update(sha256=sha256(raw).hexdigest(),byte_count=len(raw))
            (capsule/'source_captures.json').write_text(json.dumps(captures))
            altered=next(r for r in evaluate(capsule)['rows'] if r['game_pk']==base['game_pk'])
            self.assertFalse(altered['included'])
            self.assertIsNone(altered['recent10_direction'])
            self.assertTrue(any('HISTORY_COMPLETION_TIME_MISSING' in e['reason'] for e in altered['histories']['home']['errors']))

    def test_target_outcome_cannot_change_its_own_direction(self):
        base=next(r for r in self.result['rows'] if r['included'] and r['recent10_direction'] is not None)
        with TemporaryDirectory(prefix='recent10-outcome-synthetic-') as tmp:
            capsule,schedule=self.isolated(tmp)
            data=json.loads(schedule.read_bytes())
            for date in data['dates']:
                for game in date['games']:
                    if game['gamePk']==base['game_pk']:
                        home=game['teams']['home'];away=game['teams']['away'];home['score'],away['score']=away['score'],home['score']
            raw=(json.dumps(data,sort_keys=True)+'\n').encode();schedule.write_bytes(raw)
            protocol=json.loads((capsule/'protocol.json').read_bytes())
            protocol['source_hashes']['raw_schedule']=sha256(raw).hexdigest()
            (capsule/'protocol.json').write_text(json.dumps(protocol))
            outcomes=[json.loads(s) for s in (capsule/'opened_outcomes.jsonl').read_text().splitlines()]
            for outcome in outcomes:
                if int(outcome['provider_game_id'])==base['game_pk']:outcome['actual_home_win']=1-outcome['actual_home_win']
            (capsule/'opened_outcomes.jsonl').write_text(''.join(json.dumps(g)+'\n' for g in outcomes))
            changed=next(r for r in evaluate(capsule)['rows'] if r['game_pk']==base['game_pk'])
            self.assertTrue(changed['included'])
            self.assertNotEqual(changed['actual_winner'],base['actual_winner'])
            self.assertEqual(changed['recent10_direction'],base['recent10_direction'])
            self.assertEqual(changed['histories'],base['histories'])

    def test_known_uncompleted_history_does_not_enter_window(self):
        base=next(r for r in self.result['rows'] if r['included'])
        game_pk=base['histories']['home']['games'][0]['game_pk']
        with TemporaryDirectory(prefix='recent10-future-synthetic-') as tmp:
            capsule,_=self.isolated(tmp)
            path=capsule/'completion_source.json';data=json.loads(path.read_bytes())
            for date in data['dates']:
                for game in date['games']:
                    if game['gamePk']==game_pk:game['gameInfo']['firstPitch']=base['scheduled_start_utc']
            raw=(json.dumps(data,sort_keys=True)+'\n').encode();path.write_bytes(raw)
            captures=json.loads((capsule/'source_captures.json').read_bytes())
            for capture in captures:
                if capture['path']==path.name:capture.update(sha256=sha256(raw).hexdigest(),byte_count=len(raw))
            (capsule/'source_captures.json').write_text(json.dumps(captures))
            changed=next(r for r in evaluate(capsule)['rows'] if r['game_pk']==base['game_pk'])
            used={g['game_pk'] for g in changed['histories']['home']['games']}
            self.assertNotIn(game_pk,used)
            self.assertIn(game_pk,{g['game_pk'] for g in changed['histories']['home']['known_uncompleted_before_cutoff']})

if __name__=='__main__':unittest.main()
