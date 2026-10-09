"""Focused math and year-boundary tests; never refits real game data."""
from pathlib import Path
import json
import hashlib
import subprocess
import sys
import unittest
from research import annual_chronology,loss_grad_hessian,standardize,predict,season

class FrozenResearchTests(unittest.TestCase):
    def test_same_or_later_year_end_rejected(self):
        self.assertEqual(annual_chronology('2024-09-30','2025-03-18'),169)
        for completed in ['2025-03-18','2025-03-19']:
            with self.assertRaises(ValueError):annual_chronology(completed,'2025-03-18')

    def test_train_scaler_does_not_use_evaluation_inputs(self):
        mean,scale,_=standardize([{'x':-1.,'y':0},{'x':1.,'y':1}])
        self.assertEqual((mean,scale),(0.,1.))
        model={'standardization':{'mean':mean,'scale':scale},'fit':{'intercept':0.,'beta_strength':1.}}
        self.assertGreater(predict(100.,model),0.99)
        self.assertEqual(model['standardization'],{'mean':0.,'scale':1.})

    def test_l2_gradient_hessian_match_finite_difference(self):
        examples=[(-1.2,0),(-0.1,1),(0.7,0),(1.4,1)];point=[0.2,0.1];eps=1e-5
        _,gradient,hessian=loss_grad_hessian(point,examples)
        for i in range(2):
            high=point.copy();low=point.copy();high[i]+=eps;low[i]-=eps
            lh,gh,_=loss_grad_hessian(high,examples);ll,gl,_=loss_grad_hessian(low,examples)
            self.assertAlmostEqual((lh-ll)/(2*eps),gradient[i],places=8)
            for j in range(2):
                expected=[[hessian[0],hessian[1]],[hessian[1],hessian[2]]][j][i]
                self.assertAlmostEqual((gh[j]-gl[j])/(2*eps),expected,places=8)

    def test_intercept_unpenalized_beta_penalized_in_mean_loss(self):
        examples=[(-1.,0),(1.,1)]
        _,g,h=loss_grad_hessian((0.,0.),examples)
        self.assertEqual(g,(0.,-0.5))
        self.assertEqual(h,(0.25,0.,1.25))

    def test_saved_model_cannot_depend_on_current_game_outcome(self):
        model={'standardization':{'mean':0.1,'scale':1.5},'fit':{'intercept':0.05,'beta_strength':0.2}}
        row={'x':0.8,'y':0};before=predict(row['x'],model)
        row['y']=1
        self.assertEqual(predict(row['x'],model),before)

    def test_existing_real_model_cannot_be_refitted_or_overwritten(self):
        root=Path(__file__).resolve().parent;path=root/'model_artifact.json'
        before=hashlib.sha256(path.read_bytes()).hexdigest()
        blocked=subprocess.run([sys.executable,str(root/'train_once.py')],capture_output=True,text=True)
        self.assertNotEqual(blocked.returncode,0)
        self.assertIn('REAL_FIT_ALREADY_FROZEN_USE_SAVED_MODEL_REPLAY',blocked.stderr)
        self.assertEqual(hashlib.sha256(path.read_bytes()).hexdigest(),before)

if __name__=='__main__':unittest.main()
