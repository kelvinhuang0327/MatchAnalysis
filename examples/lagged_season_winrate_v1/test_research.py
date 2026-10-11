"""No fitting: derivatives, standardization, scoring and refit guards only."""
from hashlib import sha256
from pathlib import Path
import tempfile
import unittest
import research
import train_once


class ResearchTests(unittest.TestCase):
    def test_gradient_and_hessian_finite_difference(self):
        examples = [([1, -1.3, .7], 1), ([1, .9, -1.2], 0), ([1, .4, .5], 1)]
        parameters = [.15, -.2, .3]
        loss, gradient, hessian = research.objective(parameters, examples)
        epsilon = 1e-5
        for i in range(3):
            plus = parameters.copy(); minus = parameters.copy()
            plus[i] += epsilon; minus[i] -= epsilon
            lp, gp, _ = research.objective(plus, examples)
            lm, gm, _ = research.objective(minus, examples)
            self.assertAlmostEqual(gradient[i], (lp - lm) / (2 * epsilon), places=8)
            for j in range(3):
                self.assertAlmostEqual(hessian[j][i], (gp[j] - gm[j]) / (2 * epsilon), places=8)

    def test_l2_penalizes_only_feature_coefficients(self):
        loss, gradient, hessian = research.objective([0, 2, -3], [([1, 0, 0], 0), ([1, 0, 0], 1)])
        self.assertEqual(gradient, [0, 2, -3])
        self.assertEqual(hessian, [[.25, 0, 0], [0, 1, 0], [0, 0, 1]])

    def test_general_two_parameter_objective_matches_sealed_baseline(self):
        examples = [([1, -1], 0), ([1, 2], 1)]
        loss, gradient, hessian = research.objective([.2, -.1], examples)
        old_loss, old_gradient, old_hessian = research.source_gate.annual.loss_grad_hessian((.2, -.1), [(-1, 0), (2, 1)])
        self.assertAlmostEqual(loss, old_loss)
        for a, b in zip(gradient, old_gradient):
            self.assertAlmostEqual(a, b)
        for a, b in zip([hessian[0][0], hessian[0][1], hessian[1][1]], old_hessian):
            self.assertAlmostEqual(a, b)

    def test_train_only_scaler_and_label_independent_prediction(self):
        training = [{'prior_strength_difference': -2, 'current_winrate_difference': -.1, 'y': 0},
                    {'prior_strength_difference': 2, 'current_winrate_difference': .1, 'y': 1}]
        scales = research.scaler(training)
        self.assertEqual(scales[0]['mean'], 0)
        self.assertEqual(scales[0]['scale'], 2)
        target = {'prior_strength_difference': 200, 'current_winrate_difference': .3, 'y': 0}
        artifact = {'standardization': scales, 'models': {'candidate': {'parameters': [.1, .2, .3]}}}
        p, c = research.predict(target, artifact, 'candidate')
        target['y'] = 1
        self.assertEqual(research.predict(target, artifact, 'candidate'), (p, c))
        self.assertEqual(research.scaler(training), scales)

    def test_existing_artifact_refuses_before_source_read_and_preserves_bytes(self):
        with tempfile.TemporaryDirectory(prefix='match-lagged-refit-') as directory:
            root = Path(directory); path = root / 'model_artifact.json'
            path.write_bytes(b'synthetic existing artifact')
            before = sha256(path.read_bytes()).hexdigest()
            with self.assertRaisesRegex(ValueError, 'MODEL_EXISTS_REFIT_REFUSED'):
                train_once.run(root)
            self.assertEqual(before, sha256(path.read_bytes()).hexdigest())


if __name__ == '__main__':
    unittest.main()
