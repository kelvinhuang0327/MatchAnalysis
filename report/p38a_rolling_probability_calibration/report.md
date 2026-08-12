# P38A Rolling Moneyline Probability Calibration Evaluation

This is a leakage-safe offline probability-reliability evaluation. No model was promoted.

## Fixed calibration method

- Method: PLATT_LOGISTIC_RAW_PROBABILITY_LOGIT (p38a.platt_probability_calibrator.v1).
- Authority: P37A_PRIOR_TRUE_OOS_PREDICTION_LABEL_PAIRS; the source rows are prior P37A true-OOS prediction/label pairs.
- Method search and target-holdout tuning: false.

## Chronological windows

| Window | Train folds | Calibration source | Holdout | Raw | Evaluable | Excluded |
| --- | --- | --- | --- | ---: | ---: | ---: |
| window_002_holdout_wf_005 | wf_002,wf_003,wf_004 | wf_004 | wf_005 | 22 | 17 | 5 |
| window_003_holdout_wf_006 | wf_002,wf_003,wf_004,wf_005 | wf_004,wf_005 | wf_006 | 30 | 25 | 5 |

## Per-window probability metrics

| Holdout | Model | Accuracy | Brier | Log loss | ECE |
| --- | --- | ---: | ---: | ---: | ---: |
| wf_005 | champion | 0.6470588235294117647058823529 | 0.2355944398855450715523633044 | 0.66379886189740753957707477 | 0.1201045704687351817285140723 |
| wf_005 | raw_challenger | 0.6470588235294117647058823529 | 0.2363256219673267139829349482 | 0.6650635301896911119371176841 | 0.1238170387933191130805294504 |
| wf_005 | calibrated_challenger | 0.4705882352941176470588235294 | 0.3038423382484724881515290718 | 0.9267522021742132148507003318 | 0.2830848510309472026072250253 |
| wf_006 | champion | 0.56 | 0.2450962914115768163054796276 | 0.683369851705326897156439382 | 0.008050282420814521908686492260 |
| wf_006 | raw_challenger | 0.6 | 0.2441871458422676616926770260 | 0.681643548186464442742110638 | 0.03619578620800734136766159565 |
| wf_006 | calibrated_challenger | 0.6 | 0.2578319227407989933971403955 | 0.7250777635807386469537879248 | 0.1046929796634934152684337242 |

## Aggregate exact-row comparison

- Rows: 42 evaluable of 52 raw; coverage 0.8076923076923076923076923077.

| Model | Accuracy | Brier | Log loss | ECE |
| --- | ---: | ---: | ---: | ---: |
| champion | 0.5952380952380952380952380952 | 0.2412503038891353958101706395 | 0.6754482605925976333743156102 | 0.04382191993924131051589468869 |
| raw_challenger | 0.6190476190476190476190476190 | 0.2410051004643153733339718993 | 0.6749325885211038088448515852 | 0.05335689014427452139307506326 |
| calibrated_challenger | 0.5476190476190476190476190476 | 0.2764551861605715984167739071 | 0.8067078934876212101501572326 | 0.1629662320704954796587465532 |

- Calibrated vs raw deltas (right minus left): {'accuracy_delta': '-0.0714285714285714285714285714', 'brier_delta': '0.0354500856962562250828020078', 'log_loss_delta': '0.1317753049665174013053056474', 'calibration_ece_delta': '0.1096093419262209582656714899'}.
- Calibrated vs champion deltas (right minus left): {'accuracy_delta': '-0.0476190476190476190476190476', 'brier_delta': '0.0352048822714362026066032676', 'log_loss_delta': '0.1312596328950235767758416224', 'calibration_ece_delta': '0.1191443121312541691428518645'}.
- Conclusion: CALIBRATION_NOT_IMPROVED.
- Accuracy uses each model's probability threshold of 0.5; calibration can change accuracy when its fixed Platt map shifts a raw probability across that threshold.

## Safety claims

- Calibration rows strictly precede each target holdout and are game-ID disjoint from it.
- The calibrator is fit only on prior true-OOS P37A rows; target-holdout labels are not used for fitting or method selection.
- Champion, raw challenger, and calibrated challenger use identical evaluable target rows in each window.
- P37A exclusion semantics are preserved; no calibration, betting, edge, profitability, staking, or promotion claim is made.

## Not admitted

- wf_004: No prior true-OOS P37A prediction/label pairs precede wf_004; it is not used as a P38A target because its calibration lineage is unavailable.
