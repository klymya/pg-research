import numpy as np

from sort.sort import calc_detection_score, calc_metrics, measure_score

L2_TH = 0.5

pass
# test 1
gts = []
dets = [[0, 0]]

tp, fp, fn, errs = calc_detection_score(gts, dets, L2_TH)
assert tp == 0
assert fp == 1
assert fn == 0
assert errs == []

# test 2
gts = [[0, 0]]
dets = []

tp, fp, fn, errs = calc_detection_score(gts, dets, L2_TH)
assert tp == 0
assert fp == 0
assert fn == 1
assert errs == []

# test 3
gts = [[0, 0]]
dets = [[0, 0.1]]

tp, fp, fn, errs = calc_detection_score(gts, dets, L2_TH)
assert tp == 1
assert fp == 0
assert fn == 0
assert (np.array(errs) == np.array([0.1])).all()

# test 4
gts = [[0, 0]]
dets = [[0, 1]]

tp, fp, fn, errs = calc_detection_score(gts, dets, L2_TH)
assert tp == 0
assert fp == 1
assert fn == 1
assert (np.array(errs) == np.array([1])).all()

# test 5
gts = [[0, 0], [1, 0]]
dets = [[0, 1]]

tp, fp, fn, errs = calc_detection_score(gts, dets, L2_TH)
assert tp == 0
assert fp == 1
assert fn == 2
assert errs == []

# test 6
gts = [[0, 0], [1, 0]]
dets = [[0.1, 0]]

tp, fp, fn, errs = calc_detection_score(gts, dets, L2_TH)
assert tp == 1
assert fp == 0
assert fn == 1
assert (np.array(errs) == np.array([0.1])).all()

# test 7
gts = [[0, 0.1]]
dets = [[0, 0], [0, 1]]

tp, fp, fn, errs = calc_detection_score(gts, dets, L2_TH)
assert tp == 1
assert fp == 1
assert fn == 0
assert (np.array(errs) == np.array([0.1])).all()

# test 8
gts = [[0, 0.1], [1, 0]]
dets = [[0, 0], [0.9, 0]]


tp, fp, fn, errs = calc_detection_score(gts, dets, L2_TH)
assert tp == 2
assert fp == 0
assert fn == 0
assert (np.array(errs) - np.array([0.1, 0.1])).sum() < 1e-3
