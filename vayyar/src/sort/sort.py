"""
    SORT: A Simple, Online and Realtime Tracker
    Copyright (C) 2016-2020 Alex Bewley alex@bewley.ai

    This program is free software: you can redistribute it and/or modify
    it under the terms of the GNU General Public License as published by
    the Free Software Foundation, either version 3 of the License, or
    (at your option) any later version.

    This program is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
    GNU General Public License for more details.

    You should have received a copy of the GNU General Public License
    along with this program.  If not, see <http://www.gnu.org/licenses/>.
"""
from __future__ import print_function

import numpy as np
from scipy.linalg import block_diag
from sklearn.metrics import pairwise_distances
from filterpy.kalman import KalmanFilter
from filterpy.common import Q_discrete_white_noise

np.random.seed(0)


def linear_assignment(cost_matrix):
    try:
        import lap
        _, x, y = lap.lapjv(cost_matrix, extend_cost=True)
        return np.array([[y[i], i] for i in x if i >= 0])
    except ImportError:
        from scipy.optimize import linear_sum_assignment
        x, y = linear_sum_assignment(cost_matrix)
        return np.array(list(zip(x, y)))


def l2_batch(bb_test, bb_gt):
    return pairwise_distances(bb_test, bb_gt)


def convert_point_to_z(point):
    return np.array(point).reshape((2, 1))


def convert_point_to_x(point):
    x = np.zeros((4, 1))
    x[0, 0] = point[0]
    x[2, 0] = point[1]
    return x


def convert_x_to_point(x):
    """
    Takes a bounding box in the centre form [x,y,s,r] and returns it in the form
    [x1,y1,x2,y2] where x1,y1 is the top left and x2,y2 is the bottom right
    """
    return np.array([x[0], x[2]]).reshape((1, 2))


class KalmanPointTracker(object):
    """
    This class represents the internal state of individual tracked objects observed as bbox.
    """
    count = 0

    def __init__(self, point, t):
        """
        Initialises a tracker using initial bounding box.
        """
        # define constant velocity model
        self.kf = KalmanFilter(dim_x=4, dim_z=2)
        self.kf.F = np.array([
            [1, 1, 0, 0],
            [0, 1, 0, 0],
            [0, 0, 1, 1],
            [0, 0, 0, 1]
        ])
        self.kf.H = np.array([
            [1, 0, 0, 0],
            [0, 0, 1, 0]
        ])

        R_std = 0.5
        Q_std = 0.05

        self.kf.P *= 10.

        self.kf.R = np.eye(2) * R_std**2
        q = Q_discrete_white_noise(dim=2, dt=1, var=Q_std**2)
        self.kf.Q = block_diag(q, q)

        self.kf.x = convert_point_to_x(point)
        self.time_since_update = 0
        self.id = KalmanPointTracker.count
        KalmanPointTracker.count += 1
        self.history = []
        self.hits = 0
        self.hit_streak = 0
        self.age = 0

        self.raw = [point]
        self.ts = [t]

    def update(self, point, t):
        """
        Updates the state vector with observed bbox.
        """
        self.time_since_update = 0
        self.history = []
        self.hits += 1
        self.hit_streak += 1
        self.kf.update(convert_point_to_z(point))

        self.raw.append(point)
        self.ts.append(t)

    def predict(self):
        """
        Advances the state vector and returns the predicted bounding box estimate.
        """
        # if ((self.kf.x[6]+self.kf.x[2]) <= 0):
        #     self.kf.x[6] *= 0.0
        self.kf.predict()
        self.age += 1
        if (self.time_since_update > 0):
            self.hit_streak = 0
        self.time_since_update += 1
        self.history.append(convert_x_to_point(self.kf.x))
        return self.history[-1]

    def get_state(self):
        """
        Returns the current bounding box estimate.
        """
        return convert_x_to_point(self.kf.x)


def associate_detections_to_trackers(detections, trackers, l2_threshold=0.3):
    """
    Assigns detections to tracked object (both represented as bounding boxes)

    Returns 3 lists of matches, unmatched_detections and unmatched_trackers
    """
    if (len(trackers) == 0):
        return np.empty((0, 2), dtype=int), np.arange(len(detections)), np.empty((0, 2), dtype=int), []
    if (len(detections) == 0):
        return np.empty((0, 2), dtype=int), np.empty((0, 2), dtype=int), np.arange(len(trackers)), []

    l2_matrix = l2_batch(detections, trackers)

    if min(l2_matrix.shape) > 0:
        a = (l2_matrix < l2_threshold).astype(np.int32)
        if a.sum(1).min() == 0 and a.sum(0).min() == 0:
            matched_indices = np.stack(np.where(a), axis=1)
        else:
            matched_indices = linear_assignment(l2_matrix)
    else:
        matched_indices = np.empty(shape=(0, 2))

    unmatched_detections = []
    for d, det in enumerate(detections):
        if (d not in matched_indices[:, 0]):
            unmatched_detections.append(d)
    unmatched_trackers = []
    for t, trk in enumerate(trackers):
        if (t not in matched_indices[:, 1]):
            unmatched_trackers.append(t)

    # filter out matched with hight L2
    matches = []
    for m in matched_indices:
        if (l2_matrix[m[0], m[1]] > l2_threshold):
            unmatched_detections.append(m[0])
            unmatched_trackers.append(m[1])
        else:
            matches.append(m.reshape(1, 2))
    if (len(matches) == 0):
        matches = np.empty((0, 2), dtype=int)
    else:
        matches = np.concatenate(matches, axis=0)

    return matches, np.array(unmatched_detections), np.array(unmatched_trackers), l2_matrix


class Sort(object):
    def __init__(self, max_age=1, min_hits=3, l2_threshold=0.3, return_only_hits=True):
        """
        Sets key parameters for SORT
        """
        self.max_age = max_age
        self.min_hits = min_hits
        self.l2_threshold = l2_threshold
        self.trackers = []
        self.frame_count = 0
        self.dead_trackers = []

        self.return_only_hits = return_only_hits

    def update(self, dets=np.empty((0, 2))):
        """
        Params:
        dets - a numpy array of detections in the format [[x1,y1,x2,y2,score],[x1,y1,x2,y2,score],...]
        Requires: this method must be called once for each frame even with empty detections (use np.empty((0, 5))
            for frames without detections).
        Returns the a similar array, where the last column is the object ID.

        NOTE: The number of objects returned may differ from the number of detections provided.
        """
        self.frame_count += 1
        # get predicted locations from existing trackers.
        trks = np.zeros((len(self.trackers), 2))
        to_del = []
        ret = []
        for t, trk in enumerate(trks):
            pos = self.trackers[t].predict()[0]
            trk[:] = [pos[0], pos[1]]
            if np.any(np.isnan(pos)):
                to_del.append(t)
        trks = np.ma.compress_rows(np.ma.masked_invalid(trks))
        for t in reversed(to_del):
            self.trackers.pop(t)
        matched, unmatched_dets, unmatched_trks, _ = associate_detections_to_trackers(dets, trks, self.l2_threshold)

        # update matched trackers with assigned detections
        for m in matched:
            self.trackers[m[1]].update(dets[m[0], :], self.frame_count)

        # create and initialise new trackers for unmatched detections
        for i in unmatched_dets:
            trk = KalmanPointTracker(dets[i, :], self.frame_count)
            self.trackers.append(trk)
        i = len(self.trackers)
        for trk in reversed(self.trackers):
            d = trk.get_state()[0]
            if ((trk.time_since_update < 1) and
                    (trk.hit_streak >= self.min_hits or self.frame_count <= self.min_hits)) or \
                    not self.return_only_hits:
                ret.append(np.concatenate((d, [trk.id+1])).reshape(1, -1))  # +1 as MOT benchmark requires positive
            i -= 1
            # remove dead tracklet
            if (trk.time_since_update > self.max_age):
                self.trackers.pop(i)
                self.dead_trackers.append(trk)

        if (len(ret) > 0):
            return np.concatenate(ret)
        return np.empty((0, 5))


def calc_detection_score(gt_obj, det_obj, l2_th):
    matched, unmatched_dets, unmatched_gts, l2_mtx = associate_detections_to_trackers(det_obj, gt_obj, l2_th)

    tp = len(matched)
    fp = len(unmatched_dets)
    fn = len(unmatched_gts)

    errs = []
    for idxs in matched:
        errs.append(l2_mtx[idxs[0], idxs[1]])

    return tp, fp, fn, errs


def calc_metrics(tp, fp, fn, beta=1):
    rec = np.divide(tp, tp + fn)
    prec = np.divide(tp, tp + fp)
    f_beta_score = (1 + beta**2) * prec * rec / (beta**2 * prec + rec + 1e-6)

    return rec, prec, f_beta_score


def measure_score(gts, dets, l2_th):
    tp = 0
    fp = 0
    fn = 0
    l2_errs = []

    for gt, det in zip(gts, gts):
        tmp_tp, tmp_fp, tmp_fn, tmp_errs = calc_detection_score(gt, det, l2_th)
        tp += tmp_tp
        fp += tmp_fp
        fn += tmp_fn
        l2_errs.append(tmp_errs)

    rec, prec, f_beta_score = calc_metrics(tp, fp, fn)
    return rec, prec, f_beta_score, np.mean(l2_errs), np.std(l2_errs), np.median(l2_errs)
