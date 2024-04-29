import argparse
import time

import numpy as np

import models


def test_model(model, img, n=100):
    ts = []
    for _ in range(n):
        t0 = time.time()
        model.infer(img)
        t1 = time.time()
        ts.append(t1 - t0)

    print(f"mean {np.mean(ts):.03f}. std {np.std(ts):.03f}")


def main(args):
    det_input = np.random.randint(0, 255, size=(640, 480, 3), dtype=np.uint8)
    seg_input = np.random.randint(0, 255, size=(128, 96, 3), dtype=np.uint8)

    detector = models.Detector(args.d)
    segmentor = models.Segmentator(args.s)

    #warm-up
    detector.infer(det_input)
    segmentor.infer(seg_input)

    print("YOLO:")
    test_model(detector, det_input)

    print("PAN:")
    test_model(segmentor, seg_input)


parser = argparse.ArgumentParser()
parser.add_argument("-d")
parser.add_argument("-s")
args = parser.parse_args()

main(args)
