from metricsevaluator import MetricsEvaluator

# BBOX EXAMPLE
# init evaluator
evaluator = MetricsEvaluator(
    is_relative_coordinates=False,
    is_xywh_coordinates=True,
)

# add gt bbox
evaluator.add_gt_box(
    img_path="img0",
    class_name="person",
    bbox=[100, 100, 100, 100]
)

# add predicted bbox
evaluator.add_det_box(
    img_path="img0",
    class_name="person",
    bbox=[90, 90, 90, 110],
    confidence=0.9
)

# evaluate
result = evaluator.eval_conf_vals(0.5)

print("\n bboxes result")
print(result)

# POINTS EXAMPLE
# by converting the points to bounfing boxes and using it with small IoU value, e.g. 0.01 we emulate L1 distance.
def point2bbox(point: list, width: int) -> list:
    return [point[0] - width // 2, point[1] - width // 2, width, width]

WIDTH = 100

# init evaluator
evaluator = MetricsEvaluator(
    is_relative_coordinates=False,
    is_xywh_coordinates=True,
)

# add gt points
evaluator.add_gt_box(
    img_path="img0",
    class_name="person",
    bbox=point2bbox([100, 100], WIDTH)
)

evaluator.add_gt_box(
    img_path="img1",
    class_name="person",
    bbox=point2bbox([100, 100], WIDTH)
)

# add predicted points
evaluator.add_det_box(
    img_path="img0",
    class_name="person",
    bbox=point2bbox([100, 190], WIDTH),
    confidence=1
)

# add predicted points
evaluator.add_det_box(
    img_path="img1",
    class_name="person",
    bbox=point2bbox([100, 200], WIDTH),
    confidence=1
)

# evaluate
result = evaluator.eval_conf_vals(0.01)

print("\n points result")
print(result)