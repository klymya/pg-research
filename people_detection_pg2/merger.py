import onnx
from onnx.compose import merge_models
from typing import List, Tuple


def get_tensor_shapes(tensors) -> List[Tuple[str, List[int]]]:
    shape_info = []
    for t in tensors:
        shape = []
        for dim in t.type.tensor_type.shape.dim:
            if dim.HasField("dim_value"):
                shape.append(dim.dim_value)
            else:
                shape.append(-1)  # symbolic or unknown dimension
        shape_info.append((t.name, shape))
    return shape_info


def shapes_match(shape1: List[int], shape2: List[int]) -> bool:
    if len(shape1) != len(shape2):
        return False
    return all((a == b or a == -1 or b == -1) for a, b in zip(shape1, shape2))


def auto_map_by_shape(model1: onnx.ModelProto, model2: onnx.ModelProto) -> List[Tuple[str, str]]:
    outputs1 = get_tensor_shapes(model1.graph.output)
    inputs2 = get_tensor_shapes(model2.graph.input)

    mapping = []
    used_outputs = set()

    for in_name, in_shape in inputs2:
        for out_name, out_shape in outputs1:
            if out_name in used_outputs:
                continue
            if shapes_match(out_shape, in_shape):
                mapping.append((out_name, in_name))
                used_outputs.add(out_name)
                break

    return mapping


def merge_yolo(detector_pth: str, head_pth: str, output_pth: str) -> onnx.ModelProto:
    detector = onnx.load(detector_pth)
    head = onnx.load(head_pth)

    # Ensure the models are compatible
    if len(detector.graph.output) != len(head.graph.input):
        raise ValueError(
            "The number of outputs in the detector model does not match the number of inputs in the head model.")

    # Merge the models
    merged = merge_models(
        detector,
        head,
        io_map=auto_map_by_shape(detector, head),
    )

    onnx.save(merged, output_pth)
