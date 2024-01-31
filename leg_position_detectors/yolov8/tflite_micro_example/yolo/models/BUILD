load("//tensorflow/lite/micro:build_def.bzl", "generate_cc_arrays")

package(
    default_visibility = ["//visibility:public"],
)

exports_files(
    srcs = [
        "yolov8_256_int8.tflite",
        "head_256_float32.tflite"
    ],
    visibility = [
        "//codegen/examples/yolo:__subpackages__",
        "//tensorflow/lite/micro/examples/yolo:__subpackages__",
    ],
)

generate_cc_arrays(
    name = "generated_yolov8_256_int8_model_cc",
    src = "yolov8_256_int8.tflite",
    out = "yolov8_256_int8_model_data.cc",
)

generate_cc_arrays(
    name = "generated_yolov8_256_int8_model_hdr",
    src = "yolov8_256_int8.tflite",
    out = "yolov8_256_int8_model_data.h",
)

generate_cc_arrays(
    name = "generated_head_256_float32_model_cc",
    src = "head_256_float32.tflite",
    out = "head_256_float32_model_data.cc",
)

generate_cc_arrays(
    name = "generated_head_256_float32_model_hdr",
    src = "head_256_float32.tflite",
    out = "head_256_float32_model_data.h",
)
