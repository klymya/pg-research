# Description:
#   TensorFlow Lite for Microcontrollers "hello world" example.

load(
    "//tensorflow/lite/micro:build_def.bzl",
    "micro_copts",
    "generate_cc_arrays"
)

package(
    # Disabling layering_check because of http://b/177257332
    features = ["-layering_check"],
    licenses = ["notice"],
)

generate_cc_arrays(
    name = "generated_img_256_bmp_cc",
    src = "testdata/img_256.bmp",
    out = "testdata/img_256_image_data.cc",
)

generate_cc_arrays(
    name = "generated_img_256_bmp_hdr",
    src = "testdata/img_256.bmp",
    out = "testdata/img_256_image_data.h",
)

cc_library(
    name = "model",
    srcs = [
        "//tensorflow/lite/micro/examples/yolo/models:generated_yolov8_256_int8_model_cc",
        "//tensorflow/lite/micro/examples/yolo/models:generated_head_256_float32_model_cc",
    ],
    hdrs = [
        "//tensorflow/lite/micro/examples/yolo/models:generated_yolov8_256_int8_model_hdr",
        "//tensorflow/lite/micro/examples/yolo/models:generated_head_256_float32_model_hdr",
    ],
    copts = micro_copts(),
)

cc_library(
    name = "simple_images_test_data",
    srcs = [
        ":generated_img_256_bmp_cc",
    ],
    hdrs = [
        ":generated_img_256_bmp_hdr",
    ]
)

cc_test(
    name = "yolo_test",
    srcs = [
        "yolo_test.cc",
    ],
    deps = [
        ":model",
        ":simple_images_test_data",
        "//tensorflow/lite/micro:micro_framework",
        "//tensorflow/lite/micro:micro_log",
        "//tensorflow/lite/micro:micro_profiler",
        "//tensorflow/lite/micro:op_resolvers",
        "//tensorflow/lite/micro:recording_allocators",
        "//tensorflow/lite/micro/testing:micro_test",
        "//tensorflow/lite/schema:schema_fbs",
    ],
)
