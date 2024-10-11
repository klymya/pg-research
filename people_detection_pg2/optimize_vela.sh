# RTSS-HE
vela results/filtered_v3_margin_15/weights/best_saved_model/best_full_integer_quant.tflite \
    --output-dir results/filtered_v3_margin_15/weights/best_saved_model/vela_he \
    --accelerator-config ethos-u55-128 \
    --optimise Performance \
    --config vela.ini \
    --system-config Ethos_U55_High_End_Embedded \
    --memory-mode Shared_Sram

vela results/filtered_v3_margin_15/weights/head_saved_model/head_float32.tflite \
    --output-dir results/filtered_v3_margin_15/weights/head_saved_model/vela_he \
    --accelerator-config ethos-u55-128 \
    --optimise Performance \
    --config vela.ini \
    --system-config Ethos_U55_High_End_Embedded \
    --memory-mode Shared_Sram
