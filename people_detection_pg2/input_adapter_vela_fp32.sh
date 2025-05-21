vela input_adapter/input_adapter_float32.tflite \
    --output-dir  input_adapter/vela_he_fp32 \
    --accelerator-config ethos-u55-128 \
    --optimise Performance \
    --config vela.ini \
    --system-config Ethos_U55_High_End_Embedded \
    --memory-mode Shared_Sram