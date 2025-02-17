#!/bin/bash

# Check if the weights_path argument is provided
if [ -z "$1" ]; then
  echo "Usage: $0 <weights_path>"
  exit 1
fi

# Assign the first argument to weights_path
weights_path=$1

# Run the vela command with the provided weights_path
vela "${weights_path}/best_saved_model/best_full_integer_quant.tflite" \
    --output-dir "${weights_path}/best_saved_model/vela_he" \
    --accelerator-config ethos-u55-128 \
    --optimise Performance \
    --config vela.ini \
    --system-config Ethos_U55_High_End_Embedded \
    --memory-mode Shared_Sram

vela "${weights_path}/head_saved_model/head_float32.tflite" \
    --output-dir "${weights_path}/head_saved_model/vela_he" \
    --accelerator-config ethos-u55-128 \
    --optimise Performance \
    --config vela.ini \
    --system-config Ethos_U55_High_End_Embedded \
    --memory-mode Shared_Sram