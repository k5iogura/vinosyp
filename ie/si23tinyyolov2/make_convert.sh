#!/bin/bash -v

# For Float inference
#tflite_convert --output_file=q.tflite --graph_def_file=y.pb --inference_type=FLOAT \
#               --inference_input_type=FLOAT --input_arrays=input/Placeholder --output_arrays=xoutput


# For Integer inference
#tflite_convert --output_file=yq.tflite --graph_def_file=y.pb --inference_type=QUANTIZED_UINT8 \
#               --inference_input_type=QUANTIZED_UINT8 \
#               --input_arrays=input/Placeholder --output_arrays=xoutput \
#               --std_dev_values 1 --mean_values 127 --default_ranges_min=-50 --default_ranges_max=255

# For Integer inference
tflite_convert --output_file=yq.tflite --graph_def_file=y.pb --inference_type=QUANTIZED_UINT8 \
               --inference_input_type=QUANTIZED_UINT8 \
               --input_arrays=input/Placeholder --output_arrays=xoutput \
               --std_dev_values 127 --mean_values 127 --default_ranges_min=-50 --default_ranges_max=255
