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
#tflite_convert --output_file=yq.tflite --graph_def_file=y.pb --inference_type=QUANTIZED_UINT8 \
#               --inference_input_type=QUANTIZED_UINT8 \
#               --input_arrays=input/Placeholder --output_arrays=xoutput \
#               --std_dev_values 127 --mean_values 127 --default_ranges_min=-50 --default_ranges_max=255

# For all node keep'in
tflite_convert \
--output_file=yq.tflite \
--graph_def_file=y.pb \
--inference_type=QUANTIZED_UINT8 \
--inference_input_type=QUANTIZED_UINT8 \
--input_arrays=input/Placeholder \
--std_dev_values 127 \
--mean_values 127 \
--default_ranges_min=-50 \
--default_ranges_max=255 \
--output_array="xoutput,add,mul,Maximum,MaxPool,add_1,mul_1,Maximum_1,MaxPool_1,add_2,mul_2,Maximum_2,MaxPool_2,add_3,mul_3,Maximum_3,MaxPool_3,add_4,mul_4,Maximum_4,MaxPool_4,add_5,mul_5,Maximum_5,MaxPool_5,add_6,mul_6,Maximum_6,add_7,mul_7,Maximum_7"

