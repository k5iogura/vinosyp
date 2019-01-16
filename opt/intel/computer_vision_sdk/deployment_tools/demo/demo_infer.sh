#!/bin/bash

# Copyright (c) 2018 Intel Corporation
#
#  Licensed under the Apache License, Version 2.0 (the "License");
#  you may not use this file except in compliance with the License.
#  You may obtain a copy of the License at
#
#       http://www.apache.org/licenses/LICENSE-2.0
#
#  Unless required by applicable law or agreed to in writing, software
#  distributed under the License is distributed on an "AS IS" BASIS,
#  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#  See the License for the specific language governing permissions and
#  limitations under the License.

usage() {
    echo "Classification demo using public SqueezeNet topology"
    echo "-d name     specify the target device to infer on; CPU, GPU, FPGA or MYRIAD are acceptable. Sample will look for a suitable plugin for device specified"
    echo "-help            print help message"
    exit 1
}

error() {
    local code="${3:-1}"
    if [[ -n "$2" ]];then
        echo "Error on or near line $1: $2; exiting with status ${code}"
    else
        echo "Error on or near line $1; exiting with status ${code}"
    fi
    exit "${code}"
}
trap 'error ${LINENO}' ERR

target="CPU"

# parse command line options
while [[ $# -gt 0 ]]
do
key="$1"

case $key in
    -h | -help | --help)
    usage
    ;;
    -d)
    target="$2"
    echo target = "${target}"
    shift
    ;;
    -sample-options)
    sampleoptions="$2 $3 $4 $5 $6"
    echo sample-options = "${sampleoptions}"
    shift
    ;;
    *)
    # unknown option
    ;;
esac
shift
done

ROOT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"

if [ "$target" = "MYRIAD" ]; then
    # MYRIAD supports networks with FP16 format only
    target_precision="FP16"
else
    target_precision="FP32"
fi
printf "target_precision = ${target_precision}\n"

models_path="$HOME/openvino_models/models/${target_precision}"
irs_path="$HOME/openvino_models/ir/${target_precision}/"

model_name="squeezenet"
model_version="1.1"
model_type="classification"
model_framework="caffe"
dest_model_proto="${model_name}${model_version}.prototxt"
dest_model_weights="${model_name}${model_version}.caffemodel"

model_dir="${model_type}/${model_name}/${model_version}/${model_framework}"
ir_dir="${irs_path}/${model_dir}"

proto_file_path="${models_path}/${model_dir}/${dest_model_proto}"
weights_file_path="${models_path}/${model_dir}/${dest_model_weights}"

target_image_path="$ROOT_DIR/car.png"

run_again="Then run the script again\n\n"
dashes="\n\n###################################################\n\n"


#if [[ -z "${INTEL_CVSDK_DIR}" ]]; then
#    printf "\n\nINTEL_CVSDK_DIR environment variable is not set. Trying to run ./setupvars.sh to set it. \n"
#
#    if [ -e "$ROOT_DIR/../inference_engine/bin/setvars.sh" ]; then # for Intel Deep Learning Deployment Toolkit package
#        setupvars_path="$ROOT_DIR/../inference_engine/bin/setvars.sh"
#    elif [ -e "$ROOT_DIR/../../bin/setupvars.sh" ]; then # for Intel CV SDK package
#        setupvars_path="$ROOT_DIR/../../bin/setupvars.sh"
#    elif [ -e "$ROOT_DIR/../../setupvars.sh" ]; then # for Intel GO SDK package
#        setupvars_path="$ROOT_DIR/../../setupvars.sh"
#    else
#        printf "Error: setupvars.sh is not found\n"
#    fi
#    if ! source $setupvars_path ; then
#        printf "Unable to run ./setupvars.sh. Please check its presence. ${run_again}"
#        exit 1
#    fi
#fi

# Step 1. Download the Caffe model and the prototxt of the model
#printf "${dashes}"
#printf "\n\nDownloading the Caffe model and the prototxt"

cur_path=$PWD

#printf "\nInstalling dependencies\n"
#
#if [[ -f /etc/centos-release ]]; then
#    DISTRO="centos"
#elif [[ -f /etc/lsb-release ]]; then
#    DISTRO="ubuntu"
#fi

#if [[ $DISTRO == "centos" ]]; then
#    sudo -E yum install -y centos-release-scl epel-release
#    sudo -E yum install -y gcc gcc-c++ make glibc-static glibc-devel libstdc++-static libstdc++-devel libstdc++ libgcc \
#                           glibc-static.i686 glibc-devel.i686 libstdc++-static.i686 libstdc++.i686 libgcc.i686 cmake
#
#    sudo -E rpm -Uvh http://li.nux.ro/download/nux/dextop/el7/x86_64/nux-dextop-release-0-1.el7.nux.noarch.rpm || true
#    sudo -E yum install -y epel-release
#    sudo -E yum install -y cmake ffmpeg gstreamer1 gstreamer1-plugins-base libusbx-devel
#
#    # check installed Python version
#    if command -v python3.5 >/dev/null 2>&1; then
#        python_binary=python3.5
#        pip_binary=pip3.5
#    fi
#    if command -v python3.6 >/dev/null 2>&1; then
#        python_binary=python3.6
#        pip_binary=pip3.6
#    fi
#    if [ -z "$python_binary" ]; then
#        sudo -E yum install -y rh-python36 || true
#        source scl_source enable rh-python36
#        python_binary=python3.6
#        pip_binary=pip3.6
#    fi
#elif [[ $DISTRO == "ubuntu" ]]; then
#    printf "Run sudo -E apt -y install build-essential python3-pip virtualenv cmake libcairo2-dev libpango1.0-dev libglib2.0-dev libgtk2.0-dev libswscale-dev libavcodec-dev libavformat-dev libgstreamer1.0-0 gstreamer1.0-plugins-base\n"
#    sudo -E apt update
#    sudo -E apt -y install build-essential python3-pip virtualenv cmake libcairo2-dev libpango1.0-dev libglib2.0-dev libgtk2.0-dev libswscale-dev libavcodec-dev libavformat-dev libgstreamer1.0-0 gstreamer1.0-plugins-base
#    python_binary=python3
#    pip_binary=pip3
#
#    system_ver=`cat /etc/lsb-release | grep -i "DISTRIB_RELEASE" | cut -d "=" -f2`
#    if [ $system_ver = "18.04" ]; then
#        sudo -E apt-get install -y libpng-dev
#    else
#        sudo -E apt-get install -y libpng12-dev
#    fi
#fi
#
#if ! command -v $python_binary &>/dev/null; then
#    printf "\n\nPython 3.5 (x64) or higher is not installed. It is required to run Model Optimizer, please install it. ${run_again}"
#    exit 1
#fi
#
#sudo -E $pip_binary install pyyaml requests
#
#downloader_path="${INTEL_CVSDK_DIR}/deployment_tools/model_downloader/downloader.py"
#
#if ! [ -f "${proto_file_path}" ] && ! [ -f "${weights_file_path}" ]; then
#    printf "\nRun $downloader_path --name ${model_name}${model_version} --output_dir ${models_path}\n\n"
#    $python_binary $downloader_path --name ${model_name}${model_version} --output_dir ${models_path};
#else
#    printf "\nModels have been loaded previously. Skip loading model step."
#    printf "\nModel path: ${proto_file_path}\n"
#fi

if [ ! -e "$ir_dir" ]; then
    # Step 2. Configure Model Optimizer
    printf "${dashes}"
    printf "Install Model Optimizer dependencies\n\n"
    cd "${INTEL_CVSDK_DIR}/deployment_tools/model_optimizer/install_prerequisites"
    source "install_prerequisites.sh" caffe
    cd $cur_path

    # Step 3. Convert a model with Model Optimizer
    printf "${dashes}"
    printf "Convert a model with Model Optimizer\n\n"

    mo_path="${INTEL_CVSDK_DIR}/deployment_tools/model_optimizer/mo.py"

    export PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=cpp
    printf "Run $python_binary $mo_path --input_model ${weights_file_path} $ir_dir --data_type $target_precision\n\n"
    $python_binary $mo_path --input_model ${weights_file_path} --output_dir $ir_dir --data_type $target_precision
else
    printf "\n\nTarget folder ${ir_dir} already exists. Skipping IR generation  with Model Optimizer."
    printf "If you want to convert a model again, remove the entire ${ir_dir} folder. ${run_again}"
fi

# Step 4. Build samples
printf "${dashes}"
printf "Build Inference Engine samples\n\n"

samples_path="${INTEL_CVSDK_DIR}/deployment_tools/inference_engine/samples"
build_dir="$HOME/inference_engine_samples"
binaries_dir="${build_dir}/intel64/Release"
if [ ! -e "$binaries_dir/classification_sample" ]; then
    if ! command -v cmake &>/dev/null; then
        printf "\n\nCMAKE is not installed. It is required to build Inference Engine samples. Please install it. ${run_again}"
        exit 1
    fi
    mkdir -p $build_dir
    cd $build_dir
    cmake -DCMAKE_BUILD_TYPE=Release $samples_path
    make -j8 classification_sample
else
    printf "\n\nTarget folder ${build_dir} already exists. Skipping samples building. "
    printf "If you want to rebuild samples, remove the entire ${build_dir} folder. ${run_again}"
fi

# Step 5. Run samples
printf "${dashes}"
printf "Run Inference Engine classification sample\n\n"

cd $binaries_dir

cp -f $ROOT_DIR/${model_name}${model_version}.labels ${ir_dir}/

printf "Run ./classification_sample -d $target -i $target_image_path -m ${ir_dir}/${model_name}${model_version}.xml ${sampleoptions}\n\n"
./classification_sample -d $target -i $target_image_path -m "${ir_dir}/${model_name}${model_version}.xml" ${sampleoptions}

printf "${dashes}"
printf "Demo completed successfully.\n\n"
