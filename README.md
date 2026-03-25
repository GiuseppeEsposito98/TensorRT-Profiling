# TensorRT-Profiling
This repository provides the code to:
- Convert the Pytorch-based Neural Networks for autonomous drone navigation in TensorRT compatible format
- Profile at the kernel-level the operations graph by inspecting each NN layer.
- Profile the inference time through the Hardware Program Counters.
- Profile the energy consumption through internal Hardware Telemetry Sensors.
On a NVIDIA Jetson-based device, such as Jetson NANO.

## Requirements
TensorRT version >= 10.x
Python version >= 3.9

## Install
1. Download [TensorRT](https://developer.nvidia.com/tensorrt) compatible with your system setup
2. Install TensorRT following the instructions provided at this Official [Repository] (https://github.com/NVIDIA/TensorRT/tree/a180e08111b61adf0fee4baa86bc33f1633745f2)

3. Setup the python environment in such a way that the system packages are visible.
```bash
python -m venv --system-site-packages benchmark
source benchmark/bin/activate
```
4. Install the version of the following python libraries compatible with your Jetson device (in particular with its Jetpack version). For exmaple for the latest Jetson Orin Nano, you can:
```bash
pip3 install --no-cache https://developer.download.nvidia.com/compute/redist/jp/v61/pytorch/torch-2.5.0a0+872d972e41.nv24.08.17622132-cp310-cp310-linux_aarch64.whl
pip3 install pycuda
pip3 install tensorrt==10.3.0
pip install tqdm
pip install jetson-stats
```

5. Setup the Python environment variable

```bash
cd ~
PWD=`pwd`
export PYTHONPATH="$PWD"
```

## Usage

1. Run the script tensorrtConversion/torch2trt.py with the desired data type and the desired map
```bash
python tensorrtConversion/torch2trt.py --format FP16 --map NH
```

2. Profile energy on the model for autonomous drone navigation trained for Airsim Neighbourhood map, on 10 runs, 10 samples exported in the FP16 format.
```bash
bash complete_profiling.sh ./ConvertedNNs NH 10 10 FP16
```

3. You will find the final report in out_report/NH/report.csv
