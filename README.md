# TensorRT Profiling
This repository provides the scripts to:
- Convert the Pytorch-based Neural Networks for autonomous drone navigation in TensorRT compatible format
- Profile at the kernel-level the operations graph by inspecting each NN layer.
- Profile the inference time through the Hardware Program Counters.
- Profile the energy consumption through internal Hardware Telemetry Sensors.

## Requirements
TensorRT version >= 10.x

## Usage
1. Download [TensorRT](https://developer.nvidia.com/tensorrt) compatible with your system setup
2. Install TensorRT following the instructions provided at this Official [Repository] (https://github.com/NVIDIA/TensorRT/tree/a180e08111b61adf0fee4baa86bc33f1633745f2)

```bash
cd ~/Desktop/experimental/benchmarks
PWD=`pwd`
export PYTHONPATH="$PWD"
cd ~/Desktop/experimental/benchmarks/map_tool_box/scripts
source ~/benchmark/bin/activate
```

3. In PTmodels you can setup a folder that contains all you need to instantiate your NN

4. If you follow the procedure available in tensorrtConversion/torch2trt.py script (lines 115-117) you don't need to make any modification. Otherwise, you can modify those lines according to your NN initialization modalities

5. Run the script tensorrtConversion/torch2trt.py with the desired data type and the desired map
```bash
python tensorrtConversion/torch2trt.py --format FP16 --map NH
```

6. Profile telemetry metrics and inference time
```bash
bash complete_profiling.sh ./ConvertedNNs NH 10 10 FP16
```

7. You will find the final report in out_report/NH/report.csv
