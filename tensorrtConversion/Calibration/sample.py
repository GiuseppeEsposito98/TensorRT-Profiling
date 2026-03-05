# In questo esempio facciamo solo la calibrazione e l'inferenza del modello in formato int8
import tensorrt as trt
import pycuda.driver as cuda
import pycuda.autoinit

import numpy as np
import random

# For our custom calibrator
from calibrator import load_data, load_labels, EntropyCalibrator

import sys, os
from map_tool_box.scripts.tensorrtConversion.common import get_binding_info, allocate_bindings, load_numpy_or_random, np_dtype_from_trt
from map_tool_box.scripts.tensorrtConversion.ConverterUtils import build_int8_engine_from_onnx


# This function builds an engine from a Caffe model.


def inference(context, bindings_ptrs, host_inout, device_inout, stream, batch_size, obs_npy=None, vec_npy=None):
    for name, meta in host_inout.items():
        if meta["is_input"]:
            if name == "obs":
                meta["buffer"][:] = load_numpy_or_random(obs_npy, meta["shape"], meta["dtype"]).ravel()
            elif name == "vec":
                meta["buffer"][:] = load_numpy_or_random(vec_npy, meta["shape"], meta["dtype"]).ravel()
            else:
                # per eventuali input addizionali
                meta["buffer"][:] = load_numpy_or_random(None, meta["shape"], meta["dtype"]).ravel()

    # H2D per tutti gli input
    for name, meta in host_inout.items():
        if meta["is_input"]:
            cuda.memcpy_htod_async(device_inout[name], meta["buffer"], stream)

    # Inference
    ok = context.execute_v2(bindings_ptrs)  # esecuzione sincrona sullo stream di default
    if not ok:
        raise RuntimeError("execute_v2 ha restituito False.")

    # D2H per tutti gli output
    for name, meta in host_inout.items():
        if not meta["is_input"]:
            cuda.memcpy_dtoh_async(meta["buffer"], device_inout[name], stream)

    stream.synchronize()


def main():

    # Now we create a calibrator and give it the location of our calibration data.
    # We also allow it to cache calibration data for faster engine building.
    model_file = '../ConvertedNNs/NN/FP16/qnet.onnx'
    calibration_cache = "calibration.cache"

    obs_npy = None
    vec_npy = None

    calib = EntropyCalibrator(training_data=None, cache_file=calibration_cache)

    stream = cuda.Stream()
    with build_int8_engine_from_onnx(model_file, calib) as engine, engine.create_execution_context() as context:
        bindings_ptrs, host_inout, device_inout = allocate_bindings(engine, context, stream)
        print(bindings_ptrs)
        bindings = get_binding_info(engine)
        batch_size = bindings[0]['shape'][0]
        inference(context, bindings_ptrs, host_inout, device_inout, stream, batch_size, obs_npy, vec_npy)

if __name__ == '__main__':
    main()