
import tensorrt as trt
import pycuda.driver as cuda
import pycuda.autoinit

import numpy as np
import random

# We need a custom calibrator because the model expects 2 inputs
from tensorrtConversion.Calibration.calibrator import load_data, load_labels, EntropyCalibrator
import sys, os

# include the following row in the brackets to inspect the verbose TensorRT log
# trt.Logger.Severity.VERBOSE
TRT_LOGGER = trt.Logger()


def build_int8_engine_from_onnx(onnx_path, calibrator, plan_path=None, fp16_fallback=False, explicit_batch=False):

    if explicit_batch:
        flags = 1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH)
    else:
        flags = 0

    with trt.Builder(TRT_LOGGER) as builder, \
         builder.create_network(flags) as network, \
         trt.OnnxParser(network, TRT_LOGGER) as parser, \
         builder.create_builder_config() as config, \
         trt.Runtime(TRT_LOGGER) as runtime:

        with open(onnx_path, 'rb') as f:
            parsed = parser.parse(f.read())
        if not parsed:
            for i in range(parser.num_errors):
                print("ONNX parser error:", parser.get_error(i))
            raise RuntimeError("Parsing ONNX failed.")
        config.profiling_verbosity = trt.ProfilingVerbosity.DETAILED

        # Config the maximum workspace that TRT can take during the conversion to avoid memory overflow
        # config.max_workspace_size = max_workspace
        if builder.platform_has_fast_int8:
            config.set_flag(trt.BuilderFlag.INT8)

        # Set the INT8 Calibrator for a multi-input policy, other
        config.int8_calibrator = calibrator

        profile = builder.create_optimization_profile()

        if explicit_batch:
            for i in range(network.num_inputs):
                inp = network.get_input(i)
                name = inp.name
                shape = inp.shape 
                if shape[0] == -1:
                    min_shape = (1, *shape[1:])
                    opt_shape = (8, *shape[1:])
                    max_shape = (32, *shape[1:])
                    profile.set_shape(name, min_shape, opt_shape, max_shape)
        config.add_optimization_profile(profile)

        # Build the TRT engine
        serialized = builder.build_serialized_network(network, config)
        if serialized is None:
            raise RuntimeError("build_serialized_network returned None (build failed).")
        if plan_path:
            with open(plan_path, "wb") as f:
                f.write(serialized)
        print(f"[OK] Engine saved in: {plan_path}")


def build_trt_engine(onnx_path: str,
                     plan_path: str = None,
                     fp16: bool = True
                     ) -> None:
    logger = trt.Logger(trt.Logger.WARNING)
    explicit_batch = 1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH)

    with trt.Builder(logger) as builder, \
        builder.create_network(explicit_batch) as network, \
        trt.OnnxParser(network, logger) as parser, \
        builder.create_builder_config() as config:

        # FP16 (if supported)
        if fp16 and builder.platform_has_fast_fp16:
            config.set_flag(trt.BuilderFlag.FP16)

        # Parsing ONNX
        with open(onnx_path, "rb") as f:
            if not parser.parse(f.read()):
                for i in range(parser.num_errors):
                    print(f"[TRT][Parser] {parser.get_error(i)}")
                raise RuntimeError("Parsing ONNX failed.")

        # Optimization Profile 
        profile = builder.create_optimization_profile()
        input_tensor = network.get_input(0)
        
        config.add_optimization_profile(profile)
        config.profiling_verbosity = trt.ProfilingVerbosity.DETAILED

        # Build and serialize the engine
        engine_bytes = builder.build_serialized_network(network, config)
        if engine_bytes is None:
            raise RuntimeError("Build engine failed.")

        if plan_path:
            with open(plan_path, "wb") as f:
                f.write(engine_bytes)
        print(f"[OK] Engine saved in: {plan_path}")