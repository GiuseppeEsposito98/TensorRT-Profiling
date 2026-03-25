
import tensorrt as trt
import pycuda.driver as cuda
import pycuda.autoinit

import numpy as np
import random

# For our custom calibrator
from tensorrtConversion.Calibration.calibrator import load_data, load_labels, EntropyCalibrator
import sys, os
# trt.Logger.Severity.VERBOSE
TRT_LOGGER = trt.Logger()


def build_int8_engine_from_onnx(onnx_path, calibrator, plan_path=None, fp16_fallback=False, explicit_batch=False):
    # explicit_batch consigliato con modelli moderni (ONNX)
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
        # Config
        # config.max_workspace_size = max_workspace
        if builder.platform_has_fast_int8:
            config.set_flag(trt.BuilderFlag.INT8)

        # Calibratore INT8
        config.int8_calibrator = calibrator

        # Profili per shape dinamiche (se servono)
        # ESEMPIO: imposta un profilo per input 'obs' e 'vec'
        profile = builder.create_optimization_profile()

        if explicit_batch:
            for i in range(network.num_inputs):
                inp = network.get_input(i)
                name = inp.name
                shape = inp.shape  # es: (-1, 3, 144, 256) con batch dinamico
                if shape[0] == -1:
                    # Definisci (min, opt, max) batch sizes a tua scelta
                    min_shape = (1, *shape[1:])
                    opt_shape = (8, *shape[1:])
                    max_shape = (32, *shape[1:])
                    profile.set_shape(name, min_shape, opt_shape, max_shape)
        config.add_optimization_profile(profile)

        # Costruisci il motore
        serialized = builder.build_serialized_network(network, config)
        if serialized is None:
            raise RuntimeError("build_serialized_network returned None (build failed).")
        if plan_path:
            with open(plan_path, "wb") as f:
                f.write(serialized)
        print(f"[OK] Engine salvato: {plan_path}")


def build_trt_engine(onnx_path: str,
                     plan_path: str = None,
                     fp16: bool = True
                    #  workspace_gb: float = 2.0,
                     # Profili dinamici (N,C,H,W). Deve essere coerente con l'ONNX.
                    #  min_shape=(1, 3, 224, 224),
                    #  opt_shape=(4, 3, 640, 640),
                    #  max_shape=(8, 3, 1280, 1280)
                     ) -> None:
    logger = trt.Logger(trt.Logger.WARNING)
    explicit_batch = 1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH)

    with trt.Builder(logger) as builder, \
        builder.create_network(explicit_batch) as network, \
        trt.OnnxParser(network, logger) as parser, \
        builder.create_builder_config() as config:

        # FP16 (se supportato dalla piattaforma)
        if fp16 and builder.platform_has_fast_fp16:
            config.set_flag(trt.BuilderFlag.FP16)

        # Parsing ONNX
        with open(onnx_path, "rb") as f:
            if not parser.parse(f.read()):
                for i in range(parser.num_errors):
                    print(f"[TRT][Parser] {parser.get_error(i)}")
                raise RuntimeError("Parsing ONNX fallito. Vedi errori sopra.")

        # Optimization Profile per input dinamico
        profile = builder.create_optimization_profile()
        input_tensor = network.get_input(0)
        assert input_tensor.name, "Input senza nome?"
        # profile.set_shape(input_tensor.name, min=min_shape, opt=opt_shape, max=max_shape)
        config.add_optimization_profile(profile)
        config.profiling_verbosity = trt.ProfilingVerbosity.DETAILED

        # Costruisci e serializza
        engine_bytes = builder.build_serialized_network(network, config)
        if engine_bytes is None:
            raise RuntimeError("Build engine fallita.")

        if plan_path:
            with open(plan_path, "wb") as f:
                f.write(engine_bytes)
        print(f"[OK] Engine salvato: {plan_path}")