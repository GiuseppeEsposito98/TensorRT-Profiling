#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import argparse
import torch
import onnx
import pickle
from pathlib import Path
import tensorrt as trt
from map_tool_box.scripts.PTmodels.sb3net import SB3Net
import numpy as np
from collections import OrderedDict
from map_tool_box.scripts.tensorrtConversion.ConverterUtils import build_int8_engine_from_onnx, build_trt_engine
from map_tool_box.scripts.tensorrtConversion.Calibration.calibrator import EntropyCalibrator
import json
import sys
from pathlib import Path


def iter_shape_leaves(s):
    if isinstance(s, int):
        yield (s,)
    elif isinstance(s, (tuple, list)) and all(isinstance(d, int) for d in s):
        yield tuple(s)
    elif isinstance(s, (tuple, list)):
        for x in s:
            yield from iter_shape_leaves(x)
    else:
        yield (int(s),)

def make_inputs(input_shapes, low=0, high=100):
    leaves = list(iter_shape_leaves(input_shapes))
    return tuple(torch.randn(*sh, dtype=torch.float32, device='cuda') for sh in leaves)
    # print(len(inputs))
# Esempio
# input_shapes = [[(3, 3, 144, 256), (3, 12)]]
# inputs = make_inputs(input_shapes, dtype='float32')


def export_to_onnx(model: torch.nn.Module,
                   onnx_path: str,
                   input_shapes=None,
                   dynamic: bool = True,
                   export_mode=None) -> None:
    
    inputs = make_inputs(input_shapes)



    # input_names = ["obs", "vec"]
    output_names = ["output"]
    print(model)

    # dummy_input = tuple([dummy_obs, dummy_vec])

    dynamic_axes = None
    print(f'input shape: {inputs[0].shape}')
    torch.onnx.export(
        model, 
        inputs, 
        onnx_path,
        output_names=output_names
    )

    onnx_model = onnx.load(onnx_path)
    onnx.checker.check_model(onnx_model)

    

def iter_leaves(m, p=None):
    for n, c in m.named_children():
        f = f"{p}.{n}" if p else n
        if any(c.named_children()):
            yield from iter_leaves(c, f)
        else:
            yield c, f

def pick_layer_by_idx(model, lyr_idx):
    for i, (mod, name) in enumerate(iter_leaves(model)):
        if i == lyr_idx:
            return mod, name
    raise IndexError("Layer index fuori range")

def main():

    ap = argparse.ArgumentParser(description="Benchmarking NN performance")
    ap.add_argument("--format", default = "FP16", help="Target data type")
    ap.add_argument("--map", default = "blocks", help="Target map")
    args = ap.parse_args()

    # default export parameters
    no_dynamic = True
    workspace_gb=None
    min_shape = None # change these if you want to set dynamic shapes (i.e., if the NN can receive inputs with different resolutions w.r.t. the expected one)
    opt_shape = None
    max_shape = None
    export_mode = 'NN' # if set to 'layers' this script will perform a layer-wise conversion
    mapUT = args.map

    input_shapes = list()

    if mapUT == 'NH':
        input_shapes.append((3, 3,144,256))
        input_shapes.append((3,12))
    elif mapUT == 'blocks':
        input_shapes.append((1,4,36,64))
        input_shapes.append((1,12))
        
    # default paths
    root_save_path = './'
    onnx_path = 'NN.onnx'
    pickle_path = f'./PTmodels/{mapUT}/sb3net.p'

    with open(pickle_path, 'rb') as f:
        model_arch = pickle.load(f)
    model = SB3Net(model_arch.cnn_extractor, model_arch.linear_extractor, model_arch.vec_extractor, model_arch.q_net)

    
    print(model)
    
    root_save_path = Path(f'ConvertedNNs/{mapUT}/{export_mode}')
    
    plan_path = 'NN.plan'

    onnx_path = os.path.join(root_save_path, onnx_path)

    if 'fp16' in args.format.lower():
        root_save_path = os.path.join(root_save_path, 'FP16')
    elif 'int8' in args.format.lower():
        root_save_path = os.path.join(root_save_path, 'INT8')

    Path(root_save_path).mkdir(parents=True, exist_ok=True)

    plan_path = os.path.join(root_save_path, plan_path)
        
    

    if 'fp16' in args.format.lower():
        export_to_onnx(
            model=model,
            onnx_path=onnx_path,
            input_shapes = input_shapes,
            dynamic=no_dynamic
        )
        print(f"[OK] ONNX salvato: {onnx_path}")
        build_trt_engine(
            onnx_path=onnx_path,
            plan_path=plan_path,
        )

    elif 'int8' in args.format.lower():
        export_to_onnx(
            model=model,
            onnx_path=onnx_path,
            input_shapes = input_shapes,
            dynamic=no_dynamic
        )
        print(f"[OK] ONNX salvato: {onnx_path}")
        try:
            calibration_cache = os.path.join(root_save_path, 'calibration.cache')
            calib = EntropyCalibrator(training_data=None, cache_file=calibration_cache, inputs_shape=input_shapes)
            build_int8_engine_from_onnx(
                onnx_path=onnx_path,
                plan_path=plan_path,
                calibrator=calib
                )

        except RuntimeError as e:
            msg = f"Returned error: {e} and saved in {os.path.join(root_save_path, 'log.txt')}"
            with open(os.path.join(root_save_path, 'log.txt'), 'w') as f:
                f.write(msg)

if __name__ == "__main__":
    main()