import os
import json
import argparse
import numpy as np
import tensorrt as trt
import pycuda.driver as cuda
import pycuda.autoinit
from tensorrtConversion.common import get_binding_info, allocate_bindings, load_numpy_or_random, np_dtype_from_trt, load_engine


TRT_LOGGER = trt.Logger(trt.Logger.WARNING)


def run_pipeline():
    export_mode = 'NN'
    fp16=True
    int8=False
    mapUT ='blocks'

    root_plan_path = 'ConvertedNNs'
    engine_file_name = 'qnet.plan'
    obs_npy = None
    vec_npy = None
    run=5

    root_plan_path = os.path.join(root_plan_path, mapUT)
    root_plan_path = os.path.join(root_plan_path, export_mode)
    
    if int8:
        root_plan_path = os.path.join(root_plan_path, 'INT8')
    else:
        root_plan_path = os.path.join(root_plan_path, 'FP16')
    
    root_plan_path = os.path.join(root_plan_path, engine_file_name)

    # 1) Load the engine
    engine = load_engine(root_plan_path)
    context = engine.create_execution_context()

    if not context:
        raise RuntimeError("Impossibile creare IExecutionContext.")
    
    # 2) Info binding
    bindings = get_binding_info(engine)

    print("== Bindings ==")
    for b in bindings:
        print(f"[{'IN ' if b['is_input'] else 'OUT'}] {b['index']:2d}  {b['name']:<20} "
              f"dtype={b['dtype']}  shape={b['shape']}")

    # 4) Allocate buffer H2D/D2H for all the necessary bindings
    stream = cuda.Stream()
    bindings_ptrs, host_inout, device_inout = allocate_bindings(engine, context, stream)

    # 5) Initialize inputs (from .npy or random)
    dtype = np.float32
    if int8:
        dtype = np.int8
    
    for name, meta in host_inout.items():
        if meta["is_input"]:
            if name == "obs": # meta["dtype"]
                meta["buffer"][:] = load_numpy_or_random(obs_npy, meta["shape"], dtype).ravel()
                print(f'ARRAY DTYPE: {meta["buffer"][:].dtype}')
            elif name == "vec":
                meta["buffer"][:] = load_numpy_or_random(vec_npy, meta["shape"], dtype).ravel()
                print(f'ARRAY DTYPE: {meta["buffer"][:].dtype}')

    # 6) Perform one or more inferences
    for it in range(run):
        # H2D for all the inputs
        for name, meta in host_inout.items():
            if meta["is_input"]:
                cuda.memcpy_htod_async(device_inout[name], meta["buffer"], stream)

        # Inference
        ok = context.execute_v2(bindings_ptrs)  # Synchronous inference run
        if not ok:
            raise RuntimeError("execute_v2 ha restituito False.")

        # D2H for all the outputs
        for name, meta in host_inout.items():
            if not meta["is_input"]:
                cuda.memcpy_dtoh_async(meta["buffer"], device_inout[name], stream)

        stream.synchronize()

        print(f"\n[Inference {it+1}] Output:")
        for name, meta in host_inout.items():
            if not meta["is_input"]:
                arr = meta["buffer"].reshape(meta["shape"])
                print(f" - {name}: shape={meta['shape']} dtype={meta['dtype']}, "
                      f"min={arr.min():.5f} max={arr.max():.5f} mean={arr.mean():.5f}")


def main():
    run_pipeline()


if __name__ == "__main__":
    main()