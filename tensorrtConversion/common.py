from tqdm import tqdm
import pickle
from jtop import jtop
from time import time

import tensorrt as trt
import pycuda.driver as cuda
import pycuda.autoinit

import numpy as np
import random
import torch

import json
import tensorrt as trt

import sys, os
# trt.Logger.Severity.VERBOSE
TRT_LOGGER = trt.Logger()

def get_binding_info(engine: trt.ICudaEngine):
    
    info = []
    for i in range(engine.num_io_tensors):
        name = engine.get_tensor_name(i)
        is_input = engine.get_tensor_mode(name) == trt.TensorIOMode.INPUT
        dtype = engine.get_tensor_dtype(name)
        shape = engine.get_tensor_shape(name) 
        info.append(dict(index=i, name=name, is_input=is_input, dtype=dtype, shape=tuple(shape)))
    return info

def allocate_bindings(engine: trt.ICudaEngine, context: trt.IExecutionContext, stream):
    
    if engine.num_optimization_profiles > 0:
        context.set_optimization_profile_async(0, stream.handle)

    host_inout = {}
    device_inout = {}
    bindings_ptrs = [None] * engine.num_io_tensors

    # Set the shape for all the inputs
    for i in range(engine.num_io_tensors):
        name = engine.get_tensor_name(i)
        if engine.get_tensor_mode(name) == trt.TensorIOMode.INPUT:
            engine_shape = engine.get_tensor_shape(name)
            
            if any(dim < 0 for dim in engine_shape):
                raise ValueError(
                    f"The input named '{name}' requires dynamic shape. "
                    f"Include --shape {name}=dim1,dim2,..."
                )

    # Allocate bindings with proper shapes
    for i in range(engine.num_io_tensors):
        name = engine.get_tensor_name(i)
        is_input = engine.get_tensor_mode(name) == trt.TensorIOMode.INPUT
        dtype = engine.get_tensor_dtype(name)
        
        np_dtype = np_dtype_from_trt(dtype)

        shape = tuple(context.get_tensor_shape(name))

        # Compute the array size in bytes based on the required format
        vol = int(np.prod(shape)) if len(shape) > 0 else 1
        host_buf = np.empty(vol, dtype=np_dtype)
        mem_pointer = cuda.mem_alloc(host_buf.nbytes)

        host_inout[name] = dict(is_input=is_input, shape=shape, dtype=np_dtype, buffer=host_buf)
        device_inout[name] = mem_pointer
        bindings_ptrs[i] = int(mem_pointer)

    return bindings_ptrs, host_inout, device_inout

def load_numpy_or_random(path: str | None, shape: tuple[int, ...], dtype):

    if path:
        arr = np.load(path)
        if tuple(arr.shape) != tuple(shape):
            raise ValueError(f"Shape .npy {arr.shape} different from the expected one (i.e., {shape})")
        return arr.astype(dtype, copy=False)

    if np.issubdtype(dtype, np.floating):
        return (np.random.rand(*shape).astype(dtype) * 1.0)
    elif np.issubdtype(dtype, np.integer):
        return np.random.randint(low=0, high=127, size=shape, dtype=dtype)
    elif dtype == np.bool_:
        return np.random.randint(0, 2, size=shape).astype(np.bool_)
    else:
        return np.zeros(shape, dtype=dtype)

def np_dtype_from_trt(dtype: trt.DataType):
    
    if dtype == trt.DataType.FLOAT:   return np.float32
    if dtype == trt.DataType.HALF:    return np.float16
    if dtype == trt.DataType.BF16:    return np.float16 
    if dtype == trt.DataType.INT8:    return np.int8
    if dtype == trt.DataType.INT32:   return np.int32
    if dtype == trt.DataType.INT64:   return np.int64
    if dtype == trt.DataType.BOOL:    return np.bool_
    if dtype == trt.DataType.UINT8:   return np.uint8
    raise NotImplementedError(f"Conversion not available for: {dtype} Data type")

def load_engine(plan_path: str) -> trt.ICudaEngine:
    
    assert os.path.isfile(plan_path), f"File not found at: {plan_path}"
    with open(plan_path, "rb") as f, trt.Runtime(TRT_LOGGER) as runtime:
        engine = runtime.deserialize_cuda_engine(f.read())
        if engine is None:
            raise RuntimeError("Failed engine deserialization.")
        return engine

def elementwise_mode3(y1: torch.Tensor, y2: torch.Tensor, y3: torch.Tensor, tol: float = 0.0) -> torch.Tensor:
    if tol > 0.0:
        eq12 = torch.le(torch.abs(y1 - y2), tol)
        eq13 = torch.le(torch.abs(y1 - y3), tol)
        eq23 = torch.le(torch.abs(y2 - y3), tol)
    else:
        eq12 = torch.eq(y1,y2)
        
        eq13 = torch.eq(y1,y3)
        eq23 = torch.eq(y2,y3)
    pick_y1 = torch.logical_or(eq12, eq13)
    out = torch.where(pick_y1, y1, torch.where(eq23, y2, y2))
    return out



def setup(engine_path):
	# 1) Load engine and context
	engine = load_engine(engine_path)

	context = engine.create_execution_context()
	if not context:
		raise RuntimeError("Impossibile creare IExecutionContext.")

	# 4) Allocate buffers
	stream = cuda.Stream()

	bindings_ptrs, host_inout, device_inout = allocate_bindings(engine, context, stream)

	return bindings_ptrs, host_inout, device_inout, context, stream

def run_benchmark(n_calls, func, func_params):

	# initialize benchmarking
	jetson = jtop() # create benchmark logging thread
	jetson.start() # start benchmark window
	start_time = time()

	# make function calls
	for run in range(n_calls):  
		func(**func_params)

	# clean up benchmarking
	latency = time() - start_time
	jetson.close() # close benchmark logging thread
	jetson_json = jetson.json()
	# print(jetson.json()['gpu'])

	return jetson_json, latency


def inference(sample_size, bindings_ptrs, host_inout, device_inout, context, stream):
	for name, meta in host_inout.items():
		if meta["is_input"]:
			cuda.memcpy_htod_async(device_inout[name], meta["buffer"], stream)
	for n in range(sample_size):
		# H2D for all inputs

		# Inference
		ok = context.execute_v2(bindings_ptrs)
		if not ok:
			raise RuntimeError("execute_v2 ha restituito False.")

	# D2H for all outputs
	for name, meta in host_inout.items():
		if not meta["is_input"]:
			cuda.memcpy_dtoh_async(meta["buffer"], device_inout[name], stream)

	stream.synchronize()

def benchmark(bindings_ptrs, host_inout, device_inout, context, stream, n_runs, sample_size):
	for name, meta in host_inout.items():
		if meta["is_input"]:
			if name == "obs":
				meta["buffer"][:] = load_numpy_or_random(obs_npy, meta["shape"], meta["dtype"]).ravel()
			elif name == "vec":
				meta["buffer"][:] = load_numpy_or_random(vec_npy, meta["shape"], meta["dtype"]).ravel()
			else:
				meta["buffer"][:] = load_numpy_or_random(None, meta["shape"], meta["dtype"]).ravel()

	jetson_json, times = run_benchmark(n_runs, 
										inference, 
										{
											'sample_size': sample_size,
											'bindings_ptrs': bindings_ptrs, 
											'host_inout': host_inout, 
											'device_inout': device_inout, 
											'context': context,
											'stream': stream
										})
	return jetson_json

def save_stats(jetson_json, file_path):
	stats = json.loads(jetson_json)
	
	with open(file_path, 'w') as f:
		json.dump(stats, f, indent=4)