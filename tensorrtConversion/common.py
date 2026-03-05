
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
    """Raccoglie info sui binding (nome, indice, is_input, dtype, shape)."""
    info = []
    for i in range(engine.num_io_tensors):
        name = engine.get_tensor_name(i)
        is_input = engine.get_tensor_mode(name) == trt.TensorIOMode.INPUT
        dtype = engine.get_tensor_dtype(name)
        shape = engine.get_tensor_shape(name)  # static shape o -1 per dim dinamiche
        info.append(dict(index=i, name=name, is_input=is_input, dtype=dtype, shape=tuple(shape)))
    return info

def allocate_bindings(engine: trt.ICudaEngine, context: trt.IExecutionContext, stream):
    """
    Alloca buffer host/device per tutti i binding, impostando le shape dinamiche se necessario.
    Ritorna: (bindings_ptrs, host_inout, device_inout)
    """
    # Se l'engine usa profili, usa l'indice 0 di default
    if engine.num_optimization_profiles > 0:
        context.set_optimization_profile_async(0, stream.handle)

    host_inout = {}
    device_inout = {}
    bindings_ptrs = [None] * engine.num_io_tensors

    # Prima imposta le shape per tutti gli input dinamici
    for i in range(engine.num_io_tensors):
        name = engine.get_tensor_name(i)
        if engine.get_tensor_mode(name) == trt.TensorIOMode.INPUT:
            engine_shape = engine.get_tensor_shape(name)
            # Shape dinamica se contiene -1
            if any(dim < 0 for dim in engine_shape):
                raise ValueError(
                    f"L'input '{name}' richiede shape dinamica, ma non è stata fornita. "
                    f"Specifica --shape {name}=dim1,dim2,..."
                )

    # Ora alloca tutti i binding con le shape effettive
    for i in range(engine.num_io_tensors):
        name = engine.get_tensor_name(i)
        is_input = engine.get_tensor_mode(name) == trt.TensorIOMode.INPUT
        dtype = engine.get_tensor_dtype(name)
        
        np_dtype = np_dtype_from_trt(dtype)

        shape = tuple(context.get_tensor_shape(name))

        # Calcola dimensione in elementi/bytes
        vol = int(np.prod(shape)) if len(shape) > 0 else 1
        host_buf = np.empty(vol, dtype=np_dtype)
        mem_pointer = cuda.mem_alloc(host_buf.nbytes)

        host_inout[name] = dict(is_input=is_input, shape=shape, dtype=np_dtype, buffer=host_buf)
        device_inout[name] = mem_pointer
        bindings_ptrs[i] = int(mem_pointer)

    return bindings_ptrs, host_inout, device_inout

def load_numpy_or_random(path: str | None, shape: tuple[int, ...], dtype):
    """Carica da .npy se path è dato; altrimenti genera un tensore randomico con shape/dtype."""
    # dtype=np.int8
    if path:
        arr = np.load(path)
        if tuple(arr.shape) != tuple(shape):
            raise ValueError(f"Shape .npy {arr.shape} diversa da shape attesa {shape}")
        return arr.astype(dtype, copy=False)
    # default: random
    if np.issubdtype(dtype, np.floating):
        return (np.random.rand(*shape).astype(dtype) * 1.0)
    elif np.issubdtype(dtype, np.integer):
        return np.random.randint(low=0, high=127, size=shape, dtype=dtype)
    elif dtype == np.bool_:
        return np.random.randint(0, 2, size=shape).astype(np.bool_)
    else:
        return np.zeros(shape, dtype=dtype)

def np_dtype_from_trt(dtype: trt.DataType):
    """Mappa DataType TensorRT -> dtype numpy."""
    if dtype == trt.DataType.FLOAT:   return np.float32
    if dtype == trt.DataType.HALF:    return np.float16
    if dtype == trt.DataType.BF16:    return np.float16  # approssimazione: I/O spesso FP32/FP16
    if dtype == trt.DataType.INT8:    return np.int8
    if dtype == trt.DataType.INT32:   return np.int32
    if dtype == trt.DataType.INT64:   return np.int64
    if dtype == trt.DataType.BOOL:    return np.bool_
    if dtype == trt.DataType.UINT8:   return np.uint8
    raise NotImplementedError(f"DataType TRT non gestito: {dtype}")

def load_engine(plan_path: str) -> trt.ICudaEngine:
    """Deserializza un engine .plan dal disco."""
    assert os.path.isfile(plan_path), f"File non trovato: {plan_path}"
    with open(plan_path, "rb") as f, trt.Runtime(TRT_LOGGER) as runtime:
        engine = runtime.deserialize_cuda_engine(f.read())
        if engine is None:
            raise RuntimeError("Deserializzazione engine fallita.")
        return engine

def elementwise_mode3(y1: torch.Tensor, y2: torch.Tensor, y3: torch.Tensor, tol: float = 0.0) -> torch.Tensor:
    if tol > 0.0:
        eq12 = torch.le(torch.abs(y1 - y2), tol)
        eq13 = torch.le(torch.abs(y1 - y3), tol)
        eq23 = torch.le(torch.abs(y2 - y3), tol)
    else:
        eq12 = torch.eq(y1,y2)
        # print(eq12)
        eq13 = torch.eq(y1,y3)
        eq23 = torch.eq(y2,y3)
    pick_y1 = torch.logical_or(eq12, eq13)
    out = torch.where(pick_y1, y1, torch.where(eq23, y2, y2))
    return out
