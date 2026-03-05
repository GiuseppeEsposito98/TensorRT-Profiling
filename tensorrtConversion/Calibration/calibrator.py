import os
import numpy as np
import tensorrt as trt
import pycuda.driver as cuda
import pycuda.autoinit

def load_labels(filepath=None, obs_shape=(100, 3, 144, 256)):
    if filepath:
        with open(filepath, "rb") as f:
            raw_buf = np.fromstring(f.read(), dtype=np.uint8)
    else:
        instances, c, h, w = obs_shape
        labels = np.random.randint(0, 20, instances, dtype=np.int8)
    return labels.astype(np.int32).reshape(instances)

def load_data(filepath=None, inputs_shape=None, num_samples=512):
    if filepath:
        raise NotImplementedError
    xs = []
    # print(f'inputs_shape: {inputs_shape}')
    for shape in inputs_shape:
        # print(f'shape: {shape}')
        x = np.random.rand(num_samples, *shape).astype(np.float32)
        xs.append(np.ascontiguousarray(x))
    return xs

class EntropyCalibrator(trt.IInt8EntropyCalibrator2):
    def __init__(self, cache_file, training_data=None, batch_size=64, inputs_shape=None):
        super().__init__()
        self.cache_file = cache_file
        self.batch_size = batch_size
        self.inputs_shape = inputs_shape
        self.inputs = load_data(training_data, inputs_shape)
        self.current_index = 0
        self.num_samples = self.inputs[0].shape[0]
        if len(self.inputs) == 2:
            sample_bytes0 = self.inputs[0][0:1].nbytes
            sample_bytes1 = self.inputs[1][0:1].nbytes
            self.device_input_obs = cuda.mem_alloc(sample_bytes0 * batch_size)
            self.device_input_vec = cuda.mem_alloc(sample_bytes1 * batch_size)
        else:
            sample_bytes = self.inputs[0][0:1].nbytes
            self.device_input = cuda.mem_alloc(sample_bytes * batch_size)

    def get_batch_size(self):
        return self.batch_size

    def get_batch(self, names):
        if self.current_index + self.batch_size > self.num_samples:
            return None
        end = self.current_index + self.batch_size
        if len(self.inputs) == 2:
            obs_batch = np.ascontiguousarray(self.inputs[0][self.current_index:end])
            cuda.memcpy_htod(self.device_input_obs, obs_batch)
            vec_batch = np.ascontiguousarray(self.inputs[1][self.current_index:end])
            cuda.memcpy_htod(self.device_input_vec, vec_batch)
            self.current_index = end
            return (int(self.device_input_obs), int(self.device_input_vec))
        else:
            input_batch = np.ascontiguousarray(self.inputs[0][self.current_index:end])
            cuda.memcpy_htod(self.device_input, input_batch)
            self.current_index = end
            return (int(self.device_input),)

    def read_calibration_cache(self):
        if os.path.exists(self.cache_file):
            with open(self.cache_file, "rb") as f:
                return f.read()

    def write_calibration_cache(self, cache):
        with open(self.cache_file, "wb") as f:
            f.write(cache)