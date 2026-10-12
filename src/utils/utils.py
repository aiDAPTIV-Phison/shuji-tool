import ctypes
def is_cuda_available():
    try:
        ctypes.CDLL("libcuda.so")  # 尝试加载 CUDA 库
        return True
    except OSError:
        return False

device = "cuda" if is_cuda_available() else "cpu"
print(f"Device detected: {device}")
