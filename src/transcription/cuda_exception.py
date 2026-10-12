from __future__ import annotations

def cuda_exception_classifier(e: Exception) -> CudaException | Exception:
    """
    把 Exception 分類為 CUDAException 或 Exception，如果非 CUDA 相關錯誤，則返回原 Exception
    """
    msg = str(e).lower()
    if "cuda" in msg and ("out of memory" in msg or "oom" in msg):
        return CudaOutOfMemoryError(str(e))
    elif "cuda" in msg:
        return CudaException(str(e))
    return e

class CudaException(Exception):
    """
    CUDA 錯誤
    """
    def __init__(self, message: str):
        self.message = message
        super().__init__(self.message)

class CudaOutOfMemoryError(CudaException):
    """
    CUDA 記憶體不足的錯誤
    """
    def __init__(self, message: str):
        super().__init__(message)