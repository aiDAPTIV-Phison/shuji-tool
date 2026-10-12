# api/app.py
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.exceptions import RequestValidationError
from api.openai_router import router as openai_router
from api.openai_errors import (
    OpenAICompatError,
    openai_error_handler,
    openai_validation_error_handler,
)
from transcription.engine_loader import load
from utils.logger import Logger
from utils.call_site import get_call_site_info
from utils.LogHandler import LogHandler as lh
from utils.ConfigHandler import ConfigHandler
from utils.telemetry import setup_opentelemetry, shutdown_opentelemetry

setup_opentelemetry()

app_port = ConfigHandler.get_app_port()
app_workers = ConfigHandler.get_app_workers()
root_path = ConfigHandler.get_uvicorn_root_path()

# 啟動時載入引擎；失敗則讓應用啟動失敗（不吞例外）
@asynccontextmanager
async def lifespan(target_app: FastAPI):
    load()
    try:
        yield
    finally:
        shutdown_opentelemetry()

app = FastAPI(lifespan=lifespan)
callSite = get_call_site_info()

# 初始化 Logger
logger = Logger(name="ApplicationLogger", log_file="logs/app.log")
lh.handle_info(logger, "VTT-start", callSite,f"Application starting...")

# 註冊路由
app.include_router(openai_router, prefix="/v1", tags=["OpenAI Compatible"])

# /v1 使用 OpenAI 錯誤信封
app.add_exception_handler(OpenAICompatError, openai_error_handler)
app.add_exception_handler(RequestValidationError, openai_validation_error_handler)

# /health endpoint
@app.get("/health")
async def health_check():
    """Health check endpoint for Kubernetes liveness and readiness probes."""
    return {"status": "healthy"}

# /metrics endpoint
@app.get("/metrics")
async def get_metrics():
    """Metrics endpoint for fake response, used by dispacher service?"""
    return {"status": "healthy"}

lh.handle_info(logger, "VTT-run", callSite, f"Routes registered successfully.")
def main():
    """
    主程式入口，使用 uvicorn 啟動 FastAPI 應用
    """
    from multiprocessing import freeze_support
    import uvicorn

    # 對 Windows 平台做必要的初始化處理
    freeze_support()

    uvicorn.run(
        "app:app",  
        host="0.0.0.0",
        port=app_port,
        workers=app_workers,
        log_level="info",
        root_path=root_path,
    )

if __name__ == "__main__":
    main()
