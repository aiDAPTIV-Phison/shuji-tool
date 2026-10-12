"""OpenAI 相容端點的錯誤信封。

openai-python SDK 解析錯誤時會找 body 的 `error.message`，因此 /v1 路徑必須回
`{"error": {...}}`。現有 /audio/* 端點沿用 LogHandler 的
`{"detail": {"message", "errorCode"}}` 契約，兩者互不影響。
"""

from typing import Optional

from fastapi import Request
from fastapi.exception_handlers import request_validation_exception_handler
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse

OPENAI_ROUTE_PREFIX = "/v1/"


def _error_type_for(status_code: int) -> str:
    """依 HTTP 狀態碼決定 OpenAI 錯誤型別。

    Args:
        status_code: HTTP 狀態碼。

    Returns:
        4xx 為 invalid_request_error，其餘為 server_error。
    """
    return "invalid_request_error" if 400 <= status_code < 500 else "server_error"


class OpenAICompatError(Exception):
    """以 OpenAI 錯誤信封回應的例外。

    僅供 /v1 路徑使用，避免影響既有端點的錯誤格式。
    """

    def __init__(
        self,
        status_code: int,
        message: str,
        code: Optional[str] = None,
        param: Optional[str] = None,
        error_type: Optional[str] = None,
    ):
        """建立 OpenAI 相容錯誤。

        Args:
            status_code: HTTP 狀態碼。
            message: 錯誤描述。
            code: 錯誤代碼，沿用既有 MSS-VTT-xxxxxx 以保留追溯性。
            param: 造成錯誤的請求欄位名稱。
            error_type: OpenAI 錯誤型別；省略時依狀態碼推導。
        """
        super().__init__(message)
        self.status_code = status_code
        self.message = message
        self.code = code
        self.param = param
        self.error_type = error_type or _error_type_for(status_code)

    def to_payload(self) -> dict:
        """組出 OpenAI 錯誤信封。

        Returns:
            含單一 error 物件的 dict。
        """
        return {
            "error": {
                "message": self.message,
                "type": self.error_type,
                "param": self.param,
                "code": self.code,
            }
        }


async def openai_error_handler(request: Request, exc: Exception) -> JSONResponse:
    """將 OpenAICompatError 轉為 OpenAI 錯誤信封回應。

    Args:
        request: 觸發錯誤的請求，未使用。
        exc: 待轉換的 OpenAICompatError。

    Returns:
        帶 error 物件的 JSONResponse。
    """
    del request
    assert isinstance(exc, OpenAICompatError)
    return JSONResponse(status_code=exc.status_code, content=exc.to_payload())


async def openai_validation_error_handler(request: Request, exc: Exception):
    """讓 /v1 的參數驗證錯誤也回 OpenAI 錯誤信封。

    非 /v1 路徑委派回 FastAPI 預設處理，維持既有端點的 422 回應格式不變。

    Args:
        request: 觸發錯誤的請求。
        exc: FastAPI 的 RequestValidationError。

    Returns:
        /v1 路徑回 400 error 信封；其餘回 FastAPI 預設 422。
    """
    assert isinstance(exc, RequestValidationError)
    if not request.url.path.startswith(OPENAI_ROUTE_PREFIX):
        return await request_validation_exception_handler(request, exc)

    first_error = exc.errors()[0] if exc.errors() else {}
    location = [str(item) for item in first_error.get("loc", []) if str(item) != "body"]
    param = ".".join(location) or None
    message = first_error.get("msg", "Invalid request.")
    return JSONResponse(
        status_code=400,
        content=OpenAICompatError(
            status_code=400,
            message=f"{message} (param: {param})" if param else message,
            code="MSS-VTT-000101",
            param=param,
        ).to_payload(),
    )
