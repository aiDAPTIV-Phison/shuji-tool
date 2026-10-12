import logging
import os
from contextlib import contextmanager
from functools import wraps
import inspect

from utils.logger import Logger

logger = Logger(name="TelemetryLogger", log_file="logs/telemetry.log")

_telemetry_active = False
_is_initialized = False
_tracer_provider = None
_logger_provider = None
_meter_provider = None
_global_tracer = None
_logging_handler = None

_SIGNAL_ENDPOINT_ENV = {
    "traces": "OTEL_EXPORTER_OTLP_TRACES_ENDPOINT",
    "metrics": "OTEL_EXPORTER_OTLP_METRICS_ENDPOINT",
    "logs": "OTEL_EXPORTER_OTLP_LOGS_ENDPOINT",
}


def _sdk_disabled() -> bool:
    """是否依 OTEL_SDK_DISABLED 關閉 SDK；未設定時視為關閉。

    Returns:
        True 表示不初始化 SDK。
    """
    value = os.getenv("OTEL_SDK_DISABLED", "true").strip().lower()
    return value not in ("false", "0", "no")


def _resolve_otlp_endpoint(signal: str) -> str | None:
    """依 OTEL 標準疊加解析某訊號的 OTLP endpoint。

    Args:
        signal: traces、metrics 或 logs。

    Returns:
        解析後的 endpoint；皆未設時為 None。
    """
    override = os.getenv(_SIGNAL_ENDPOINT_ENV[signal], "").strip()
    if override:
        return override
    shared = os.getenv("OTEL_EXPORTER_OTLP_ENDPOINT", "").strip()
    return shared or None


def _warn_unsupported_protocol():
    """protocol 非 grpc 時 warning，仍走 grpc exporter。"""
    protocol = os.getenv("OTEL_EXPORTER_OTLP_PROTOCOL", "grpc").strip().lower()
    if protocol != "grpc":
        logger.warning(
            f"OTEL_EXPORTER_OTLP_PROTOCOL={protocol} is not supported; using grpc."
        )


def _exporter_kwargs(signal: str) -> dict:
    """組 OTLP gRPC exporter 的 endpoint 參數。

    Args:
        signal: traces、metrics 或 logs。

    Returns:
        含 endpoint 的 kwargs；未解析到 endpoint 時為空 dict。
    """
    endpoint = _resolve_otlp_endpoint(signal)
    if endpoint:
        return {"endpoint": endpoint}
    return {}


def _new_span_exporter():
    """建立 OTLP span exporter；本服務只支援 grpc。

    Returns:
        OTLP gRPC SpanExporter。端點由 OTEL_EXPORTER_OTLP_ENDPOINT 決定。
    """
    from opentelemetry.exporter.otlp.proto.grpc.trace_exporter import OTLPSpanExporter

    return OTLPSpanExporter(**_exporter_kwargs("traces"))


def _new_log_exporter():
    """建立 OTLP log exporter；本服務只支援 grpc。

    Returns:
        OTLP gRPC LogExporter。
    """
    from opentelemetry.exporter.otlp.proto.grpc._log_exporter import OTLPLogExporter

    return OTLPLogExporter(**_exporter_kwargs("logs"))


def _new_metric_reader():
    """建立 OTLP metrics reader；本服務只支援 grpc。

    Returns:
        PeriodicExportingMetricReader 包裝的 OTLP gRPC metric exporter。
    """
    from opentelemetry.exporter.otlp.proto.grpc.metric_exporter import OTLPMetricExporter
    from opentelemetry.sdk.metrics.export import PeriodicExportingMetricReader

    return PeriodicExportingMetricReader(
        OTLPMetricExporter(**_exporter_kwargs("metrics"))
    )


def setup_opentelemetry():
    """依 OTEL_* 環境變數初始化 traces／logs／metrics；關閉時為 no-op。"""
    global _telemetry_active, _is_initialized, _tracer_provider, _logger_provider
    global _meter_provider, _global_tracer, _logging_handler
    if _sdk_disabled():
        _telemetry_active = False
        if _logging_handler is not None:
            logging.getLogger().removeHandler(_logging_handler)
            _logging_handler = None
        logger.info("Telemetry disabled (OTEL_SDK_DISABLED).")
        return
    if _is_initialized:
        _telemetry_active = True
        if _logging_handler is None and _logger_provider is not None:
            from opentelemetry.sdk._logs import LoggingHandler

            _logging_handler = LoggingHandler(
                level=logging.NOTSET, logger_provider=_logger_provider
            )
            logging.getLogger().addHandler(_logging_handler)
        return
    try:
        _warn_unsupported_protocol()
        from opentelemetry import metrics, trace
        from opentelemetry._logs import set_logger_provider
        from opentelemetry.metrics import set_meter_provider
        from opentelemetry.sdk._logs import LoggerProvider, LoggingHandler
        from opentelemetry.sdk._logs.export import BatchLogRecordProcessor
        from opentelemetry.sdk.metrics import MeterProvider
        from opentelemetry.sdk.resources import Resource
        from opentelemetry.sdk.trace import TracerProvider
        from opentelemetry.sdk.trace.export import BatchSpanProcessor

        resource = Resource.create()

        _tracer_provider = TracerProvider(resource=resource)
        _tracer_provider.add_span_processor(BatchSpanProcessor(_new_span_exporter()))
        trace.set_tracer_provider(_tracer_provider)
        _global_tracer = trace.get_tracer(__name__)

        _logger_provider = LoggerProvider(resource=resource)
        _logger_provider.add_log_record_processor(
            BatchLogRecordProcessor(_new_log_exporter())
        )
        set_logger_provider(_logger_provider)
        _logging_handler = LoggingHandler(
            level=logging.NOTSET, logger_provider=_logger_provider
        )
        root_logger = logging.getLogger()
        root_logger.addHandler(_logging_handler)
        if root_logger.level == logging.WARNING:
            root_logger.setLevel(logging.NOTSET)

        _meter_provider = MeterProvider(
            resource=resource, metric_readers=[_new_metric_reader()]
        )
        set_meter_provider(_meter_provider)

        _is_initialized = True
        _telemetry_active = True
        logger.info("Telemetry enabled: OpenTelemetry SDK initialized.")
    except ImportError as e:
        logger.warning(
            f"OTEL_SDK_DISABLED is false, but OpenTelemetry packages are not installed: {e}. "
            "Falling back to no-op stubs."
        )
        _telemetry_active = False


def shutdown_opentelemetry():
    """Flush 並關閉 traces／logs／metrics provider，避免 process 結束時丟掉未送出資料。"""
    global _telemetry_active, _logging_handler
    if _logging_handler is not None:
        logging.getLogger().removeHandler(_logging_handler)
        _logging_handler = None
    if _logger_provider is not None:
        _logger_provider.shutdown()
    if _meter_provider is not None:
        _meter_provider.shutdown()
    if _tracer_provider is not None:
        _tracer_provider.shutdown()
    _telemetry_active = False


def instrument_app(app):
    """為 FastAPI 掛自動打點（排除 /health、/metrics）。模型 process 通常不呼叫。

    Args:
        app: FastAPI 應用實例。
    """
    if not _telemetry_active:
        return
    from opentelemetry.instrumentation.fastapi import FastAPIInstrumentor

    FastAPIInstrumentor.instrument_app(app, excluded_urls="/health,/metrics")


def get_tracer():
    """取得全域 tracer；SDK 未啟用時為 None。

    Returns:
        OpenTelemetry Tracer 或 None。
    """
    return _global_tracer


def trace_attributes(**attributes):
    """裝飾器：在函式外層建立 span，並寫入 Function 與額外屬性。

    Args:
        **attributes: 寫入 span 的鍵值屬性。
    """
    def decorator(f):
        is_async = inspect.iscoroutinefunction(f)

        if is_async:
            @wraps(f)
            async def async_wrapper(*args, **kwargs):
                if not _telemetry_active or _global_tracer is None:
                    return await f(*args, **kwargs)
                with _global_tracer.start_as_current_span(f.__name__) as nested_span:
                    nested_span.set_attribute("Function", f.__name__)
                    for key, value in attributes.items():
                        nested_span.set_attribute(key, value)
                    return await f(*args, **kwargs)
            return async_wrapper

        @wraps(f)
        def sync_wrapper(*args, **kwargs):
            if not _telemetry_active or _global_tracer is None:
                return f(*args, **kwargs)
            with _global_tracer.start_as_current_span(f.__name__) as nested_span:
                nested_span.set_attribute("Function", f.__name__)
                for key, value in attributes.items():
                    nested_span.set_attribute(key, value)
                return f(*args, **kwargs)
        return sync_wrapper
    return decorator


@contextmanager
def trace_context(customfeature="", **attributes):
    """在現有 span 寫入屬性，或選擇性開啟具名子 span。

    Args:
        customfeature: 非空時以此名稱建立子 span。
        **attributes: 寫入 span 的鍵值屬性。
    """
    if not _telemetry_active or _global_tracer is None:
        yield
        return
    from opentelemetry import trace

    current_span = trace.get_current_span()
    if current_span and current_span.is_recording():
        for key, value in attributes.items():
            current_span.set_attribute(key, value)

    if customfeature != "":
        with _global_tracer.start_as_current_span(customfeature) as nested_span:
            nested_span.set_attribute("Function", customfeature)
            for key, value in attributes.items():
                nested_span.set_attribute(key, value)
            yield
    else:
        yield


def set_trace_attribute(**attributes):
    """若有正在 recording 的 span，則寫入屬性；否則忽略。

    Args:
        **attributes: 寫入 span 的鍵值屬性。
    """
    if not _telemetry_active:
        return
    from opentelemetry import trace

    current_span = trace.get_current_span()
    if current_span and current_span.is_recording():
        for key, value in attributes.items():
            current_span.set_attribute(key, value)


def set_trace_attribute_safe(**attributes):
    """寫入目前 span 屬性；若無 active span 則短暫開一個臨時 span。

    Args:
        **attributes: 寫入 span 的鍵值屬性；可用 operation 指定臨時 span 名稱。
    """
    if not _telemetry_active or _global_tracer is None:
        return
    from opentelemetry import trace

    current_span = trace.get_current_span()
    if current_span and current_span.is_recording():
        for key, value in attributes.items():
            current_span.set_attribute(key, value)
        return
    span_name = attributes.get("operation", "temp_attribute_span")
    with _global_tracer.start_as_current_span(span_name) as temp_span:
        temp_span.set_attribute("auto_created", True)
        for key, value in attributes.items():
            temp_span.set_attribute(key, value)


def capture_trace_context():
    """擷取當前 OpenTelemetry context，供跨 asyncio Task 傳遞。

    Returns:
        不透明的 context 物件；telemetry 關閉時為 None。
    """
    if not _telemetry_active:
        return None
    from opentelemetry import context as otel_context

    return otel_context.get_current()


def attach_trace_context(captured):
    """把先前擷取的 context 掛到目前執行緒。

    Args:
        captured: capture_trace_context 的回傳值。

    Returns:
        之後 detach 要用的 token；無需掛上時為 None。
    """
    if not _telemetry_active or captured is None:
        return None
    from opentelemetry import context as otel_context

    return otel_context.attach(captured)


def detach_trace_context(token):
    """解除 attach_trace_context 掛上的 context。

    Args:
        token: attach_trace_context 的回傳值。
    """
    if token is None:
        return
    from opentelemetry import context as otel_context

    otel_context.detach(token)
