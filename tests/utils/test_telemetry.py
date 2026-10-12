import logging
import os
from unittest.mock import patch

import pytest
from fastapi import HTTPException
from opentelemetry import metrics, trace


def _enable_shared_exporters(monkeypatch):
    """啟用 SDK 並重用 in-memory exporter（process 內 provider 只能設一次）。"""
    from opentelemetry.sdk._logs.export import InMemoryLogExporter
    from opentelemetry.sdk.metrics.export import InMemoryMetricReader
    from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter

    from utils.telemetry import setup_opentelemetry

    if not getattr(_enable_shared_exporters, "_initialized", False):
        span_exporter = InMemorySpanExporter()
        log_exporter = InMemoryLogExporter()
        metric_reader = InMemoryMetricReader()
        _enable_shared_exporters._span_exporter = span_exporter
        _enable_shared_exporters._log_exporter = log_exporter
        _enable_shared_exporters._metric_reader = metric_reader
        monkeypatch.setattr(
            "utils.telemetry._new_span_exporter",
            lambda: span_exporter,
        )
        monkeypatch.setattr(
            "utils.telemetry._new_log_exporter",
            lambda: log_exporter,
        )
        monkeypatch.setattr(
            "utils.telemetry._new_metric_reader",
            lambda: metric_reader,
        )
        _enable_shared_exporters._initialized = True

    monkeypatch.setenv("OTEL_SDK_DISABLED", "false")
    monkeypatch.setenv("OTEL_SERVICE_NAME", "voice-to-text-test")
    setup_opentelemetry()
    _enable_shared_exporters._span_exporter.clear()
    _enable_shared_exporters._log_exporter.clear()
    return (
        _enable_shared_exporters._span_exporter,
        _enable_shared_exporters._log_exporter,
        _enable_shared_exporters._metric_reader,
    )


def test_disabled_decorator_runs_function_without_span():
    """OTEL_SDK_DISABLED 預設關閉時，decorator 不得阻擋原函式。"""
    from opentelemetry.sdk._logs import LoggingHandler

    from utils.telemetry import (
        capture_trace_context,
        setup_opentelemetry,
        trace_attributes,
    )

    with patch.dict(os.environ, {"OTEL_SDK_DISABLED": "true"}, clear=False):
        setup_opentelemetry()

        @trace_attributes(Action="Test")
        def ping():
            return "ok"

        assert ping() == "ok"
        assert capture_trace_context() is None
        assert not any(
            isinstance(handler, LoggingHandler) for handler in logging.getLogger().handlers
        )


def test_enabled_decorator_records_function_and_action_attributes(monkeypatch):
    """啟用 SDK 時，decorator 應建立以函式名為名的 span，並寫入 Function／Action。"""
    from utils.telemetry import trace_attributes

    span_exporter, _, _ = _enable_shared_exporters(monkeypatch)

    @trace_attributes(Action="Split_Text_Into_Batches")
    def split_batches():
        return 3

    assert split_batches() == 3
    trace.get_tracer_provider().force_flush()

    spans = [span for span in span_exporter.get_finished_spans() if span.name == "split_batches"]
    assert len(spans) == 1
    assert spans[0].attributes["Function"] == "split_batches"
    assert spans[0].attributes["Action"] == "Split_Text_Into_Batches"


def test_capture_and_attach_share_the_same_trace(monkeypatch):
    """capture 後 attach，後續 span 應屬於同一條 trace。"""
    from utils.telemetry import (
        attach_trace_context,
        capture_trace_context,
        detach_trace_context,
    )

    span_exporter, _, _ = _enable_shared_exporters(monkeypatch)

    tracer = trace.get_tracer("test")
    with tracer.start_as_current_span("parent"):
        captured = capture_trace_context()
        parent_id = trace.get_current_span().get_span_context().trace_id

    token = attach_trace_context(captured)
    try:
        with tracer.start_as_current_span("queued_child"):
            child_id = trace.get_current_span().get_span_context().trace_id
    finally:
        detach_trace_context(token)
    trace.get_tracer_provider().force_flush()

    assert captured is not None
    assert child_id == parent_id
    names = {span.name for span in span_exporter.get_finished_spans()}
    assert "parent" in names
    assert "queued_child" in names


def test_resolve_otlp_endpoint_uses_shared_when_no_override(monkeypatch):
    """未設分項 endpoint 時，三個訊號都應使用共用 OTEL_EXPORTER_OTLP_ENDPOINT。"""
    from utils.telemetry import _resolve_otlp_endpoint

    monkeypatch.setenv("OTEL_EXPORTER_OTLP_ENDPOINT", "http://tempo:4317")
    monkeypatch.delenv("OTEL_EXPORTER_OTLP_TRACES_ENDPOINT", raising=False)
    monkeypatch.delenv("OTEL_EXPORTER_OTLP_METRICS_ENDPOINT", raising=False)
    monkeypatch.delenv("OTEL_EXPORTER_OTLP_LOGS_ENDPOINT", raising=False)

    assert _resolve_otlp_endpoint("traces") == "http://tempo:4317"
    assert _resolve_otlp_endpoint("metrics") == "http://tempo:4317"
    assert _resolve_otlp_endpoint("logs") == "http://tempo:4317"


def test_resolve_otlp_endpoint_per_signal_override(monkeypatch):
    """設了分項 endpoint 時，只有該訊號使用覆寫位址。"""
    from utils.telemetry import _resolve_otlp_endpoint

    monkeypatch.setenv("OTEL_EXPORTER_OTLP_ENDPOINT", "http://shared:4317")
    monkeypatch.setenv("OTEL_EXPORTER_OTLP_LOGS_ENDPOINT", "http://signoz:4317")

    assert _resolve_otlp_endpoint("traces") == "http://shared:4317"
    assert _resolve_otlp_endpoint("metrics") == "http://shared:4317"
    assert _resolve_otlp_endpoint("logs") == "http://signoz:4317"


def test_resolve_otlp_endpoint_empty_override_falls_back_to_shared(monkeypatch):
    """空字串分項覆寫不得蓋掉共用 endpoint。"""
    from utils.telemetry import _resolve_otlp_endpoint

    monkeypatch.setenv("OTEL_EXPORTER_OTLP_ENDPOINT", "http://tempo:4317")
    monkeypatch.setenv("OTEL_EXPORTER_OTLP_TRACES_ENDPOINT", "  ")

    assert _resolve_otlp_endpoint("traces") == "http://tempo:4317"


def test_enabled_standard_logging_exports_to_otlp(monkeypatch):
    """SDK 打開時，未經 LogHandler 的標準 logging 也應進 OTLP log exporter。"""
    _, log_exporter, _ = _enable_shared_exporters(monkeypatch)

    logging.getLogger("telemetry_test").info("plain logger message")
    from opentelemetry._logs import get_logger_provider

    get_logger_provider().force_flush()

    exported = log_exporter.get_finished_logs()
    assert any("plain logger message" in (record.log_record.body or "") for record in exported)


def test_enabled_meter_provider_force_flush(monkeypatch):
    """SDK 打開時 MeterProvider 應可 force_flush。"""
    _enable_shared_exporters(monkeypatch)

    metrics.get_meter_provider().force_flush()


def test_log_handler_exports_flow_id_and_call_site(monkeypatch):
    """SDK 打開時，LogHandler 產出的 log 應帶 flow_id 與 call_site attribute。"""
    from utils.LogHandler import LogHandler
    from utils.logger import Logger

    _, log_exporter, _ = _enable_shared_exporters(monkeypatch)
    test_logger = Logger(
        name="log_handler_test",
        log_file="/tmp/log_handler_test.log",
    )

    LogHandler.handle_info(
        test_logger,
        "flow-abc",
        "tests.utils.test_telemetry.test_log_handler_exports_flow_id_and_call_site",
        "handler info message",
    )
    from opentelemetry._logs import get_logger_provider

    get_logger_provider().force_flush()

    exported = log_exporter.get_finished_logs()
    matching = [
        record
        for record in exported
        if "handler info message" in (record.log_record.body or "")
    ]
    assert len(matching) == 1
    attrs = matching[0].log_record.attributes or {}
    assert attrs.get("flow_id") == "flow-abc"
    assert attrs.get("call_site") == (
        "tests.utils.test_telemetry.test_log_handler_exports_flow_id_and_call_site"
    )


def test_log_handler_error_exports_error_code(monkeypatch):
    """LogHandler 錯誤 log 應帶 error_code attribute，並仍 raise HTTPException。"""
    from utils.LogHandler import LogHandler
    from utils.logger import Logger

    _, log_exporter, _ = _enable_shared_exporters(monkeypatch)
    test_logger = Logger(
        name="log_handler_error_test",
        log_file="/tmp/log_handler_error_test.log",
    )

    with pytest.raises(HTTPException):
        raise LogHandler.handle_error(
            test_logger,
            "flow-err",
            "site.error",
            "something failed",
            "VTT-001",
        )
    from opentelemetry._logs import get_logger_provider

    get_logger_provider().force_flush()

    exported = log_exporter.get_finished_logs()
    matching = [
        record
        for record in exported
        if "something failed" in (record.log_record.body or "")
    ]
    assert len(matching) == 1
    attrs = matching[0].log_record.attributes or {}
    assert attrs.get("error_code") == "VTT-001"


def test_shutdown_opentelemetry_flushes_all_providers(monkeypatch):
    """lifespan stop 路徑應 flush／shutdown traces、logs、metrics。"""
    from utils.telemetry import shutdown_opentelemetry

    span_exporter, log_exporter, metric_reader = _enable_shared_exporters(monkeypatch)

    tracer = trace.get_tracer("shutdown_test")
    with tracer.start_as_current_span("before_shutdown"):
        pass
    logging.getLogger("shutdown_test").warning("before shutdown")

    trace.get_tracer_provider().force_flush()
    from opentelemetry._logs import get_logger_provider

    get_logger_provider().force_flush()
    metrics.get_meter_provider().force_flush()

    shutdown_opentelemetry()

    assert len(span_exporter.get_finished_spans()) >= 1
    assert any(
        "before shutdown" in (record.log_record.body or "")
        for record in log_exporter.get_finished_logs()
    )
    assert metric_reader.shutdown() is None


def test_log_to_api_not_importable_from_prosuite_utils():
    """公開介面不得再提供 log_to_api。"""
    with pytest.raises(ImportError):
        from utils.prosuite_utils import log_to_api  # noqa: F401


def test_get_call_site_info_importable_from_call_site_module():
    """call site helper 應改由非 Prosuite 模組提供。"""
    from utils.call_site import get_call_site_info

    site = get_call_site_info()
    assert "get_call_site_info" in site


def test_config_no_longer_reads_prosuite_log_vars():
    """設定層不再讀取已刪的 Prosuite log／舊 TRACE_ENDPOINT 變數。"""
    from utils.ConfigHandler import ConfigHandler

    assert not hasattr(ConfigHandler, "get_log_URL")
    assert not hasattr(ConfigHandler, "get_log_service_id")
    assert not hasattr(ConfigHandler, "get_log_service_ip")
    assert not hasattr(ConfigHandler, "get_trace_endpoint")


def test_chunk_decode_high_freq_logs_use_handle_debug():
    """RealtimeSession 點名的 partial chunk 解碼路徑應走 DEBUG。"""
    from pathlib import Path

    root = Path(__file__).resolve().parents[2]
    router_src = (root / "src/api/openai_router.py").read_text(encoding="utf-8")
    service_src = (root / "src/transcription/service.py").read_text(encoding="utf-8")

    assert "lh.handle_debug" in router_src
    assert "Received chunk decode request" in router_src
    assert router_src.index("lh.handle_debug") < router_src.index("Received chunk decode request")

    decode_start = service_src.index("async def decode_chunk")
    decode_end = service_src.index("\n    def _cleanse_segments", decode_start)
    decode_body = service_src[decode_start:decode_end]
    assert "lh.handle_info" not in decode_body
    for needle in (
        "Decoding audio chunk",
        "Detected language",
        "Chunk decode completed",
    ):
        assert needle in decode_body
        assert "lh.handle_debug" in decode_body[: decode_body.index(needle)]


def test_debug_log_handler_not_exported_at_info_level(monkeypatch):
    """預設 INFO 時，LogHandler DEBUG 不得進 OTLP exporter。"""
    from utils.LogHandler import LogHandler
    from utils.logger import Logger

    monkeypatch.setenv("LOG_LEVEL", "INFO")
    _, log_exporter, _ = _enable_shared_exporters(monkeypatch)
    test_logger = Logger(
        name="log_handler_debug_level_test",
        log_file="/tmp/log_handler_debug_level_test.log",
    )
    # 既有 Logger 單例可能已建過；強制對齊 INFO 以驗證級別契約
    test_logger.logger.setLevel(logging.INFO)

    LogHandler.handle_debug(
        test_logger,
        "flow-dbg",
        "site.debug",
        "debug only message should not export",
    )
    from opentelemetry._logs import get_logger_provider

    get_logger_provider().force_flush()

    exported = log_exporter.get_finished_logs()
    assert not any(
        "debug only message should not export" in (record.log_record.body or "")
        for record in exported
    )
