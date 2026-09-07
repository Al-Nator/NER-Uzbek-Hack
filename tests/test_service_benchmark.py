"""Проверки тайминга открытого входного потока без GPU и HTTP сервера."""

from scripts import benchmark_service as benchmark


def test_open_loop_latency_includes_dispatch_lag(monkeypatch):
    """Включает задержку диспетчеризации в полный open-loop тайминг."""
    ticks = iter([10.0, 12.0, 15.0])
    sleeps = []
    monkeypatch.setattr(benchmark, "perf_counter", lambda: next(ticks))
    monkeypatch.setattr(benchmark, "sleep", sleeps.append)
    monkeypatch.setattr(
        benchmark, "request", lambda client, docs: benchmark.ResponseSample(3.0, ())
    )
    result = benchmark.request_scheduled(None, (), 11.0)
    assert sleeps == [1.0]
    assert result.seconds == 4.0
    assert result.dispatch_lag_seconds == 1.0


def test_closed_loop_sample_has_no_synthetic_queue_time():
    """Не добавляет искусственную очередь к обычному HTTP-измерению."""
    result = benchmark.ResponseSample(0.02, ())
    assert result.dispatch_lag_seconds == 0.0
