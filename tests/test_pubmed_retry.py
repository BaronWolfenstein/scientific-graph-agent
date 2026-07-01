"""Retry/backoff for PubMed E-utilities (NCBI returns HTTP 429 under rapid calls,
which skipped whole query families in the GEPA harvest)."""
import urllib.error
import pytest


class _FakeResp:
    def __enter__(self): return self
    def __exit__(self, *a): return False
    def read(self): return b"OK"


def test_retries_on_429_then_succeeds(monkeypatch):
    from agent_graph import tools
    calls = {"n": 0}

    def fake_urlopen(url, timeout=None):
        calls["n"] += 1
        if calls["n"] < 3:
            raise urllib.error.HTTPError(url, 429, "Too Many Requests", None, None)
        return _FakeResp()

    monkeypatch.setattr(tools.urllib.request, "urlopen", fake_urlopen)
    monkeypatch.setattr(tools.time, "sleep", lambda s: None)  # no real backoff in tests

    with tools._urlopen_retry("http://x", retries=4, backoff=0.01) as r:
        assert r.read() == b"OK"
    assert calls["n"] == 3  # failed twice, succeeded on the third


def test_raises_after_exhausting_retries(monkeypatch):
    from agent_graph import tools

    def always_429(url, timeout=None):
        raise urllib.error.HTTPError(url, 429, "Too Many Requests", None, None)

    monkeypatch.setattr(tools.urllib.request, "urlopen", always_429)
    monkeypatch.setattr(tools.time, "sleep", lambda s: None)

    with pytest.raises(urllib.error.HTTPError):
        tools._urlopen_retry("http://x", retries=3, backoff=0.01)


def test_non_rate_limit_error_not_retried(monkeypatch):
    from agent_graph import tools
    calls = {"n": 0}

    def fake_404(url, timeout=None):
        calls["n"] += 1
        raise urllib.error.HTTPError(url, 404, "Not Found", None, None)

    monkeypatch.setattr(tools.urllib.request, "urlopen", fake_404)
    monkeypatch.setattr(tools.time, "sleep", lambda s: None)

    with pytest.raises(urllib.error.HTTPError):
        tools._urlopen_retry("http://x", retries=4, backoff=0.01)
    assert calls["n"] == 1  # 404 is not retried
