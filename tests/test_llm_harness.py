"""Unit tests for the llm_benchmark harness pieces added for the
production-sample run: firm-block consecutive-origin sampling, the
raw-completion JSONL archive, and the OpenAI driver's pure functions.

No network, no API keys, no DuckDB — everything here is pure-Python/pandas.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "scripts" / "llm"))  # harness lives under scripts/llm

# The harness imports httpx at module level, which ships in the [llm] extra --
# without this guard a plain [dev] install aborts COLLECTION for the whole
# suite instead of skipping the harness tests.
pytest.importorskip("httpx", reason="LLM harness tests need the [llm] extra")

import llm_benchmark as lb  # noqa: E402


# ── synthetic origin pool ─────────────────────────────────────────────────────

def _quarter_run(gvkey: str, start: str, n: int) -> pd.DataFrame:
    """One unbroken run of n consecutive quarter-end origins."""
    dates = pd.date_range(start, periods=n, freq="QE")
    return pd.DataFrame({
        "gvkey": gvkey,
        "q0_date": dates,
        "max_horizon": 20,
    })


@pytest.fixture
def pool() -> pd.DataFrame:
    """30 firms x 12-quarter run, plus one firm with two short runs split by
    a >120-day gap, plus one firm too short to host any block."""
    parts = [_quarter_run(f"{i:06d}", "2012-03-31", 12) for i in range(1, 31)]
    # firm 000031: two runs of 5 and 6 quarters, 2 years apart
    parts.append(_quarter_run("000031", "2011-03-31", 5))
    parts.append(_quarter_run("000031", "2015-03-31", 6))
    # firm 000032: only 2 eligible origins — can't host a 4-block
    parts.append(_quarter_run("000032", "2013-03-31", 2))
    return pd.concat(parts, ignore_index=True).sample(frac=1.0, random_state=0)


# ── firm-block sampling ───────────────────────────────────────────────────────

def test_firm_block_shape_and_uniqueness(pool):
    out = lb.firm_block_sample_origins(pool, n_firms=10, block_len=4, seed=7)
    assert len(out) == 40
    assert out["gvkey"].nunique() == 10
    assert out.groupby("gvkey").size().eq(4).all()


def test_firm_block_consecutive_quarters(pool):
    out = lb.firm_block_sample_origins(pool, n_firms=20, block_len=4, seed=7)
    for _, grp in out.groupby("gvkey"):
        diffs = grp["q0_date"].sort_values().diff().dropna().dt.days
        assert (diffs <= lb.MAX_GAP_DAYS).all(), \
            "block spans a coverage gap — revisions would be non-adjacent"


def test_firm_block_deterministic(pool):
    a = lb.firm_block_sample_origins(pool, n_firms=10, block_len=4, seed=7)
    b = lb.firm_block_sample_origins(pool, n_firms=10, block_len=4, seed=7)
    pd.testing.assert_frame_equal(a.reset_index(drop=True),
                                  b.reset_index(drop=True))
    c = lb.firm_block_sample_origins(pool, n_firms=10, block_len=4, seed=8)
    assert not a.equals(c)


def test_firm_block_excludes_short_firms(pool):
    out = lb.firm_block_sample_origins(pool, n_firms=100, block_len=4, seed=7)
    assert "000032" not in set(out["gvkey"])  # only 2 eligible origins
    # firm 000031's 5q run CAN host a 4-block; if selected, the block must
    # not bridge its two runs (handled by the consecutive test above).


def test_firm_block_block_len_too_large(pool):
    with pytest.raises(ValueError):
        lb.firm_block_sample_origins(pool, n_firms=5, block_len=13, seed=7)


def test_origins_csv_path_modes_distinct():
    iid = lb.origins_csv_path(False, 42, 500)
    blk = lb.origins_csv_path(False, 42, 500, sample_mode="firm_block",
                              n_firms=1000, block_len=4)
    assert iid != blk
    assert "blk4" in blk.name and "f1000" in blk.name
    # iid filename uses the arm-agnostic shared-origins scope tag
    assert iid.name == "origins__shared__seed42__n500.csv"


# ── raw-response archive ──────────────────────────────────────────────────────

def test_raw_response_log_roundtrip(tmp_path):
    p = tmp_path / "responses.jsonl"
    log = lb.RawResponseLog(p, model_name="claude-sonnet-4-6",
                            prompt_version="unstructured", system_prompt="SYSTEM",
                            resume=False, log_prompts=False)
    log.log(1004, "2015-03-31", "Q1: revtq=1.0", {"input_tokens": 10},
            user_msg="USER MSG", thinking="I reasoned about it")
    log.log(1004, "2015-06-30", None, None, error="boom", user_msg="USER MSG")
    log.close()

    recs = [json.loads(line) for line in p.read_text(encoding="utf-8").splitlines()]
    assert len(recs) == 2
    ok, err = recs
    assert ok["gvkey"] == "001004"
    assert ok["completion"] == "Q1: revtq=1.0"
    assert ok["thinking"] == "I reasoned about it"
    assert ok["usage"] == {"input_tokens": 10}
    assert ok["error"] is None
    assert ok["system_prompt_sha"] == lb._sha256("SYSTEM")
    assert ok["user_msg_sha"] == lb._sha256("USER MSG")
    assert "user_msg" not in ok  # log_prompts=False stores only the sha
    assert err["completion"] is None and err["error"] == "boom"
    assert err["thinking"] is None  # default when no reasoning passed


def test_raw_response_log_prompts_flag(tmp_path):
    p = tmp_path / "responses.jsonl"
    log = lb.RawResponseLog(p, "m", "unstructured", "S", resume=False, log_prompts=True)
    log.log(1, "2015-03-31", "text", None, user_msg="THE PROMPT")
    log.close()
    rec = json.loads(p.read_text(encoding="utf-8").splitlines()[0])
    assert rec["user_msg"] == "THE PROMPT"


def test_raw_response_log_resume_appends(tmp_path):
    p = tmp_path / "responses.jsonl"
    log = lb.RawResponseLog(p, "m", "unstructured", "S", resume=False)
    log.log(1, "2015-03-31", "first", None)
    log.close()
    log = lb.RawResponseLog(p, "m", "unstructured", "S", resume=True)
    log.log(2, "2015-06-30", "second", None)
    log.close()
    lines = p.read_text(encoding="utf-8").splitlines()
    assert len(lines) == 2
    # resume=False truncates
    log = lb.RawResponseLog(p, "m", "unstructured", "S", resume=False)
    log.close()
    assert p.read_text(encoding="utf-8") == ""


# ── OpenAI driver pure functions ──────────────────────────────────────────────

def test_openai_batch_line_shape():
    line = lb.build_openai_batch_line("001004_20150331", "SYS", "USER",
                                      "gpt-5.5", max_tokens=123)
    assert line["custom_id"] == "001004_20150331"
    assert line["method"] == "POST"
    assert line["url"] == "/v1/chat/completions"
    body = line["body"]
    assert body["model"] == "gpt-5.5"
    assert body["max_completion_tokens"] == 123
    assert "max_tokens" not in body          # rejected by reasoning models
    assert "temperature" not in body         # ditto
    assert body["messages"][0] == {"role": "system", "content": "SYS"}
    assert body["messages"][1] == {"role": "user", "content": "USER"}


def test_normalize_openai_usage_subtracts_cached():
    u = lb._normalize_openai_usage({
        "prompt_tokens": 1000,
        "completion_tokens": 200,
        "prompt_tokens_details": {"cached_tokens": 400},
    })
    # OpenAI's prompt_tokens INCLUDES cached tokens; the report's identity
    # total_input = input + cache_read + cache_write must still hold.
    assert u["input_tokens"] == 600
    assert u["cache_read_input_tokens"] == 400
    assert u["output_tokens"] == 200
    assert u["cache_creation_input_tokens"] == 0


def test_thinking_params_request_summarized_display():
    # display MUST be "summarized" — the adaptive-only prod models (Sonnet 5,
    # Opus 4.8) default thinking.display to "omitted" (empty thinking field), so
    # without this the reasoning-capture archive would be blank.
    for model in ("claude-opus-4-8", "claude-sonnet-5", "claude-opus-4-7"):
        tp = lb._anthropic_thinking_params(model)
        assert tp["thinking"]["type"] == "adaptive"
        assert tp["thinking"]["display"] == "summarized", model
    # Budget-token models (Sonnet 4.6 / pre-4.6) also carry the explicit display.
    for model in ("claude-sonnet-4-6", "claude-haiku-4-5"):
        tp = lb._anthropic_thinking_params(model)
        assert tp["thinking"]["type"] == "enabled"
        assert tp["thinking"]["display"] == "summarized", model


def test_parse_openai_batch_result_success():
    obj = {
        "custom_id": "001004_20150331",
        "response": {
            "status_code": 200,
            "body": {
                "choices": [{"message": {"content": "Q1: revtq=1.0"}}],
                "usage": {"prompt_tokens": 10, "completion_tokens": 5},
            },
        },
    }
    cid, text, usage, err, thinking = lb._parse_openai_batch_result_line(obj)
    assert cid == "001004_20150331"
    assert text == "Q1: revtq=1.0"
    assert usage["input_tokens"] == 10 and usage["output_tokens"] == 5
    assert err is None
    assert thinking is None  # chat/completions returns no reasoning summary


def test_parse_openai_batch_result_reasoning_when_present():
    # If a gateway surfaces reasoning on the message, keep it.
    obj = {
        "custom_id": "c",
        "response": {"status_code": 200, "body": {
            "choices": [{"message": {"content": "answer",
                                     "reasoning_content": "because X"}}],
            "usage": {"prompt_tokens": 1, "completion_tokens": 1}}},
    }
    cid, text, usage, err, thinking = lb._parse_openai_batch_result_line(obj)
    assert text == "answer" and thinking == "because X"


def test_parse_openai_batch_result_errors():
    cid, text, usage, err, thinking = lb._parse_openai_batch_result_line({
        "custom_id": "x", "error": {"message": "rate limited"},
    })
    assert text is None and "rate limited" in err and thinking is None

    cid, text, usage, err, thinking = lb._parse_openai_batch_result_line({
        "custom_id": "y",
        "response": {"status_code": 400,
                     "body": {"error": {"message": "bad request"}}},
    })
    assert text is None and "400" in err and "bad request" in err

    cid, text, usage, err, thinking = lb._parse_openai_batch_result_line({
        "custom_id": "z",
        "response": {"status_code": 200,
                     "body": {"choices": [{"message": {"content": ""}}],
                              "usage": {}}},
    })
    assert text is None and "no text" in err


def test_parse_anthropic_batch_result_captures_thinking():
    # Extended/adaptive thinking: content leads with thinking block(s), then
    # the text block. The parser returns the text as the completion and the
    # concatenated reasoning as the 5th element.
    obj = {
        "custom_id": "001004_20150331",
        "result": {
            "type": "succeeded",
            "message": {
                "content": [
                    {"type": "thinking", "thinking": "step one"},
                    {"type": "thinking", "thinking": "step two"},
                    {"type": "text", "text": "Q1: revtq=1.0"},
                ],
                "usage": {"input_tokens": 100, "output_tokens": 20},
            },
        },
    }
    cid, text, usage, err, thinking = lb._parse_batch_result_line(obj)
    assert cid == "001004_20150331"
    assert text == "Q1: revtq=1.0"
    assert thinking == "step one\nstep two"
    assert usage["input_tokens"] == 100 and usage["output_tokens"] == 20
    assert err is None


def test_parse_anthropic_batch_result_no_thinking():
    obj = {
        "custom_id": "c",
        "result": {"type": "succeeded", "message": {
            "content": [{"type": "text", "text": "hi"}],
            "usage": {"input_tokens": 1, "output_tokens": 1}}},
    }
    cid, text, usage, err, thinking = lb._parse_batch_result_line(obj)
    assert text == "hi" and thinking is None

    # A failed result carries no thinking either.
    cid, text, usage, err, thinking = lb._parse_batch_result_line({
        "custom_id": "d",
        "result": {"type": "errored", "error": {"message": "boom"}}})
    assert text is None and "boom" in err and thinking is None


# ── origins fingerprint (shared-sample guard) ─────────────────────────────────

def test_origins_fingerprint_order_and_dtype_invariant():
    a = pd.DataFrame({
        "gvkey": ["001004", "002005"],
        "q0_date": pd.to_datetime(["2015-03-31", "2016-06-30"]),
        "max_horizon": [8, 12],
    })
    # Reversed row order → same fingerprint (identity is the set, not the order).
    b = a.iloc[::-1].reset_index(drop=True)
    assert lb.origins_fingerprint(a) == lb.origins_fingerprint(b)
    # Unpadded int gvkeys + string dates → still the same (normalization).
    c = pd.DataFrame({
        "gvkey": [1004, 2005],
        "q0_date": ["2015-03-31", "2016-06-30"],
        "max_horizon": [8, 12],
    })
    assert lb.origins_fingerprint(c) == lb.origins_fingerprint(a)
    # Extra unrelated columns are ignored (only the identity triple matters).
    d = a.assign(scale=[1.0, 2.0])
    assert lb.origins_fingerprint(d) == lb.origins_fingerprint(a)


def test_origins_fingerprint_sensitive_to_any_change():
    a = pd.DataFrame({"gvkey": ["001004"],
                      "q0_date": pd.to_datetime(["2015-03-31"]),
                      "max_horizon": [8]})
    diff_date = a.assign(q0_date=pd.to_datetime(["2015-06-30"]))
    diff_horizon = a.assign(max_horizon=[12])
    diff_firm = a.assign(gvkey=["001005"])
    base = lb.origins_fingerprint(a)
    assert base != lb.origins_fingerprint(diff_date)
    assert base != lb.origins_fingerprint(diff_horizon)
    assert base != lb.origins_fingerprint(diff_firm)
    # A dropped origin changes the fingerprint too.
    two = pd.concat([a, diff_firm], ignore_index=True)
    assert lb.origins_fingerprint(two) != base


# ── unstructured response parser hardening (launch-review fixes) ────────────────────────

def test_parse_response_normalizes_zero_padded_labels():
    # "Q01:" must land on the same key finish_origin looks up ("Q1") —
    # verbatim storage would miss every lookup while still passing the
    # parse-coverage gate, zero-scoring the whole origin silently.
    out = lb.parse_response("Q01: revtq=1.5, cogsq=2.0\nQ10: revtq=3.0")
    assert set(out) == {"Q1", "Q10"}
    assert out["Q1"]["revtq"] == 1.5
    assert out["Q10"]["revtq"] == 3.0


def test_parse_response_drops_thousands_separated_values():
    # "revtq=1,234.5" splits on "," into "revtq=1" + "234.5" — keeping the
    # truncated leading digits is a silently WRONG value. The corrupted key
    # must be dropped (masked gap), the neighbours kept.
    out = lb.parse_response("Q1: revtq=1,234.5, cogsq=3.0")
    assert "revtq" not in out["Q1"]
    assert out["Q1"]["cogsq"] == 3.0
    # Multiple separators in one value.
    out2 = lb.parse_response("Q1: revtq=1,234,567.8, cogsq=3.0")
    assert "revtq" not in out2["Q1"]
    assert out2["Q1"]["cogsq"] == 3.0
    # Thousands-separated AND scientific notation (PR #194 review edge case):
    # the "234e5" tail must still be recognized as a number fragment so the
    # truncated "1.0" is dropped.
    out3 = lb.parse_response("Q1: revtq=1,234e5, cogsq=3.0")
    assert "revtq" not in out3["Q1"]
    assert out3["Q1"]["cogsq"] == 3.0
    out4 = lb.parse_response("Q1: revtq=1,234.5E-2, cogsq=3.0")
    assert "revtq" not in out4["Q1"]
    assert out4["Q1"]["cogsq"] == 3.0


def test_parse_response_basics_unchanged():
    out = lb.parse_response(
        "preamble text\nQ1: revtq=-2.5, cogsq=1e3, dpq=abc\nnot a line")
    assert out["Q1"]["revtq"] == -2.5
    assert out["Q1"]["cogsq"] == 1000.0
    assert "dpq" not in out["Q1"]  # unparseable value -> gap, not crash


# ── finish_origin: unemitted horizons stay unscored ──────────────────────────

def _mini_reg() -> "lb.RegStatLookup":
    rows = [{"feature": feat, "quarter": pd.Timestamp("2015-03-31"),
             "k": 1.0, "mu": 0.0, "sigma": 1.0}
            for feat in ("revtq", "cogsq", "gpq")]
    return lb.RegStatLookup(pd.DataFrame(rows))


def test_finish_origin_skips_unemitted_horizons():
    # A partial response passing the >=50% parse gate must leave the missing
    # horizons UNSCORED. Pre-fix, derive_all({}) fabricated all-zero derived
    # aggregates (gpq=niq=atq=...=0.0) that were scored as real predictions.
    dates = pd.date_range("2015-03-31", periods=5, freq="QE")
    window = pd.DataFrame({
        "datadate": dates,
        "revtq": [10.0, 11.0, 12.0, 13.0, 14.0],
        "cogsq": [4.0, 4.5, 5.0, 5.5, 6.0],
    })
    meta = {
        "active_list":    ["revtq", "cogsq"],
        "max_horizon":    4,
        "prompt_version": "unstructured",
        "window":         window,
        "q0_idx":         0,
        "q0_qend":        pd.Timestamp("2015-03-31"),
        "q0_row":         window.iloc[0],
        "scale_q0":       1.0,
        "gvkey":          1004,
    }
    text = "Q1: revtq=10.5, cogsq=4.2\nQ2: revtq=11.5, cogsq=4.7"
    rows, adh = lb.finish_origin(meta, text, _mini_reg())
    assert rows, "emitted horizons must still be scored"
    horizons = {r["forecast_horizon"] for r in rows}
    assert horizons == {1, 2}, (
        f"unemitted horizons must stay unscored (masked), got {sorted(horizons)}")
    assert all(np.isfinite(r["prediction"]) for r in rows)
    # Adherence still reports the shortfall: 2 primitives x 2 emitted horizons
    # passed, out of 2 x 4 asked.
    assert adh["n_pass"] == 4 and adh["n_total"] == 8


# ── batch result fetch: dedupe across mid-body stream restarts ────────────────

class _FakeResp:
    def __init__(self, lines, fail_after=None):
        self.status_code = 200
        self._lines = lines
        self._fail_after = fail_after

    def iter_lines(self):
        for i, ln in enumerate(self._lines):
            if self._fail_after is not None and i == self._fail_after:
                raise lb.httpx.ReadError("mid-body connection drop")
            yield ln


class _FakeStreamCM:
    def __init__(self, resp):
        self._r = resp

    def __enter__(self):
        return self._r

    def __exit__(self, *exc):
        return False


def test_iter_batch_results_dedupes_across_stream_restart(monkeypatch):
    # A mid-body network error re-opens the (non-resumable) output stream from
    # line 1; without custom_id dedupe every already-fetched origin would be
    # yielded — and written — twice.
    def _line(cid):
        return json.dumps({
            "custom_id": cid,
            "result": {"type": "succeeded", "message": {
                "content": [{"type": "text", "text": "Q1: revtq=1.0"}],
                "usage": {"input_tokens": 1, "output_tokens": 1}}},
        })

    lines = [_line("a"), _line("b"), _line("c")]
    calls = {"n": 0}

    def fake_stream(method, url, headers=None, timeout=None):
        calls["n"] += 1
        if calls["n"] == 1:
            # Yields a, b then dies mid-body.
            return _FakeStreamCM(_FakeResp(lines, fail_after=2))
        return _FakeStreamCM(_FakeResp(lines))  # full replay from line 1

    monkeypatch.setattr(lb, "_portkey_batch_headers",
                        lambda: {"Content-Type": "application/json"})
    monkeypatch.setattr(lb.httpx, "stream", fake_stream)
    monkeypatch.setattr(lb.time, "sleep", lambda s: None)

    got = [cid for cid, *_ in lb.iter_batch_results("batch_x", retry_secs=0)]
    assert got == ["a", "b", "c"], "restart must not re-yield a/b"
    assert calls["n"] == 2
