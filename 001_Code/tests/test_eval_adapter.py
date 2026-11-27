# the standalone adapter eval must skip finished work, complete the summary,
# never double-bill, and still bill an interrupted session
import json
import os

import pytest

from config import RunConfig
from scripts.eval_adapter import run_adapter_eval

EV = {"per_tier_acc": {"easy": 0.5}, "n_items": 4}
LP = {"per_tier_acc": {"easy": 0.6}, "n_items": 4}


def _setup(tmp_path, with_summary=True):
    out = str(tmp_path / "results" / "007_p2_TAG_seed0")
    os.makedirs(out)
    if with_summary:
        json.dump({"sft_acc": None, "sft_logprob_acc": None}, open(os.path.join(out, "summary.json"), "w"))
    return out


def test_runs_and_completes_summary(tmp_path):
    out = _setup(tmp_path)
    rows = []
    ev, lp = run_adapter_eval(RunConfig(), "adapter", "sft", out,
                              gen_eval=lambda: EV, probe_eval=lambda: LP,
                              append_row=rows.append)
    sm = json.load(open(os.path.join(out, "summary.json")))
    assert sm["sft_acc"] == {"easy": 0.5} and sm["sft_logprob_acc"] == {"easy": 0.6}
    assert len(rows) == 1 and rows[0]["completed"] is True
    assert os.path.exists(os.path.join(out, "sft_eval_manifest.json"))


def test_skips_finished_and_never_double_bills(tmp_path):
    out = _setup(tmp_path)
    json.dump(EV, open(os.path.join(out, "sft_eval.json"), "w"))
    json.dump(LP, open(os.path.join(out, "sft_logprob.json"), "w"))
    rows = []
    calls = []
    run_adapter_eval(RunConfig(), "adapter", "sft", out,
                     gen_eval=lambda: calls.append("gen") or EV,
                     probe_eval=lambda: calls.append("probe") or LP,
                     append_row=rows.append)
    assert calls == []          # nothing recomputed
    assert rows == []           # nothing billed twice


def test_interrupted_session_still_bills(tmp_path):
    out = _setup(tmp_path, with_summary=False)
    rows = []

    def dying_probe():
        raise KeyboardInterrupt

    with pytest.raises(KeyboardInterrupt):
        run_adapter_eval(RunConfig(), "adapter", "sft", out,
                         gen_eval=lambda: EV, probe_eval=dying_probe,
                         append_row=rows.append)
    assert len(rows) == 1
    assert rows[0]["completed"] is False
    assert "sft_eval" in rows[0]["timings_sec"]   # the gen eval that DID run is billed
    # and the finished gen eval is on disk, so the rerun will skip it
    assert os.path.exists(os.path.join(out, "sft_eval.json"))


def test_interrupted_during_gen_still_bills_active_time(tmp_path, monkeypatch):
    out = _setup(tmp_path, with_summary=False)
    rows = []
    clock = iter([100.0, 102.5])
    monkeypatch.setattr("scripts.eval_adapter.time.time", lambda: next(clock))

    def dying_gen():
        raise KeyboardInterrupt

    with pytest.raises(KeyboardInterrupt):
        run_adapter_eval(RunConfig(), "adapter", "sft", out,
                         gen_eval=dying_gen, probe_eval=lambda: LP,
                         append_row=rows.append)
    assert len(rows) == 1 and rows[0]["completed"] is False
    assert rows[0]["timings_sec"]["sft_eval"] == 2.5
    assert not os.path.exists(os.path.join(out, "sft_eval.json"))


def test_resume_after_interrupt_finishes_cheap(tmp_path):
    out = _setup(tmp_path)
    json.dump(EV, open(os.path.join(out, "sft_eval.json"), "w"))   # as left by the crash
    rows = []
    ev, lp = run_adapter_eval(RunConfig(), "adapter", "sft", out,
                              gen_eval=lambda: (_ for _ in ()).throw(AssertionError("must not rerun")),
                              probe_eval=lambda: LP,
                              append_row=rows.append)
    assert lp == LP and len(rows) == 1   # bills only the probe it actually ran
