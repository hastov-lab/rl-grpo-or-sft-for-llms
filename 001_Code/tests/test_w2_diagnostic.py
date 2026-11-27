# fixed routing and artifact checks for the W2 declarative diagnostic
import json
import os

from data.synthetic_facts import build_bundle, build_declarative_diagnostic
from scripts.s12_eval import atomic_write_json, read_jsonl
from scripts.s28_compose import ADAPTER_FILES
from scripts.w2_diagnostic import (classify_diagnostic, commit_diagnostic_eval,
                                   continuation_rows, diagnostic_state,
                                   first_synthetic_name, generation_rows,
                                   load_diagnostic, metric_summary,
                                   write_diagnostic)


def _write(path, text="x"):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        f.write(text)


def _rows(diagnostic, exact_injected, paraphrase_injected,
          exact_control=0.25, paraphrase_control=0.25):
    rates = {
        ("injected", "exact"): exact_injected,
        ("injected", "declarative_1"): paraphrase_injected,
        ("injected", "declarative_2"): paraphrase_injected,
        ("control", "exact"): exact_control,
        ("control", "declarative_1"): paraphrase_control,
        ("control", "declarative_2"): paraphrase_control,
    }
    selected = {(group, variant): [] for group, variant in rates}
    for row in diagnostic["rows"]:
        selected[(row["group"], row["variant"])].append(row)
    out = []
    for key, source_rows in selected.items():
        n_correct = round(len(source_rows) * rates[key])
        for i, source in enumerate(source_rows):
            out.append({"fact_id": source["fact_id"], "variant": source["variant"],
                        "group": source["group"], "correct": i < n_correct})
    return out


def _diagnostic():
    source = build_bundle("smoke", 34001, 16, 16)
    return source, build_declarative_diagnostic(source)


def test_first_synthetic_name_uses_the_first_named_entity():
    names = {"Potor", "xizor", "fawel"}
    assert first_synthetic_name(" Potor is a city in xizor.", names) == "potor"
    assert first_synthetic_name(" xizor.", names) == "xizor"
    assert first_synthetic_name("none of these", names) is None


def test_generation_and_continuation_rows_keep_fixed_identity():
    _, diagnostic = _diagnostic()
    outputs = [" " + row["object"] + "." for row in diagnostic["rows"]]
    generated = generation_rows(diagnostic, outputs)
    assert all(row["correct"] for row in generated)

    scores = []
    for row in diagnostic["rows"]:
        scores.append({value: 1.0 if value == row["object"] else 0.0
                       for value in row["candidates"]})
    continued = continuation_rows(diagnostic, scores)
    assert all(row["correct"] and row["correct_margin"] == 1.0 for row in continued)


def test_diagnostic_routing_covers_every_registered_outcome():
    _, diagnostic = _diagnostic()
    baseline = _rows(diagnostic, 0.25, 0.25)
    acquired = _rows(diagnostic, 0.75, 0.75)
    exact_only = _rows(diagnostic, 0.75, 0.25)
    failed = _rows(diagnostic, 0.25, 0.25)
    interference = _rows(diagnostic, 0.75, 0.75, exact_control=0.50)
    unexpected = _rows(diagnostic, 0.25, 0.75)

    assert classify_diagnostic(baseline, baseline, acquired, acquired)["route"] == (
        "declarative_acquired")
    assert classify_diagnostic(baseline, baseline, exact_only, exact_only)["route"] == (
        "exact_only")
    assert classify_diagnostic(baseline, baseline, failed, failed)["route"] == (
        "declarative_unqualified")
    assert classify_diagnostic(baseline, baseline, interference, interference)["route"] == (
        "interference")
    assert classify_diagnostic(baseline, baseline, unexpected, unexpected)["route"] == (
        "unexpected_order")
    assert classify_diagnostic(baseline, baseline, acquired, failed)["route"] == (
        "measure_disagreement")

    result = classify_diagnostic(baseline, baseline, acquired, acquired)
    assert result["registered_w2_evidence"] is False
    assert result["c1_claim_allowed"] is False
    assert result["repair_authorised"] is False


def test_diagnostic_commit_binds_data_start_and_adapter(tmp_path):
    source, diagnostic = _diagnostic()
    source_path = os.path.join(tmp_path, "source.json")
    diagnostic_path = os.path.join(tmp_path, "diagnostic.json")
    atomic_write_json(source_path, source)
    write_diagnostic(diagnostic_path, diagnostic, source)
    assert load_diagnostic(diagnostic_path, source) == diagnostic

    start_path = os.path.join(tmp_path, "start.json")
    atomic_write_json(start_path, {
        "dataset_sha256": diagnostic["source_dataset_sha256"],
        "merged_sha256": {"model.safetensors": "abc"},
    })
    adapter = os.path.join(tmp_path, "adapter")
    for name in ADAPTER_FILES:
        _write(os.path.join(adapter, name), name)

    outputs = [" " + row["object"] + "." for row in diagnostic["rows"]]
    gen_rows = generation_rows(diagnostic, outputs)
    values = [{candidate: 1.0 if candidate == row["object"] else 0.0
               for candidate in row["candidates"]} for row in diagnostic["rows"]]
    score_rows = continuation_rows(diagnostic, values)
    scratch = os.path.join(tmp_path, "scratch")
    os.makedirs(scratch)
    for name, rows in (("generation_preds.jsonl", gen_rows),
                       ("continuation_preds.jsonl", score_rows)):
        with open(os.path.join(scratch, name), "w") as f:
            for row in rows:
                f.write(json.dumps(row) + "\n")
    atomic_write_json(os.path.join(scratch, "generation.json"),
                      metric_summary("sft", "direct_generation", gen_rows))
    atomic_write_json(os.path.join(scratch, "continuation.json"),
                      metric_summary("sft", "continuation_score", score_rows))

    result = os.path.join(tmp_path, "result")
    assert commit_diagnostic_eval(
        result, scratch, "sft", diagnostic_path, start_path, adapter) == "committed"
    assert diagnostic_state(result, "sft", diagnostic_path, start_path, adapter) == (
        "committed")
    assert len(read_jsonl(os.path.join(result, "generation_preds.jsonl"))) == 96
    _write(os.path.join(adapter, "adapter_model.safetensors"), "changed")
    assert diagnostic_state(result, "sft", diagnostic_path, start_path, adapter) == (
        "tampered")


def test_worker_writes_both_diagnostic_measurements(monkeypatch, tmp_path):
    import train.model_utils
    import scripts.w2_diagnostic_worker as worker

    _, diagnostic = _diagnostic()

    class Model:
        def eval(self):
            return self

    monkeypatch.setattr(train.model_utils, "load_model_and_tokenizer",
                        lambda *args, **kwargs: (Model(), object()))
    monkeypatch.setattr(train.model_utils, "set_all_seeds", lambda seed: None)
    monkeypatch.setattr(worker, "_model_paths", lambda *args: ("model", None))
    monkeypatch.setattr(worker, "_generate", lambda *args: [
        " " + row["object"] + "." for row in diagnostic["rows"]])
    monkeypatch.setattr(worker, "_score_candidates", lambda *args: [
        {candidate: 1.0 if candidate == row["object"] else 0.0
         for candidate in row["candidates"]} for row in diagnostic["rows"]])

    class Eval:
        batch_size = 16

    class Train:
        seed = 0

    class Config:
        eval = Eval()
        train = Train()

    scratch = os.path.join(tmp_path, "scratch")
    worker.run_diagnostic(Config(), "cpu", diagnostic, {"merged_dir": "model"},
                          "baseline", scratch)
    assert sorted(os.listdir(scratch)) == [
        "continuation.json", "continuation_preds.jsonl",
        "generation.json", "generation_preds.jsonl"]


def test_diagnostic_notebook_keeps_the_evaluation_only_path():
    with open("notebooks/034_1_colab_w2_diagnostic.ipynb") as f:
        notebook = json.load(f)
    code = "\n".join("".join(cell.get("source", []))
                     for cell in notebook["cells"] if cell["cell_type"] == "code")
    assert "cu_before =" in code and "cu_after =" in code
    assert "unit_rate =" in code
    assert "scripts/w2_diagnostic_worker.py" in code
    assert "classify_diagnostic" in code
    assert "build_grpo_split" not in code and "build_eval_split" not in code
    assert "run_sft" not in code and "run_grpo" not in code
