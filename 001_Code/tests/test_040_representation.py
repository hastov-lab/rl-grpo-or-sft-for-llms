import json
import os
import re
from types import SimpleNamespace

import pytest

from scripts.s12_eval import ADAPTER_FILES, IntegrityError, sha256
from scripts.s40_representation import (ARMS_040, REPRESENTATIONS, SCALES_040,
                                        compare_rows, load_pool, pair_tag,
                                        pipeline_config_path, prepare_pool,
                                        result_files, run_probe, source_facts)


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value))


def make_source_tree(root, scale="B05", seed=0):
    from scripts.s32_wsft import pipeline_tag

    # mirror the archived path links checked before a real pair runs
    tag = pipeline_tag(scale, seed)
    adapter = root / "checkpoints" / tag / "sft"
    merged = root / "checkpoints" / tag / "sft_merged"
    grpo = root / "checkpoints" / tag / "grpo"
    for name in ADAPTER_FILES:
        path = adapter / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(name.encode())
    for name in ("config.json", "tokenizer.json", "tokenizer_config.json"):
        write_json(merged / name, {})
    (merged / "model.safetensors").write_bytes(b"merged")
    write_json(grpo / "adapter_base.json", {
        "kind": "local", "value": os.path.relpath(merged, root)})
    write_json(root / "results" / tag / "manifest.json", {
        "run_tag": tag, "seed": seed, "scale": scale})
    return tag, adapter, merged


def pred_rows():
    # a wide default margin so any planted change reads as a near tie
    rows = []
    for i in range(245):
        rows.append({
            "item_id": f"item-{i}", "subject": f"subject-{i % 2}",
            "gold": "A", "pred": "A", "correct": True,
            "letter_logits": {"A": 4.0, "B": 1.0, "C": 0.0, "D": -1.0},
        })
    return rows


def test_040_uses_all_pipeline_scales_and_seeds():
    assert SCALES_040 == ("B05", "B15", "B3", "B7")
    assert len(ARMS_040) == 12 and len(set(ARMS_040)) == 12
    assert set(ARMS_040) == {
        (scale, seed) for scale in SCALES_040 for seed in (0, 1, 2)}
    assert REPRESENTATIONS == ("adapter", "merged")
    for scale, seed in ARMS_040:
        assert os.path.exists(pipeline_config_path(scale, seed))
        assert pair_tag(scale, seed) == f"040_representation_{scale}_seed{seed}"


def test_source_facts_bind_the_archived_pair(tmp_path):
    # the GRPO base record links the pipeline to this merged model
    tag, adapter, merged = make_source_tree(tmp_path)
    pool = {"path": "/pool", "sha256": "pool-hash", "n_items": 245}

    record = source_facts(str(tmp_path), "B05", 0, pool, "bfloat16")

    assert record["pipeline_tag"] == tag
    assert record["adapter_dir"] == str(adapter)
    assert record["merged_dir"] == str(merged)
    assert record["adapter_sha256"]["adapter_model.safetensors"] == sha256(
        adapter / "adapter_model.safetensors")
    assert record["merged_sha256"]["model.safetensors"] == sha256(
        merged / "model.safetensors")
    assert record["resolved_dtype"] == "bfloat16"

    write_json(tmp_path / "checkpoints" / tag / "grpo" / "adapter_base.json", {
        "kind": "local", "value": "wrong"})
    with pytest.raises(IntegrityError, match="did not use this merged model"):
        source_facts(str(tmp_path), "B05", 0, pool, "bfloat16")


def test_pool_is_stable_and_unique(monkeypatch, tmp_path):
    import scripts.s40_representation as rep

    rows = [{"probe": f"question {i}", "gold": "A", "subject": "s", "tier": "easy"}
            for i in range(245)]
    monkeypatch.setattr(rep, "_pool_rows", lambda: [
        {**row, "item_id": f"item-{i}"} for i, row in enumerate(rows)])
    path = tmp_path / "pool.jsonl"

    first = prepare_pool(str(path))
    second = prepare_pool(str(path))

    assert first == second and first["n_items"] == 245
    assert len(load_pool(str(path), first["sha256"])) == 245
    # an extra row changes the hash and the count, both have to be caught
    path.write_text(path.read_text() + "{}\n")
    with pytest.raises(IntegrityError):
        load_pool(str(path), first["sha256"])


def test_comparison_counts_each_flip_type():
    # keep one example of every correctness transition in the paired summary
    adapter = pred_rows()
    merged = pred_rows()
    merged[0].update(pred="B", correct=False,
                     letter_logits={"A": 1.0, "B": 1.1, "C": 0.0, "D": -1.0})
    adapter[1].update(pred="B", correct=False)
    merged[1].update(pred="A", correct=True)
    adapter[2].update(pred="B", correct=False)
    merged[2].update(pred="C", correct=False)

    result = compare_rows(adapter, merged)

    assert result["n_changed"] == 3
    assert result["agreement"] == pytest.approx(242 / 245)
    assert result["accuracy_delta"] == 0.0
    assert result["flips"] == {
        "correct_to_wrong": 1,
        "wrong_to_correct": 1,
        "wrong_to_different_wrong": 1,
    }
    assert result["top_two_margin"]["changed_mean"] < result["top_two_margin"][
        "unchanged_mean"]


def test_comparison_rejects_a_different_item_order():
    adapter = pred_rows()
    merged = pred_rows()
    merged[0]["item_id"] = "different"
    with pytest.raises(IntegrityError, match="do not align"):
        compare_rows(adapter, merged)


def test_worker_uses_the_selected_representation(monkeypatch, tmp_path):
    # exercise both dispatch branches without loading a real model
    import checkpoint
    import eval.harness as harness
    import scripts.env_switch as env_switch
    import scripts.s40_representation as rep

    pool = tmp_path / "pool.jsonl"
    rows = pred_rows()
    pool.write_text("".join(json.dumps(row) + "\n" for row in rows))
    source = {
        "scale": "B05", "seed": 0, "config": "cfg.yaml",
        "adapter_dir": "/adapter", "merged_dir": "/merged",
        "resolved_dtype": "float32",
        "pool": {"sha256": sha256(pool)},
    }
    source_path = tmp_path / "source.json"
    write_json(source_path, source)
    cfg = SimpleNamespace(scale=SimpleNamespace(name="B05"),
                          train=SimpleNamespace(seed=0, dtype="float32"))
    calls = []
    monkeypatch.setattr(rep, "load_pool", lambda *args: rows)
    monkeypatch.setattr("config.load_config", lambda path: cfg)
    monkeypatch.setattr(env_switch, "resolve_device", lambda env: "cpu")
    cfg.env = SimpleNamespace(name="COLAB")

    def evaluate(*args, **kwargs):
        calls.append(kwargs)
        return {"n_items": 245, "overall_ci": (1.0, 1.0, 1.0)}

    monkeypatch.setattr(harness, "run_logprob_rows", evaluate)
    monkeypatch.setattr(checkpoint, "save_metrics", lambda path, metrics: None)

    run_probe(str(source_path), str(pool), "adapter", str(tmp_path / "a"))
    run_probe(str(source_path), str(pool), "merged", str(tmp_path / "m"))

    assert calls[0]["adapter_path"] == "/adapter" and "model_path" not in calls[0]
    assert calls[1]["model_path"] == "/merged" and "adapter_path" not in calls[1]
    assert all(call["include_scores"] is True for call in calls)
    assert all(call["identity_fields"] == ("item_id",) for call in calls)

    source["resolved_dtype"] = "bfloat16"
    write_json(source_path, source)
    with pytest.raises(IntegrityError, match="runtime dtype differs"):
        run_probe(str(source_path), str(pool), "adapter", str(tmp_path / "bad"))


def test_040_notebook_is_a_clean_evaluation_runner():
    path = "notebooks/040_colab_representation_numerics.ipynb"
    with open(path) as f:
        notebook = json.load(f)
    code = "\n".join("".join(cell.get("source", []))
                     for cell in notebook["cells"] if cell["cell_type"] == "code")
    ids = [cell["id"] for cell in notebook["cells"]]

    for cell in notebook["cells"]:
        if cell["cell_type"] == "code":
            compile("".join(cell.get("source", [])), path, "exec")
    assert len(ids) == len(set(ids)) and all(re.fullmatch(r"[0-9a-f]{8}", value) for value in ids)
    # runner was executed since, so allow the counts and check that nothing errored
    code_cells = [cell for cell in notebook["cells"] if cell["cell_type"] == "code"]
    counts = [cell.get("execution_count") for cell in code_cells]
    assert all(value is None or isinstance(value, int) for value in counts)
    assert not any(output.get("output_type") == "error"
                   for cell in code_cells for output in cell.get("outputs", []))
    # blank before the run, filled from the session after it
    assert re.search(r"cu_before = (None|[\d.]+)", code)
    assert re.search(r"cu_after = (None|[\d.]+)", code)
    assert "total_gib >= 79" in code
    assert "ARMS_040" in code and "REPRESENTATIONS" in code
    assert "pair_tag, pipeline_config_path, prepare_pool" in code
    assert "'resolved_dtype': source['resolved_dtype']" in code
    assert "scripts/s40_representation.py" in code
    # keep training, test data and tracking out of this probe runner
    assert "run_grpo" not in code and "run_sft" not in code
    assert "build_eval_split" not in code and "run_eval" not in code
    assert "WANDB" not in code
    # widget state arrives with any colab run, five of the older runners carry it too
    assert notebook["metadata"].get("authors") is None


def test_result_names_are_representation_specific():
    assert result_files("adapter") == (
        "adapter_logprob.json", "adapter_logprob_preds.jsonl")
    assert result_files("merged") == (
        "merged_logprob.json", "merged_logprob_preds.jsonl")
