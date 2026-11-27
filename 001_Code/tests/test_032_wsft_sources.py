import ast
import inspect
import json
import os
import re

import pytest
import yaml

from checkpoint import config_hash
from config import load_config, run_tag
from scripts.s12_eval import GateFailure, IntegrityError
from scripts.s32_wsft import (ARMS_032, GATE_FILES, GATE_MANIFEST, LINEAGE_RECORD,
                              SCALES_032, append_failed_gate_contact_correction,
                              arm_tag, commit_sft_endpoint, commit_source_gate,
                              config_path, endpoint_state, pipeline_config_path,
                              prepare_sft_start, source_gate_state, training_facts,
                              verify_source_lineage)


# the historical wsft recipes 032 has to reproduce, seed and source aside
ORIGINALS = {
    "B05": "configs/008_wsft_colab_b05.yaml",
    "B3": "configs/008_wsft_colab_b3.yaml",
    "B15": "configs/011_wsft_colab_b15.yaml",
    "B7": "configs/015_wsft_colab_b7.yaml",
}


def read_yaml(path):
    with open(path) as f:
        return yaml.safe_load(f)


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value))


def make_adapter(path, cfg=None):
    path.mkdir(parents=True, exist_ok=True)
    (path / "adapter_model.safetensors").write_bytes(b"weights")
    (path / "adapter_config.json").write_text("{}")
    (path / "adapter_base.json").write_text("{}")
    (path / "config_hash.txt").write_text(config_hash(cfg) if cfg else "hash")


def make_merged(path):
    path.mkdir(parents=True, exist_ok=True)
    (path / "config.json").write_text("{}")
    (path / "tokenizer.json").write_text("{}")
    (path / "tokenizer_config.json").write_text("{}")
    (path / "model.safetensors").write_bytes(b"merged")


def pred_rows(pred="A"):
    return [{"subject": f"subject_{i % 4}", "tier": "easy", "gold": "A",
             "pred": pred, "correct": pred == "A"} for i in range(1884)]


def write_jsonl(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        for row in rows:
            f.write(json.dumps(row) + "\n")


def test_032_card_groups_and_paths_are_fixed():
    # arms split by card size, and base evals never come from a wsft run
    assert ARMS_032["40"] == [("B05", 1), ("B05", 2), ("B3", 1), ("B3", 2)]
    assert ARMS_032["80"] == [("B15", 1), ("B15", 2), ("B7", 1), ("B7", 2)]
    assert set(ARMS_032["40"] + ARMS_032["80"]) == {
        (scale, seed) for scale in SCALES_032 for seed in (1, 2)}
    assert all("wsft" not in info["base_eval"] for info in SCALES_032.values())
    assert SCALES_032["B3"]["vram_gib"] < 40
    assert SCALES_032["B7"]["vram_gib"] < 80
    for phase in ARMS_032.values():
        for scale, seed in phase:
            assert os.path.exists(pipeline_config_path(scale, seed))


def test_032_configs_match_the_historical_recipes():
    for phase in ("40", "80"):
        for scale, seed in ARMS_032[phase]:
            old = read_yaml(ORIGINALS[scale])
            new = read_yaml(config_path(scale, seed))
            assert old.pop("experiment") in ("008_wsft", "011_wsft", "015_wsft")
            assert new.pop("experiment") == "032_wsft"
            assert old["train"].pop("seed") == 0
            assert new["train"].pop("seed") == seed
            assert old["train"].pop("sft_from_adapter").endswith("seed0/sft")
            source = new["train"].pop("sft_from_adapter")
            assert source == f"checkpoints/{SCALES_032[scale]['pipeline'].format(s=seed)}/sft"
            # older configs have no exec block at all, normalise before comparing
            old_exec = old.setdefault("exec", {})
            new_exec = new.setdefault("exec", {})
            old_exec.pop("allow_oom_fallback", True)
            assert new_exec.pop("allow_oom_fallback") is False
            if not old_exec:
                old.pop("exec")
            if not new_exec:
                new.pop("exec")
            assert old == new

            cfg = load_config(config_path(scale, seed))
            assert run_tag(cfg) == arm_tag(scale, seed)
            assert cfg.exec.allow_oom_fallback is False


def test_prepare_start_uses_the_parent_merge(tmp_path):
    # prefer the historical merge, build a bound fallback only when it is gone
    cfg = load_config(config_path("B05", 1))
    cfg.output_root = str(tmp_path)
    adapter = tmp_path / cfg.train.sft_from_adapter
    make_adapter(adapter)
    merged = adapter.parent / "sft_merged"
    make_merged(merged)
    result = tmp_path / "results" / "arm"
    sft_out = tmp_path / "checkpoints" / "arm" / "sft"

    binding = prepare_sft_start(cfg, str(result), str(sft_out))

    assert binding["start_kind"] == "parent_merged"
    assert binding["resolved_start"] == str(merged)
    assert (result / "source_start_binding.json").exists()


def test_prepare_start_builds_and_binds_the_fallback(tmp_path):
    cfg = load_config(config_path("B05", 1))
    cfg.output_root = str(tmp_path)
    adapter = tmp_path / cfg.train.sft_from_adapter
    make_adapter(adapter)
    result = tmp_path / "results" / "arm"
    sft_out = tmp_path / "checkpoints" / "arm" / "sft"

    def build():
        fallback = (tmp_path / "checkpoints" /
                    f"{cfg.experiment}_{cfg.scale.name}_{adapter.parent.name}_merged_base")
        make_merged(fallback)
        return str(fallback)

    binding = prepare_sft_start(cfg, str(result), str(sft_out), merge_fn=build)

    assert binding["start_kind"] == "fallback_merged"
    assert os.path.exists(os.path.join(binding["resolved_start"], "merge_binding.json"))


def test_source_lineage_binds_the_parent_training_path(tmp_path):
    # bind by path and provenance, byte reconstruction
    cfg = load_config(config_path("B05", 1))
    cfg.output_root = str(tmp_path)
    adapter = tmp_path / cfg.train.sft_from_adapter
    make_adapter(adapter)
    merged = adapter.parent / "sft_merged"
    make_merged(merged)
    grpo = adapter.parent / "grpo"
    make_adapter(grpo)
    expected_base = os.path.relpath(merged, tmp_path)
    write_json(grpo / "adapter_base.json", {"kind": "local", "value": expected_base})
    parent_result = tmp_path / "results" / adapter.parent.name
    parent_manifest = parent_result / "manifest.json"
    write_json(parent_manifest, {"run_tag": adapter.parent.name, "seed": 1})
    result = tmp_path / "results" / "arm"
    sft_out = tmp_path / "checkpoints" / "arm" / "sft"

    lineage = verify_source_lineage(cfg, str(result), str(sft_out), str(parent_manifest))

    assert lineage["binding_level"] == "path_and_provenance"
    assert lineage["byte_reconstruction"] is False
    assert lineage["parent_grpo_base"] == {"kind": "local", "value": expected_base}
    assert (result / LINEAGE_RECORD).exists()

    write_json(grpo / "adapter_base.json", {"kind": "hf_id", "value": "wrong"})
    with pytest.raises(IntegrityError):
        verify_source_lineage(cfg, str(result), str(sft_out), str(parent_manifest))


def test_source_lineage_never_builds_a_missing_parent_merge(tmp_path):
    # a missing parent halts the arm, nothing historical is ever rebuilt
    cfg = load_config(config_path("B05", 1))
    cfg.output_root = str(tmp_path)
    adapter = tmp_path / cfg.train.sft_from_adapter
    make_adapter(adapter)
    parent_manifest = tmp_path / "results" / adapter.parent.name / "manifest.json"
    write_json(parent_manifest, {"run_tag": adapter.parent.name, "seed": 1})

    with pytest.raises(IntegrityError, match="historical parent merge is missing"):
        verify_source_lineage(cfg, str(tmp_path / "results" / "arm"),
                              str(tmp_path / "checkpoints" / "arm" / "sft"),
                              str(parent_manifest))

    assert not (adapter.parent / "sft_merged").exists()


def test_failed_gate_contact_correction_is_append_only(tmp_path):
    # a failed gate still spent contact, append the correction and leave the row
    ledger = tmp_path / "compute_ledger.jsonl"
    original = {"run_tag": arm_tag("B05", 1), "eval_items": {"source_gate": None}}
    write_jsonl(ledger, [original])

    assert append_failed_gate_contact_correction(str(ledger)) is True
    assert append_failed_gate_contact_correction(str(ledger)) is False

    rows = [json.loads(line) for line in ledger.read_text().splitlines()]
    assert rows[0] == original
    assert rows[1]["eval_items"] == {"source_gate": 1884}
    assert rows[1]["evaluated"] is True and rows[1]["committed"] is False


def test_source_gate_commits_and_detects_archive_drift(tmp_path):
    result = tmp_path / "results"
    scratch = tmp_path / "scratch"
    archive = tmp_path / "archive.jsonl"
    rows = pred_rows()
    write_jsonl(archive, rows)
    write_jsonl(scratch / GATE_FILES[1], rows)
    write_json(scratch / GATE_FILES[0], {"n_items": 1884})
    binding = {"resolved_start": "/start", "resolved_start_sha256": {"m": "h"}}

    commit_source_gate(str(result), str(scratch), binding, str(archive))

    assert source_gate_state(str(result), binding, str(archive)) == "committed"
    manifest = json.loads((result / GATE_MANIFEST).read_text())
    # 0.99 rather than 1.0: merged and adapter representations split near ties
    assert manifest["thresholds"] == {"min_agreement": 0.99, "max_acc_diff": 0.001}
    assert manifest["review_below_exact_agreement"] is False
    write_jsonl(archive, pred_rows("B"))
    with pytest.raises(IntegrityError):
        source_gate_state(str(result), binding, str(archive))


def test_source_gate_failure_writes_nothing(tmp_path):
    result = tmp_path / "results"
    scratch = tmp_path / "scratch"
    archive = tmp_path / "archive.jsonl"
    write_jsonl(archive, pred_rows())
    write_jsonl(scratch / GATE_FILES[1], pred_rows("B"))
    write_json(scratch / GATE_FILES[0], {"n_items": 1884})

    with pytest.raises(GateFailure):
        commit_source_gate(str(result), str(scratch), {"start": "x"}, str(archive))
    assert not result.exists()


def test_source_gate_records_a_passing_non_exact_match(tmp_path):
    result = tmp_path / "results"
    scratch = tmp_path / "scratch"
    archive = tmp_path / "archive.jsonl"
    archived = pred_rows()
    current = pred_rows()
    current[0] = {**current[0], "pred": "B", "correct": False}
    write_jsonl(archive, archived)
    write_jsonl(scratch / GATE_FILES[1], current)
    write_json(scratch / GATE_FILES[0], {"n_items": 1884})

    manifest = commit_source_gate(str(result), str(scratch), {"start": "x"}, str(archive))

    assert manifest["stats"]["agreement"] < 1.0
    assert manifest["review_below_exact_agreement"] is True


def test_training_and_endpoint_commit_are_adapter_bound(tmp_path):
    cfg = load_config(config_path("B05", 1))
    cfg.output_root = str(tmp_path)
    result = tmp_path / "results" / "arm"
    adapter = tmp_path / "checkpoints" / "arm" / "sft"
    start = tmp_path / "checkpoints" / "parent" / "sft_merged"
    make_merged(start)
    make_adapter(adapter, cfg)
    write_json(adapter / "adapter_base.json",
               {"kind": "local", "value": os.path.relpath(start, tmp_path)})
    binding = {"resolved_start": str(start)}

    history = [{"step": step, "loss": 1.0,
                "learning_rate": cfg.train.sft_lr * (cfg.train.max_steps - step) /
                cfg.train.max_steps}
               for step in range(1, cfg.train.max_steps + 1)]
    write_json(result / "sft_trainlog.json", history)
    write_json(result / "sft_logprob.json", {"n_items": 1884,
                                              "per_tier_acc": {"easy": 1.0}})
    write_jsonl(result / "sft_logprob_preds.jsonl", pred_rows())
    write_json(result / "manifest.json", {"seed": 1})
    write_json(result / "summary.json", {"sft_acc": None,
                                          "sft_logprob_acc": {"easy": 1.0}})

    facts = training_facts(cfg, str(result), str(adapter), binding)
    assert facts["steps"] == 123

    scratch = tmp_path / "scratch"
    write_json(scratch / "sft_eval.json",
               {"n_items": 1884, "per_tier_acc": {"easy": 1.0}})
    write_jsonl(scratch / "sft_preds.jsonl", pred_rows())
    commit_sft_endpoint(str(result), str(scratch), str(adapter))
    assert endpoint_state(str(result), str(adapter)) == "committed"

    (adapter / "adapter_model.safetensors").write_bytes(b"changed")
    with pytest.raises(IntegrityError):
        endpoint_state(str(result), str(adapter))


def test_logprob_eval_accepts_an_explicit_model_path():
    from eval.harness import run_logprob_eval
    from train.model_utils import load_model_and_tokenizer

    assert "model_path" in inspect.signature(run_logprob_eval).parameters
    assert "model_path" in inspect.signature(load_model_and_tokenizer).parameters


def test_032_source_probe_uses_the_archived_adapter_representation():
    # probe the adapter the archive actually used, never a rebuilt merge
    source = open("scripts/s32_start_probe.py").read()
    assert "adapter_path=args.adapter" in source
    assert "model_path=args" not in source


@pytest.mark.parametrize("phase", ("40", "80"))
def test_032_notebooks_keep_the_card_group_gate_barrier(phase):
    path = f"notebooks/032_colab_wsft_{phase}gb.ipynb"
    with open(path) as f:
        notebook = json.load(f)
    cells = ["".join(cell.get("source", [])) for cell in notebook["cells"]]
    source = "\n".join(cells)

    assert f"PHASE = '{phase}'" in source
    assert "all four source gates committed before training" in source
    assert "gate_inputs[(scale, seed)]" in source
    # all four gates commit before any arm trains, that ordering is the barrier
    assert source.index("## 9. Source identity gates") < source.index(
        "## 10. Train and evaluate the four arms")
    train_cell = next(value for value in cells if "gate_inputs = {}" in value)
    assert train_cell.index("source_gate_state") < train_cell.index(
        "for scale, seed in ARMS:\n    cfg, result_dir")
    assert "scripts/run.py" in train_cell and "scripts/s32_sft_eval.py" in train_cell
    assert "verify_source_lineage" in source
    assert "--adapter" in source and "--model" not in source
    assert "032_execution_binding_v6.json" in source
    assert "f'032_preflight_{PHASE}_v6.json'" in source
    assert source.count("032 v6 preflight changed") == 2
    assert "if not (code_root / 'scripts' / 'run.py').is_file():" in source
    assert "code_root = repo_root / '001_Code'" in source
    assert source.count("os.chdir(code_root)") == 2
    # 032 is sft only, a grpo import here would mean the wrong runner
    assert "grpo_worker" not in source and "sentinel" not in source
    assert re.search(r"cu_before = (None|[\d.]+)", source)
    assert re.search(r"unit_rate = (None|[\d.]+)", source)
    assert re.search(r"cu_after = (None|[\d.]+)", source)
    assert "WANDB_PROJECT'] = 'inm363-grpo'" in source
    assert "results/base_evals/006" not in source
    assert "prepopulate_scale_base" in source
    assert "config_hash_status" in source
    assert "preflight['historical']" in source

    code_cells = [cell for cell in notebook["cells"] if cell["cell_type"] == "code"]
    counts = [cell.get("execution_count") for cell in code_cells]
    assert all(value is None or isinstance(value, int) for value in counts)
    assert not any(output.get("output_type") == "error"
                   for cell in code_cells for output in cell.get("outputs", []))
    for cell in notebook["cells"]:
        if cell["cell_type"] == "code":
            ast.parse("".join(cell["source"]))


def test_032_analysis_uses_the_registered_family_scopes():
    with open("notebooks/032_1_analysis_plots.ipynb") as f:
        notebook = json.load(f)
    source = "\n".join("".join(cell.get("source", [])) for cell in notebook["cells"])

    for family in ("direct_overall", "direct_hard", "D1_overall", "D1_hard",
                   "D2_overall", "D2_hard"):
        assert family in source
    assert "seed in (1, 2)" in source
    assert "historical and descriptive in derived tables" in source
    assert "pipeline_tag(scale, seed), 'sft_logprob'" in source
    assert "descriptive bundle comparison including merge representation and MMLU SFT" in source
    # the probe comparison stays descriptive here, no registered reading
    assert "probe_reading" not in source
    assert "align_pred_rows" in source and "holm(" in source
    assert "032_comparison_summary.json" in source
    assert "analysis/figures/032_1" in source

    code_cells = [cell for cell in notebook["cells"] if cell["cell_type"] == "code"]
    counts = [cell.get("execution_count") for cell in code_cells]
    assert all(value is None for value in counts) or all(
        isinstance(value, int) for value in counts)
    assert not any(output.get("output_type") == "error"
                   for cell in code_cells for output in cell.get("outputs", []))
    for cell in notebook["cells"]:
        if cell["cell_type"] == "code":
            ast.parse("".join(cell["source"]))
