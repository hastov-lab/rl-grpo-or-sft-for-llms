import copy
import json
import os
import sys
from dataclasses import asdict
from types import SimpleNamespace

import pytest
import yaml

from checkpoint import write_artifact_type
from config import load_config
from scripts.s12_eval import IntegrityError, sha256
from scripts.s42_worker import ENDPOINT_FILES
from scripts.s46_gpt2_xl import (A1_ERROR, ALLOCATOR_CONFIG, CONFIGS_046,
                                 RESUME_MEMORY_LABELS, SEEDS_046,
                                 a1_state, archive_memory_margin_pilot, arm_tag,
                                 check_registered_config,
                                 check_card_vram, commit_a1_conclusion, commit_card_lock,
                                 commit_eval, commit_sft_training, config_path,
                                 eval_state, pilot_passes, training_record,
                                 verify_grpo_adapter_base)
from scripts.s46_worker import (_check_eval_source,
                                _instrument_resume_memory,
                                _merge_logit_check,
                                _reference_behaviour_report)

TARGETS = ("W_query", "W_key", "W_value", "out_proj", "layers.0", "layers.2")


def make_adapter(path, salt="adapter"):
    os.makedirs(path, exist_ok=True)
    for name in ("adapter_model.safetensors", "adapter_config.json",
                 "config_hash.txt"):
        with open(os.path.join(path, name), "w") as handle:
            handle.write(f"{salt}-{name}")
    with open(os.path.join(path, "adapter_base.json"), "w") as handle:
        json.dump({"kind": "local", "value": "base"}, handle)
    write_artifact_type(path, "adapter", family="scratch_gpt", scale="GPT2_B15")
    return path


def make_endpoint(path, branch):
    os.makedirs(path, exist_ok=True)
    row = json.dumps({"subject": "test", "tier": "easy", "gold": "A",
                      "pred": "A", "correct": True}) + "\n"
    for name in ENDPOINT_FILES[branch]:
        target = os.path.join(path, name)
        with open(target, "w") as handle:
            if name.endswith("preds.jsonl"):
                handle.write(row * 1884)
            else:
                json.dump({"condition": branch, "n_items": 1884,
                           "per_tier_acc": {"easy": 1.0}}, handle)
    return path


def test_046_configs_fix_the_three_seed_lora_roster():
    # remove only the seed before checking the fixed scientific roster
    rows = []
    for seed, path in zip(SEEDS_046, CONFIGS_046):
        cfg = check_registered_config(load_config(path))
        assert cfg.train.seed == seed
        assert cfg.scale.name == "GPT2_B15"
        assert cfg.train.max_seq_len == 1024
        assert cfg.grpo.max_completion_length == cfg.eval.max_new_tokens == 512
        assert cfg.exec.gradient_checkpointing is False
        row = asdict(cfg)
        row["train"].pop("seed")
        rows.append(row)
    assert rows[0] == rows[1] == rows[2]


def test_046_yaml_files_differ_only_by_seed():
    rows = []
    for seed, path in zip(SEEDS_046, CONFIGS_046):
        with open(path) as handle:
            row = yaml.safe_load(handle)
        assert row["train"].pop("seed") == seed
        rows.append(row)
    assert rows[0] == rows[1] == rows[2]


def test_046_observe_allowance_stays_narrow(tmp_path):
    cfg = load_config(config_path(0))
    changed = copy.deepcopy(cfg)
    changed.experiment = "047"
    with pytest.raises(IntegrityError, match="registered package"):
        check_registered_config(changed)

    text = open(config_path(0)).read().replace('experiment: "046"', 'experiment: "047"')
    path = tmp_path / "bad.yaml"
    path.write_text(text)
    with pytest.raises(ValueError, match="observe mode"):
        load_config(path)


def test_046_paths_are_seed_bound():
    assert config_path(2) == "configs/046_colab_gpt2_b15_seed2.yaml"
    assert arm_tag(1) == "046_COLAB_FULL_GPT2_B15_sft_grpo_seed1"
    with pytest.raises(ValueError):
        arm_tag(3)


def test_046_pilot_requires_both_training_phases_and_cleanup():
    # memory alone cannot pass a pilot with an incomplete precision or LoRA gate
    setup = {"verified": True, "trainable_parameters": 100,
             "trainable_fraction": 0.01, "targets": list(TARGETS),
             "target_module_counts": {name: 48 for name in TARGETS}}
    report = {
        "sft_updates": 5,
        "grpo_updates": 30,
        "allocator_config": ALLOCATOR_CONFIG,
        "resume_memory": [{"label": value} for value in RESUME_MEMORY_LABELS],
        "peak_reserved_fraction": 0.8,
        "adapter_reload_verified": True,
        "merge_verified": True,
        "resume_verified": True,
        "sft_lora_setup": setup,
        "precision_audit": {
            "gradient_dtypes": ["float32"],
            "optimizer_state_dtypes": ["float32"],
            "bf16_enabled": True,
            "accelerator_mixed_precision": "bf16",
            "reference_gate": {"reference_kind": "adapter_disabled"},
            "reference_behaviour": {
                "verified": True, "reference_max_abs_drift": 0.0,
                "adapter_effect_max_abs": 0.1,
            },
            "adapter_base": {"verified": True},
            "lora_setup": setup,
        },
    }
    assert pilot_passes(report, {"completed": True, "weights_deleted": True})
    changed = copy.deepcopy(report)
    changed["resume_verified"] = False
    assert not pilot_passes(changed, {"completed": True, "weights_deleted": True})
    changed = copy.deepcopy(report)
    changed["resume_memory"].pop()
    assert not pilot_passes(changed, {"completed": True, "weights_deleted": True})


@pytest.mark.parametrize("archive_name, reason", (
    ("pilot_80gb_before_logit_projection_fix", "logit projection memory correction"),
    ("pilot_80gb_before_resume_memory_check", "resume memory diagnostic"),
))
def test_046_archives_the_failed_80gb_pilot_before_retry(
        tmp_path, archive_name, reason):
    # preserve the failed run before the corrected pilot reuses its filenames
    card = tmp_path / "pilot_80gb"
    card.mkdir()
    statuses = {
        "pilot_sft_status.json": ("pilot_sft", 5),
        "pilot_grpo_a_status.json": ("pilot_grpo_a", 15),
        "pilot_grpo_b_status.json": ("pilot_grpo_b", 30),
    }
    for name, (task, step) in statuses.items():
        (card / name).write_text(json.dumps({
            "completed": True, "task": task, "end_step": step,
        }))
    for name in ("precision_a.json", "precision_b.json",
                 "sft_trainlog.json", "grpo_trainlog.json"):
        (card / name).write_text(json.dumps({"name": name}))
    report_path = card / "pilot_report.json"
    report_path.write_text(json.dumps({
        "completed": True, "experiment": "046", "card_requested": 80,
        "sft_updates": 5, "grpo_updates": 30,
        "peak_reserved_fraction": 0.99,
        "scientific_metrics_used": False, "mmlu_test_passes": 0,
    }))
    (card / "pilot_closeout.json").write_text(json.dumps({
        "completed": True, "weights_deleted": True,
        "pilot_report_sha256": sha256(report_path),
    }))
    (card / "pilot_failure.json").write_text(json.dumps({
        "completed": True, "failure_kind": "memory", "stage": "memory margin",
        "report_sha256": sha256(report_path),
    }))

    assert archive_memory_margin_pilot(str(card), archive_name, reason) is True
    archive = tmp_path / archive_name
    manifest = json.loads((archive / "archive_manifest.json").read_text())
    assert manifest["completed"] is True
    assert manifest["reason"] == reason
    assert not any((card / name).exists() for name in manifest["files"])
    assert all(sha256(archive / name) == value
               for name, value in manifest["files"].items())
    assert archive_memory_margin_pilot(
        str(card), archive_name, reason) is True


def test_046_retry_archive_halts_on_changed_evidence(tmp_path):
    from scripts.s46_gpt2_xl import PILOT_RETRY_FILES

    card = tmp_path / "pilot_80gb"
    archive = tmp_path / "pilot_80gb_before_logit_projection_fix"
    card.mkdir()
    archive.mkdir()
    files = {}
    for name in PILOT_RETRY_FILES:
        path = archive / name
        path.write_text(name)
        files[name] = sha256(path)
    (archive / "archive_manifest.json").write_text(json.dumps({
        "completed": True, "files": files,
    }))
    (archive / "pilot_report.json").write_text("changed")
    with pytest.raises(IntegrityError, match="archive changed"):
        archive_memory_margin_pilot(str(card))


def test_046_card_lock_accepts_only_a_passing_registered_card(tmp_path):
    setup = {"verified": True, "trainable_parameters": 100,
             "trainable_fraction": 0.01, "targets": list(TARGETS),
             "target_module_counts": {name: 48 for name in TARGETS}}
    report = {
        "sft_updates": 5,
        "grpo_updates": 30,
        "allocator_config": ALLOCATOR_CONFIG,
        "resume_memory": [{"label": value} for value in RESUME_MEMORY_LABELS],
        "peak_reserved_fraction": 0.8,
        "total_vram_gib": 39.5,
        "adapter_reload_verified": True,
        "merge_verified": True,
        "resume_verified": True,
        "sft_lora_setup": setup,
        "precision_audit": {
            "gradient_dtypes": ["float32"],
            "optimizer_state_dtypes": ["float32"],
            "bf16_enabled": True,
            "accelerator_mixed_precision": "bf16",
            "reference_gate": {"reference_kind": "adapter_disabled"},
            "reference_behaviour": {
                "verified": True, "reference_max_abs_drift": 0.0,
                "adapter_effect_max_abs": 0.1,
            },
            "adapter_base": {"verified": True},
            "lora_setup": setup,
        },
    }
    report_path = tmp_path / "pilot.json"
    closeout_path = tmp_path / "closeout.json"
    report_path.write_text(json.dumps(report))
    from scripts.s12_eval import sha256
    closeout_path.write_text(json.dumps({
        "completed": True,
        "weights_deleted": True,
        "pilot_report_sha256": sha256(str(report_path)),
    }))
    record = commit_card_lock(str(tmp_path), str(report_path), str(closeout_path))
    assert record["card"] == "A100_40GB"
    assert record["allocator_config"] == ALLOCATOR_CONFIG
    assert record["minimum_vram_gib"] == 39
    assert check_card_vram(record, 39.5) == "A100_40GB"
    with pytest.raises(IntegrityError, match="card lock"):
        check_card_vram(record, 79.3)


def test_046_resume_memory_wraps_both_restore_stages(monkeypatch):
    # measure model and optimizer restoration as separate checkpoint stages
    import scripts.s46_worker as worker
    import torch

    events = []
    trace = []

    class Trainer:
        def _load_from_checkpoint(self, checkpoint, model=None):
            events.append(("model", checkpoint, model))
            return "model restored"

        def _load_optimizer_and_scheduler(self, checkpoint):
            events.append(("optimizer", checkpoint))
            return "optimizer restored"

    monkeypatch.setattr(
        worker, "_cuda_memory_point", lambda label: {"label": label})
    monkeypatch.setattr(worker.gc, "collect", lambda: events.append(("gc",)))
    monkeypatch.setattr(torch.cuda, "empty_cache", lambda: events.append(("empty",)))

    value = Trainer()
    _instrument_resume_memory(value, trace)
    assert value._load_from_checkpoint("checkpoint-15", model="model") == "model restored"
    assert value._load_optimizer_and_scheduler("checkpoint-15") == "optimizer restored"
    assert [row["label"] for row in trace] == [
        "before _load_from_checkpoint",
        "after _load_from_checkpoint",
        "after _load_from_checkpoint cleanup",
        "before _load_optimizer_and_scheduler",
        "after _load_optimizer_and_scheduler",
        "after _load_optimizer_and_scheduler cleanup",
    ]
    assert events == [
        ("model", "checkpoint-15", "model"), ("gc",), ("empty",),
        ("optimizer", "checkpoint-15"), ("gc",), ("empty",),
    ]


def test_046_resume_memory_rejects_an_unknown_trainer():
    with pytest.raises(IntegrityError, match="_load_from_checkpoint"):
        _instrument_resume_memory(SimpleNamespace(), [])


def test_046_sft_record_binds_adapter_and_complete_schedule(tmp_path):
    cfg = load_config(config_path(0))
    cfg.output_root = str(tmp_path)
    result_dir = tmp_path / "results"
    result_dir.mkdir()
    adapter = make_adapter(str(tmp_path / "adapter"))
    history = [{"step": step, "loss": 1.0 / step} for step in range(1, 501)]
    (result_dir / "sft_trainlog.json").write_text(json.dumps(history))
    base_record = tmp_path / "base.json"
    base_record.write_text(json.dumps({"artifact_sha256": {"model": "abc"}}))
    wandb_path = tmp_path / "wandb.json"
    wandb_path.write_text(json.dumps({"completed": True, "run_id": "run046"}))
    from scripts.s12_eval import sha256
    wandb = [{"path": str(wandb_path), "sha256": sha256(str(wandb_path))}]
    record = commit_sft_training(
        cfg, str(result_dir), adapter, str(base_record), wandb)
    assert record["artifact"]["path"] == os.path.abspath(adapter)
    assert training_record(str(result_dir), "sft") == record

    with open(os.path.join(adapter, "adapter_model.safetensors"), "a") as handle:
        handle.write("changed")
    with pytest.raises(IntegrityError):
        training_record(str(result_dir), "sft")


def test_046_endpoint_uses_its_own_manifest_namespace(tmp_path):
    # keep new endpoint commits separate from the completed 042 records
    result_dir = str(tmp_path / "results")
    scratch = make_endpoint(str(tmp_path / "scratch"), "sft")
    binding = {"completed": True, "artifact": {"sha256": {"x": "y"}}}
    assert eval_state(result_dir, "sft", binding) == "absent"
    commit_eval(result_dir, scratch, "sft", binding)
    assert eval_state(result_dir, "sft", binding) == "committed"
    assert os.path.exists(os.path.join(result_dir, "046_sft_eval_manifest.json"))
    assert not os.path.exists(os.path.join(result_dir, "042_sft_eval_manifest.json"))


def test_046_a1_conclusion_accepts_only_the_exact_stop(tmp_path):
    # one changed character turns the registered stop into a normal failure
    result_dir = str(tmp_path)
    (tmp_path / "046_sft_training.json").write_text(json.dumps({"completed": True}))
    status = {
        "completed": False,
        "cuda_oom": False,
        "error_type": "RuntimeError",
        "error_message": A1_ERROR,
    }
    (tmp_path / "grpo_worker_status.json").write_text(json.dumps(status))
    assert a1_state(result_dir)[0] == "pending"
    assert commit_a1_conclusion(result_dir)["outcome"] == "a1"

    os.remove(tmp_path / "046_a1_conclusion.json")
    status["error_message"] += "."
    (tmp_path / "grpo_worker_status.json").write_text(json.dumps(status))
    assert a1_state(result_dir)[0] == "failed"


def test_046_worker_dispatches_existing_training_and_evaluation_paths():
    source = open("scripts/s46_worker.py").read()
    assert "run_sft_rows" in source
    assert "run_grpo_rows" in source
    assert "run_eval_rows" in source
    assert "run_logprob_rows" in source
    assert "make_reward_funcs" not in source
    assert "GRPOTrainer(" not in source
    assert "SFTTrainer(" not in source
    assert "ensure_bound_merge" in source
    assert "expected_layers=48" not in source
    assert 'getattr(base.config, "n_layer", None)' in source


def test_046_reference_behaviour_requires_a_stable_disabled_path():
    # reject reference drift separately from an adapter that has no effect
    before = {
        "reference_logprobs": [[-1.0, -2.0]],
        "policy_logprobs": [[-1.0, -2.0]],
    }
    after = {
        "reference_logprobs": [[-1.0, -2.0]],
        "policy_logprobs": [[-0.8, -2.1]],
    }
    assert _reference_behaviour_report(before, after)["verified"] is True

    changed = copy.deepcopy(after)
    changed["reference_logprobs"][0][0] = -1.1
    with pytest.raises(IntegrityError, match="reference changed"):
        _reference_behaviour_report(before, changed)

    unchanged = copy.deepcopy(after)
    unchanged["policy_logprobs"] = copy.deepcopy(unchanged["reference_logprobs"])
    with pytest.raises(IntegrityError, match="no behavioural effect"):
        _reference_behaviour_report(before, unchanged)


def test_046_evaluation_source_matches_the_endpoint():
    _check_eval_source("base", None, "/tmp/base")
    _check_eval_source("sft", "/tmp/sft", None)
    _check_eval_source("grpo", "/tmp/grpo", None)
    with pytest.raises(IntegrityError):
        _check_eval_source("base", "/tmp/adapter", "/tmp/base")
    with pytest.raises(IntegrityError):
        _check_eval_source("grpo", None, None)


def test_046_grpo_adapter_is_bound_to_the_merged_start(tmp_path):
    cfg = load_config(config_path(0))
    cfg.output_root = str(tmp_path)
    merged = tmp_path / "merged"
    merged.mkdir()
    adapter = tmp_path / "grpo"
    adapter.mkdir()
    (adapter / "adapter_base.json").write_text(json.dumps(
        {"kind": "local", "value": "merged"}))
    record = verify_grpo_adapter_base(cfg, str(adapter), str(merged))
    assert record["verified"] is True
    assert record["resolved_path"] == os.path.realpath(merged)

    other = tmp_path / "other"
    other.mkdir()
    (adapter / "adapter_base.json").write_text(json.dumps(
        {"kind": "local", "value": "other"}))
    with pytest.raises(IntegrityError, match="adapter base differs"):
        verify_grpo_adapter_base(cfg, str(adapter), str(merged))


def test_046_merge_uses_the_registered_sft_storage_dtype(
        tmp_path, monkeypatch):
    from train import model_utils

    # merge from the same storage precision used during SFT
    cfg = load_config(config_path(0))
    cfg.output_root = str(tmp_path)
    (tmp_path / "base").mkdir()
    adapter = make_adapter(str(tmp_path / "adapter"))
    seen = []

    def stop_after_load(cfg_arg, source, dtype=None, device=None):
        seen.append((source, dtype, device))
        raise RuntimeError("stop after dtype check")

    monkeypatch.setattr(
        model_utils, "load_scratch_model_and_tokenizer", stop_after_load)
    with pytest.raises(RuntimeError, match="dtype check"):
        model_utils.merge_adapter(cfg, adapter, str(tmp_path / "merged"))
    assert seen == [(str(tmp_path / "base"), "float32", None)]


def test_046_merge_parity_uses_the_same_float32_storage(monkeypatch):
    torch = pytest.importorskip("torch")
    from train import model_utils

    cfg = load_config(config_path(0))
    calls = []

    class Batch(dict):
        def to(self, device):
            return self

    class Tokenizer:
        def __call__(self, text, return_tensors=None):
            return Batch(input_ids=torch.tensor([[1, 2]]))

    class Model:
        def eval(self):
            return self

        def __call__(self, **kwargs):
            return SimpleNamespace(logits=torch.tensor([[[1.0, 2.0]]]))

    def fake_load(cfg_arg, device, adapter_path=None, model_path=None,
                  dtype=None):
        calls.append({"adapter": adapter_path, "model": model_path,
                      "dtype": dtype})
        return Model(), Tokenizer()

    monkeypatch.setattr(model_utils, "load_model_and_tokenizer", fake_load)
    monkeypatch.setattr(torch.cuda, "empty_cache", lambda: None)
    report = _merge_logit_check(cfg, "adapter", "merged")
    assert report == {"max_abs_logit_error": 0.0,
                      "comparison_dtype": "float32"}
    assert calls == [
        {"adapter": "adapter", "model": None, "dtype": "float32"},
        {"adapter": None, "model": "merged", "dtype": "float32"},
    ]


def test_046_wandb_records_keep_interrupted_runs(tmp_path, monkeypatch):
    # both online run ids remain available after an interrupted resume
    from scripts.s42_worker import _capture_wandb_run, _wandb_records

    cfg = load_config(config_path(0))
    result_dir = tmp_path / arm_tag(0)
    result_dir.mkdir()
    for run_id in ("run_a", "run_b"):
        active = SimpleNamespace(
            id=run_id, project=cfg.log.wandb_project,
            name=f"{result_dir.name}_grpo", entity="student")
        monkeypatch.setitem(sys.modules, "wandb", SimpleNamespace(run=active))
        _capture_wandb_run(
            str(result_dir), cfg, 1, package="046", stage="grpo")
    rows = _wandb_records(
        str(result_dir), cfg, package="046", stage="grpo")
    assert [row["run_id"] for row in rows] == ["run_a", "run_b"]
