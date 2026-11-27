import copy
import ast
import json
import os
import re
import sys
import types
from types import SimpleNamespace

import pytest
import yaml

from checkpoint import _config_core, config_hash
from config import RunConfig, load_config
from enums import Condition
from scripts.s12_eval import IntegrityError, sha256
from scripts.s35_matched_grpo import (AGC_FULL_A1_ERROR, B15_DRIVE_RESUME,
                                      B15_PAIR_SETTLED, B15_RESUME_STATE,
                                      B15_SCALE_CHECK, B15_WANDB_RUNS,
                                      PARITY_TOLERANCE,
                                      PILOT_RUNGS_035,
                                      RUNGS_035, SCIENCE_RUNGS_035, apply_rung,
                                      ALLOCATOR_CONFIG, bind_execution,
                                      b15_arm_state, check_agc_authorisation,
                                      capture_wandb_run, check_pair_parity,
                                      commit_b15_a1_conclusion,
                                      commit_b15_drive_resume,
                                      commit_b15_pair_settled,
                                      commit_execution_lock,
                                      commit_execution_override,
                                      commit_pair_parity, commit_pilot_report,
                                      commit_scale_selection, config_path,
                                      execution_settings, pilot_passes,
                                      prepare_b15_scale_check,
                                      prepare_agc_lora_control, prepare_agc_science,
                                      prepare_allocator_retry,
                                      safe_close_checkpoints)


def _module(name, **attrs):
    module = types.ModuleType(name)
    for key, value in attrs.items():
        setattr(module, key, value)
    return module


def test_035_configs_cover_one_matched_pair_per_seed_and_scale():
    configs = []
    for scale in ("B3", "B15"):
        for seed in (0, 1, 2):
            lora = load_config(config_path(scale, seed, "lora"))
            full = load_config(config_path(scale, seed, "full"))
            assert lora.scale.name == full.scale.name == scale
            assert lora.train.seed == full.train.seed == seed
            assert lora.condition == Condition.MATCHED_LORA_GRPO
            assert full.condition == Condition.MATCHED_FULL_GRPO
            assert lora.train.grpo_parameter_mode == "lora"
            assert full.train.grpo_parameter_mode == "full"
            assert lora.train.grpo_precision_mode == full.train.grpo_precision_mode == "fp32_master_bf16"
            assert lora.train.a2_action == full.train.a2_action == "observe"
            assert lora.exec.allow_oom_fallback is full.exec.allow_oom_fallback is False
            configs.extend((lora, full))
    assert len(configs) == 12


def test_035_pair_configs_differ_only_in_condition_and_parameter_mode():
    for scale in ("B3", "B15"):
        for seed in (0, 1, 2):
            with open(config_path(scale, seed, "lora")) as f:
                lora = yaml.safe_load(f)
            with open(config_path(scale, seed, "full")) as f:
                full = yaml.safe_load(f)
            assert lora.pop("condition") == "MATCHED_LORA_GRPO"
            assert full.pop("condition") == "MATCHED_FULL_GRPO"
            assert lora["train"].pop("grpo_parameter_mode") == "lora"
            assert full["train"].pop("grpo_parameter_mode") == "full"
            assert lora == full


def test_035_seed_configs_differ_only_in_seed():
    for scale in ("B3", "B15"):
        for mode in ("lora", "full"):
            rows = []
            for seed in (0, 1, 2):
                with open(config_path(scale, seed, mode)) as f:
                    row = yaml.safe_load(f)
                assert row["train"].pop("seed") == seed
                rows.append(row)
            assert rows[0] == rows[1] == rows[2]


def test_matched_fields_are_hashed_but_defaults_keep_the_old_hash():
    cfg = RunConfig()
    base = config_hash(cfg)
    assert base == config_hash(cfg, ("grpo_parameter_mode", "grpo_precision_mode"))
    cfg.experiment = "035"
    cfg.exploratory = True
    cfg.condition = Condition.MATCHED_LORA_GRPO
    cfg.train.grpo_precision_mode = "fp32_master_bf16"
    cfg.train.a2_action = "observe"
    assert config_hash(cfg) != base
    full = copy.deepcopy(cfg)
    full.condition = Condition.MATCHED_FULL_GRPO
    full.train.grpo_parameter_mode = "full"
    assert config_hash(full) != config_hash(cfg)


def test_historical_config_cores_do_not_gain_the_new_default_fields():
    # keep archived checkpoint hashes valid when the matched fields stay neutral
    for path in ("configs/003_colab_full_b3_seed0.yaml",
                 "configs/009_colab_full_b15_seed0.yaml",
                 "configs/031_colab_b7_n4_a2_record.yaml"):
        train = _config_core(load_config(path))["train"]
        assert "grpo_parameter_mode" not in train
        assert "grpo_precision_mode" not in train


def test_matched_modes_are_rejected_outside_035(tmp_path):
    path = tmp_path / "bad.yaml"
    path.write_text("""
experiment: '034'
exploratory: true
condition: MATCHED_FULL_GRPO
train:
  grpo_parameter_mode: full
  grpo_precision_mode: fp32_master_bf16
  a2_action: observe
""")
    with pytest.raises(ValueError, match="reserved for experiment 035"):
        load_config(path)


def test_matched_trainer_switches_only_precision_and_peft(monkeypatch):
    import train.grpo as grpo

    captured = []

    class FakeParameter:
        def __init__(self):
            self.requires_grad = True

        def requires_grad_(self, value):
            self.requires_grad = value

    class FakeReference:
        def __init__(self):
            self.params = [FakeParameter(), FakeParameter()]

        def parameters(self):
            return self.params

    class FakeTrainer:
        def __init__(self, **kwargs):
            self.kwargs = kwargs
            self.ref_model = FakeReference() if kwargs["peft_config"] is None else None
            captured.append(kwargs)

    monkeypatch.setitem(sys.modules, "trl", _module(
        "trl", GRPOConfig=lambda **kwargs: SimpleNamespace(**kwargs),
        GRPOTrainer=FakeTrainer))
    monkeypatch.setattr(grpo, "make_reward_funcs", lambda cfg: ["reward"])
    monkeypatch.setattr(grpo, "make_lora", lambda cfg: "lora-config")

    lora = load_config("configs/035_colab_b3_lora_seed0.yaml")
    full = load_config("configs/035_colab_b3_full_seed0.yaml")
    grpo.make_grpo_trainer(lora, [], "/start", "/out", [], "lora", 8, 1, False)
    full_trainer = grpo.make_grpo_trainer(
        full, [], "/start", "/out", [], "full", 8, 1, False)

    assert captured[0]["peft_config"] == "lora-config"
    assert captured[1]["peft_config"] is None
    assert captured[0]["args"].model_init_kwargs == {"dtype": "float32"}
    assert captured[1]["args"].model_init_kwargs == {"dtype": "float32"}
    assert captured[0]["args"].bf16 is captured[1]["args"].bf16 is True
    assert all(not param.requires_grad for param in full_trainer.ref_model.parameters())


def test_registered_rungs_keep_the_group_and_apply_without_fallback():
    assert list(RUNGS_035) == ["A", "B", "C", "D"]
    assert PILOT_RUNGS_035 == {
        "A_GC": {"batch": 8, "accum": 1, "checkpointing": True}}
    assert SCIENCE_RUNGS_035 == ("A", "B", "C", "D")
    assert [(row["batch"], row["accum"], row["checkpointing"])
            for row in RUNGS_035.values()] == [
                (8, 1, False), (4, 2, False), (4, 2, True), (2, 4, True)]
    cfg = load_config("configs/035_colab_b3_full_seed0.yaml")
    for rung, row in {**RUNGS_035, **PILOT_RUNGS_035}.items():
        assert execution_settings(rung) == row
        assert apply_rung(cfg, rung) == row
        assert cfg.exec.grpo_batch_size * cfg.exec.grpo_grad_accum == 8
        assert cfg.exec.allow_oom_fallback is False


def test_execution_binding_rejects_rung_drift_on_resume(tmp_path):
    cfg = load_config("configs/035_colab_b3_full_seed0.yaml")
    start = {"start": "/fixed", "start_sha256": {"model": "abc"}}
    environment = {"gpu_device": "A100", "versions": {"torch": "fixed"},
                   "allocator_config": None}
    bind_execution(str(tmp_path), cfg, start, "A", "train", environment)
    bind_execution(str(tmp_path), cfg, start, "A", "train", environment)
    with pytest.raises(IntegrityError, match="committed record changed"):
        bind_execution(str(tmp_path), cfg, start, "B", "train", environment)
    changed = copy.deepcopy(environment)
    changed["versions"]["torch"] = "different"
    with pytest.raises(IntegrityError, match="committed record changed"):
        bind_execution(str(tmp_path), cfg, start, "A", "train", changed)


def test_worker_reads_the_actual_checkpoint_interval(tmp_path):
    from scripts.s35_worker import _latest_checkpoint_step

    assert _latest_checkpoint_step(str(tmp_path)) == 0
    (tmp_path / "checkpoint-30").mkdir()
    (tmp_path / "checkpoint-150").mkdir()
    (tmp_path / "checkpoint-bad").mkdir()
    assert _latest_checkpoint_step(str(tmp_path)) == 150


def _pilot_report(changed=True, fraction=0.9):
    return {
        "completed_updates": 30,
        "finite": True,
        "storage_dtype": "float32",
        "compute_dtype": "bfloat16",
        "gradient_dtype": "float32",
        "optimizer_state_dtypes": ["float32"],
        "checkpoint_bytes": 1024,
        "peak_reserved_fraction": fraction,
        "precision_audit": {"accelerator_mixed_precision": "bf16",
                            "reference_gate": {"verified": True}, "module_changes": {
            "attention": {"changed_elements": 1 if changed else 0}}},
    }


def test_precision_audit_can_wait_for_the_first_nonzero_grpo_update():
    torch = pytest.importorskip("torch")
    from scripts.s35_matched_grpo import PrecisionAudit

    parameter = torch.nn.Parameter(torch.ones(4, dtype=torch.float32))
    audit = PrecisionAudit(allow_initial_zero_gradient=True)
    parameter.grad = torch.zeros_like(parameter)
    audit._capture_gradient_step((parameter,), 1)
    assert audit.gradient["nonzero"] == 0
    assert audit.gradient["updates_checked"] == 1
    assert audit.gradient["first_nonzero_update"] is None

    parameter.grad = torch.ones_like(parameter)
    audit._capture_gradient_step((parameter,), 2)
    assert audit.gradient["nonzero"] == 4
    assert audit.gradient["updates_checked"] == 2
    assert audit.gradient["first_nonzero_update"] == 2

    strict = PrecisionAudit()
    parameter.grad = torch.zeros_like(parameter)
    strict._capture_gradient_step((parameter,), 1)
    parameter.grad = torch.ones_like(parameter)
    strict._capture_gradient_step((parameter,), 2)
    assert strict.gradient["nonzero"] == 0
    assert strict.gradient["updates_checked"] == 1


def test_pilot_verdict_uses_dtype_change_and_memory_only(tmp_path):
    good = _pilot_report()
    good.update({"reward": -99, "accuracy": 0})
    assert pilot_passes(good)
    assert not pilot_passes(_pilot_report(changed=False))
    assert not pilot_passes(_pilot_report(fraction=0.951))
    path = tmp_path / "pilot.json"
    record = commit_pilot_report(str(path), good)
    assert record["passed"] is True and record["scientific_metrics_used"] is False


def test_scale_selection_uses_3b_first_and_is_immutable(tmp_path):
    root = tmp_path / "results" / "035_technical" / "pilots" / "B3" / "B"
    root.mkdir(parents=True)
    (root / "pilot_report.json").write_text(json.dumps({"passed": True}))
    selection = commit_scale_selection(str(tmp_path))
    assert selection == {"completed": True, "selected_scale": "B3",
                         "pilot_rung": "B", "selection_basis": "technical_pilot_only"}
    (root / "pilot_report.json").write_text(json.dumps({"passed": False}))
    assert commit_scale_selection(str(tmp_path)) == selection


def test_science_override_keeps_the_pilot_selection_and_disables_fallback(tmp_path):
    root = tmp_path / "results" / "035_technical"
    for rung, passed, peak in (("A", False, 0.993), ("C", True, 0.911)):
        path = root / "pilots" / "B3" / rung
        path.mkdir(parents=True)
        (path / "pilot_report.json").write_text(json.dumps({
            "passed": passed, "peak_reserved_fraction": peak}))
    selection = {"completed": True, "selected_scale": "B3", "pilot_rung": "C",
                 "selection_basis": "technical_pilot_only"}
    record = commit_execution_override(str(tmp_path), selection, "A")
    assert record["pilot_selected_rung"] == "C"
    assert record["requested_rung"] == "A"
    assert record["automatic_fallback"] is False
    (root / "execution_lock.json").write_text("{}")
    assert commit_execution_override(str(tmp_path), selection, "A") == record
    with pytest.raises(IntegrityError, match="differs from plan v7"):
        commit_execution_override(str(tmp_path), selection, "B")


def _failed_a_attempt(root, cuda_oom=True, allocator=None):
    tag = "035_COLAB_FULL_B3_matched_full_grpo_seed0"
    result = root / "results" / tag
    checkpoint = root / "checkpoints" / tag
    parity = root / "results" / "035_technical" / "parity" / "B3" / "seed0" / "A"
    result.mkdir(parents=True)
    checkpoint.mkdir(parents=True)
    parity.mkdir(parents=True)
    (result / "worker_status.json").write_text(json.dumps({
        "completed": False, "task": "train", "rung": "A",
        "cuda_oom": cuda_oom, "allocator_config": allocator}))
    (result / "035_execution_binding.json").write_text(json.dumps({
        "rung": "A", "environment": {"allocator_config": allocator}}))
    (checkpoint / "partial.txt").write_text("failed")
    (parity / "pair_parity.json").write_text("{}")
    return result, checkpoint, parity


def test_allocator_retry_archives_the_failed_a_attempt_and_blocks_a_second_oom(tmp_path):
    selection = {"selected_scale": "B3", "pilot_rung": "C"}
    override = {"requested_rung": "A", "automatic_fallback": False}
    result, checkpoint, parity = _failed_a_attempt(tmp_path)
    record = prepare_allocator_retry(str(tmp_path), selection, override)
    archive = tmp_path / "failed_attempts" / "035" / "B3_seed0_full_A_default_allocator"
    assert record["allocator_config"] == ALLOCATOR_CONFIG
    assert record["clean_restart"] is True
    assert not result.exists() and not checkpoint.exists() and not parity.exists()
    assert (archive / "results" / "worker_status.json").exists()
    assert (archive / "checkpoints" / "partial.txt").exists()
    assert (archive / "parity" / "pair_parity.json").exists()
    assert prepare_allocator_retry(str(tmp_path), selection, override) == record

    active = tmp_path / "results" / "035_COLAB_FULL_B3_matched_full_grpo_seed0"
    active.mkdir()
    (active / "worker_status.json").write_text(json.dumps({
        "completed": False, "task": "train", "rung": "A", "cuda_oom": True,
        "allocator_config": ALLOCATOR_CONFIG}))
    with pytest.raises(IntegrityError, match="allocator retry already OOMed"):
        prepare_allocator_retry(str(tmp_path), selection, override)


def test_allocator_retry_rejects_a_non_oom_attempt(tmp_path):
    selection = {"selected_scale": "B3", "pilot_rung": "C"}
    override = {"requested_rung": "A", "automatic_fallback": False}
    _failed_a_attempt(tmp_path, cuda_oom=False)
    with pytest.raises(IntegrityError, match="requires the recorded rung-A OOM"):
        prepare_allocator_retry(str(tmp_path), selection, override)


def _passed_agc_pilot(root):
    path = root / "results" / "035_technical" / "pilots" / "B3" / "A_GC"
    path.mkdir(parents=True)
    report = _pilot_report(fraction=0.848)
    report.update({
        "passed": True,
        "execution": PILOT_RUNGS_035["A_GC"],
        "allocator_config": ALLOCATOR_CONFIG,
        "pilot_weights_retained": False,
        "scientific_metrics_used": False,
    })
    (path / "pilot_report.json").write_text(json.dumps(report))


def test_agc_science_archives_the_second_a_failure_and_limits_the_stage(tmp_path):
    selection = {"selected_scale": "B3", "pilot_rung": "C"}
    override = {"requested_rung": "A"}
    retry = {"rung": "A", "allocator_config": ALLOCATOR_CONFIG}
    _passed_agc_pilot(tmp_path)
    result, checkpoint, parity = _failed_a_attempt(
        tmp_path, allocator=ALLOCATOR_CONFIG)

    record = prepare_agc_science(str(tmp_path), selection, override, retry)
    archive = (tmp_path / "failed_attempts" / "035"
               / "B3_seed0_full_A_expandable_allocator")
    assert record["rung"] == "A_GC"
    assert record["execution"] == PILOT_RUNGS_035["A_GC"]
    assert record["authorised_training"] == ["full_seed0"]
    assert not result.exists() and not checkpoint.exists() and not parity.exists()
    assert (archive / "results" / "worker_status.json").exists()
    assert prepare_agc_science(str(tmp_path), selection, override, retry) == record

    full = load_config("configs/035_colab_b3_full_seed0.yaml")
    full.output_root = str(tmp_path)
    lora = load_config("configs/035_colab_b3_lora_seed0.yaml")
    lora.output_root = str(tmp_path)
    assert check_agc_authorisation(full, "A_GC", "train") == record
    assert check_agc_authorisation(full, "A_GC", "eval") == record
    assert check_agc_authorisation(lora, "A_GC", "parity") == record
    with pytest.raises(IntegrityError, match="outside the staged run"):
        check_agc_authorisation(lora, "A_GC", "train")

    record_path = tmp_path / "results" / "035_technical" / "035_agc_science.json"
    changed = copy.deepcopy(record)
    changed["authorised_training"] = ["lora_seed0", "full_seed0"]
    record_path.write_text(json.dumps(changed))
    with pytest.raises(IntegrityError, match="science record changed"):
        check_agc_authorisation(full, "A_GC", "train")
    record_path.write_text(json.dumps(record))

    (archive / "results" / "worker_status.json").write_text("{}")
    with pytest.raises(IntegrityError, match="archive changed"):
        check_agc_authorisation(full, "A_GC", "train")


def test_agc_science_rejects_a_failed_pilot(tmp_path):
    selection = {"selected_scale": "B3", "pilot_rung": "C"}
    override = {"requested_rung": "A"}
    retry = {"rung": "A", "allocator_config": ALLOCATOR_CONFIG}
    _passed_agc_pilot(tmp_path)
    pilot = (tmp_path / "results" / "035_technical" / "pilots" / "B3"
             / "A_GC" / "pilot_report.json")
    row = json.loads(pilot.read_text())
    row["passed"] = False
    pilot.write_text(json.dumps(row))
    _failed_a_attempt(tmp_path, allocator=ALLOCATOR_CONFIG)
    with pytest.raises(IntegrityError, match="did not pass"):
        prepare_agc_science(str(tmp_path), selection, override, retry)


def test_execution_lock_accepts_the_authorised_agc_setting(tmp_path):
    (tmp_path / "results" / "035_technical").mkdir(parents=True)
    training_path = tmp_path / "training.json"
    training_path.write_text("{}")
    training = {"completed": True, "scale": "B3", "seed": 0,
                "parameter_mode": "full", "rung": "A_GC",
                "execution": PILOT_RUNGS_035["A_GC"],
                "record_path": str(training_path)}
    record = commit_execution_lock(str(tmp_path), "B3", "A_GC", training)
    assert record["rung"] == "A_GC"
    assert record["execution"] == PILOT_RUNGS_035["A_GC"]
    changed = copy.deepcopy(training)
    changed["rung"] = "A"
    with pytest.raises(IntegrityError, match="wrong full seed-0 record"):
        commit_execution_lock(str(tmp_path), "B3", "A_GC", changed)


def _parity(mode, offset=0.0):
    values = [[offset, 0.1 + offset], [0.2 + offset, 0.3 + offset]]
    return {"completed": True, "parameter_mode": mode, "scale": "B3", "seed": 0,
            "rung": "A", "beta": 0.04, "start_record": {"start": "/same"},
            "storage_dtype": "float32", "compute_dtype": "bfloat16",
            "reference_structure": {"verified": True},
            "versions": {"torch": "same"}, "gpu": {"device": "same"},
            "allocator_config": None,
            "policy_logprobs": values, "reference_logprobs": values}


def _committed_endpoint(result, training):
    names = ("grpo_preds.jsonl", "grpo_eval.json",
             "grpo_logprob_preds.jsonl", "grpo_logprob.json")
    result.mkdir(parents=True, exist_ok=True)
    for name in names:
        (result / name).write_text(name)
    manifest = {
        "completed": True, "training_record": training,
        "sha256": {name: sha256(result / name) for name in names},
    }
    (result / "035_eval_manifest.json").write_text(json.dumps(manifest))


def _b15_registered_state(root):
    _agc_full_a1_state(root)
    control = prepare_agc_lora_control(str(root))
    b3_lora = root / "results" / "035_COLAB_FULL_B3_matched_lora_grpo_seed0"
    training = {"completed": True, "parameter_mode": "lora"}
    b3_lora.mkdir(parents=True)
    (b3_lora / "035_training_complete.json").write_text(json.dumps(training))
    _committed_endpoint(b3_lora, training)
    selection = {"completed": True, "selected_scale": "B3", "pilot_rung": "C",
                 "selection_basis": "technical_pilot_only"}
    technical = root / "results" / "035_technical"
    (technical / "035_scale_selection.json").write_text(json.dumps(selection))
    start_path = technical / "starts" / "B15" / "seed0.json"
    start_path.parent.mkdir(parents=True)
    start_path.write_text(json.dumps({
        "experiment": "035", "scale": "B15", "seed": 0,
        "source_tag": "009_COLAB_FULL_B15_sft_grpo_seed0",
        "start": "/fixed", "start_sha256": {"model": "abc"},
    }))
    return prepare_b15_scale_check(str(root), str(start_path)), control


def _passed_b15_pilot(root):
    path = root / "results" / "035_technical" / "pilots" / "B15" / "A_GC"
    path.mkdir(parents=True)
    report = _pilot_report(fraction=0.7)
    report.update({
        "passed": True, "scale": "B15", "seed": 0, "rung": "A_GC",
        "execution": PILOT_RUNGS_035["A_GC"],
        "allocator_config": ALLOCATOR_CONFIG,
        "pilot_weights_retained": False, "scientific_metrics_used": False,
    })
    (path / "pilot_report.json").write_text(json.dumps(report))


def _b15_parity(root):
    path = (root / "results" / "035_technical" / "parity" / "B15"
            / "seed0" / "A_GC" / "pair_parity.json")
    path.parent.mkdir(parents=True)
    lora = _parity("lora")
    full = _parity("full")
    start = json.loads((root / "results" / "035_technical" / "starts"
                        / "B15" / "seed0.json").read_text())
    for mode, row in (("lora", lora), ("full", full)):
        row.update({"scale": "B15", "rung": "A_GC",
                    "execution": PILOT_RUNGS_035["A_GC"],
                    "allocator_config": ALLOCATOR_CONFIG,
                    "config_hash": config_hash(
                        load_config(config_path("B15", 0, mode))),
                    "start_record": start})
    return commit_pair_parity(str(path), lora, full)


def test_pair_parity_commits_within_tolerance_and_rejects_drift(tmp_path):
    # compare both policy and reference distributions before either arm trains
    path = tmp_path / "pair.json"
    record = commit_pair_parity(str(path), _parity("lora"), _parity("full", 0.0001))
    assert record["completed"] is True
    assert max(record["checks"].values()) <= PARITY_TOLERANCE
    with pytest.raises(IntegrityError, match="parity tolerance"):
        commit_pair_parity(str(tmp_path / "bad.json"), _parity("lora"), _parity("full", 0.01))


def _agc_full_a1_state(root):
    selection = {"selected_scale": "B3", "pilot_rung": "C"}
    override = {"requested_rung": "A"}
    retry = {"rung": "A", "allocator_config": ALLOCATOR_CONFIG}
    _passed_agc_pilot(root)
    _failed_a_attempt(root, allocator=ALLOCATOR_CONFIG)
    science = prepare_agc_science(str(root), selection, override, retry)

    result = root / "results" / "035_COLAB_FULL_B3_matched_full_grpo_seed0"
    result.mkdir(parents=True)
    status = {
        "completed": False, "task": "train", "rung": "A_GC",
        "execution": PILOT_RUNGS_035["A_GC"],
        "allocator_config": ALLOCATOR_CONFIG,
        "cuda_oom": False, "error_type": "RuntimeError",
        "error_message": AGC_FULL_A1_ERROR,
    }
    full_cfg = load_config("configs/035_colab_b3_full_seed0.yaml")
    binding = {
        "scale": "B3", "seed": 0, "parameter_mode": "full", "task": "train",
        "rung": "A_GC", "execution": PILOT_RUNGS_035["A_GC"],
        "config_hash": config_hash(full_cfg), "start_record": {"start": "/same"},
        "environment": {"allocator_config": ALLOCATOR_CONFIG},
    }
    (result / "worker_status.json").write_text(json.dumps(status))
    (result / "035_execution_binding.json").write_text(json.dumps(binding))

    parity_dir = root / "results" / "035_technical" / "parity" / "B3" / "seed0" / "A_GC"
    parity_dir.mkdir(parents=True)
    lora = _parity("lora")
    full = _parity("full")
    for row in (lora, full):
        row["rung"] = "A_GC"
        row["execution"] = PILOT_RUNGS_035["A_GC"]
        row["allocator_config"] = ALLOCATOR_CONFIG
    commit_pair_parity(str(parity_dir / "pair_parity.json"), lora, full)
    return science, result


def test_agc_lora_control_binds_the_full_a1_stop_without_rewriting_science(tmp_path):
    science, _ = _agc_full_a1_state(tmp_path)
    science_path = tmp_path / "results" / "035_technical" / "035_agc_science.json"
    before = science_path.read_bytes()

    record = prepare_agc_lora_control(str(tmp_path))
    assert record["parameter_mode"] == "lora"
    assert record["lora_config_hash"] == config_hash(
        load_config("configs/035_colab_b3_lora_seed0.yaml"))
    assert record["authorised_training"] == ["lora_seed0"]
    assert record["authorised_evaluation"] == ["lora_seed0"]
    assert science_path.read_bytes() == before
    assert json.loads(before) == science

    lora = load_config("configs/035_colab_b3_lora_seed0.yaml")
    lora.output_root = str(tmp_path)
    full = load_config("configs/035_colab_b3_full_seed0.yaml")
    full.output_root = str(tmp_path)
    assert check_agc_authorisation(lora, "A_GC", "train") == record
    assert check_agc_authorisation(lora, "A_GC", "eval") == record
    with pytest.raises(IntegrityError, match="closed after A1"):
        check_agc_authorisation(full, "A_GC", "train")


def test_agc_lora_control_halts_if_the_full_failure_or_parity_changes(tmp_path):
    _, result = _agc_full_a1_state(tmp_path)
    prepare_agc_lora_control(str(tmp_path))
    status_path = result / "worker_status.json"
    original_status = status_path.read_text()
    status = json.loads(original_status)
    status["error_message"] = "different"
    status_path.write_text(json.dumps(status))

    lora = load_config("configs/035_colab_b3_lora_seed0.yaml")
    lora.output_root = str(tmp_path)
    with pytest.raises(IntegrityError, match="control record changed"):
        check_agc_authorisation(lora, "A_GC", "train")

    status_path.write_text(original_status)
    parity_path = (tmp_path / "results" / "035_technical" / "parity"
                   / "B3" / "seed0" / "A_GC" / "pair_parity.json")
    parity = json.loads(parity_path.read_text())
    parity["changed"] = True
    parity_path.write_text(json.dumps(parity))
    with pytest.raises(IntegrityError, match="control record changed"):
        check_agc_authorisation(lora, "A_GC", "train")


def _write_b15_a1(root, mode):
    result = root / "results" / f"035_COLAB_FULL_B15_matched_{mode}_grpo_seed0"
    result.mkdir(parents=True)
    cfg = load_config(config_path("B15", 0, mode))
    cfg.output_root = str(root)
    start = json.loads((root / "results" / "035_technical" / "starts"
                        / "B15" / "seed0.json").read_text())
    bind_execution(
        str(result), cfg, start, "A_GC", "train",
        {"gpu_device": "A100", "versions": {},
         "allocator_config": ALLOCATOR_CONFIG})
    run_id = f"run_{mode}"
    tracking = result / B15_WANDB_RUNS
    tracking.mkdir()
    binding_path = result / "035_execution_binding.json"
    (tracking / f"{run_id}.json").write_text(json.dumps({
        "completed": True,
        "experiment": "035b",
        "scale": "B15",
        "seed": 0,
        "parameter_mode": mode,
        "rung": "A_GC",
        "project": cfg.log.wandb_project,
        "run_id": run_id,
        "run_name": f"{result.name}_grpo",
        "entity": "test",
        "config_hash": config_hash(cfg),
        "execution_binding_sha256": sha256(str(binding_path)),
        "first_recorded_step": 1,
    }))
    status = {
        "completed": False, "task": "train", "rung": "A_GC",
        "execution": PILOT_RUNGS_035["A_GC"],
        "allocator_config": ALLOCATOR_CONFIG,
        "cuda_oom": False, "error_type": "RuntimeError",
        "error_message": AGC_FULL_A1_ERROR,
    }
    (result / "worker_status.json").write_text(json.dumps(status))
    return commit_b15_a1_conclusion(str(root), mode)


def _write_b15_drive_failure(root):
    mode = "lora"
    result = root / "results" / f"035_COLAB_FULL_B15_matched_{mode}_grpo_seed0"
    output = root / "checkpoints" / f"035_COLAB_FULL_B15_matched_{mode}_grpo_seed0" / "grpo"
    checkpoint = output / "checkpoint-100"
    result.mkdir(parents=True)
    checkpoint.mkdir(parents=True)
    cfg = load_config(config_path("B15", 0, mode))
    cfg.output_root = str(root)
    start = json.loads((root / "results" / "035_technical" / "starts"
                        / "B15" / "seed0.json").read_text())
    bind_execution(
        str(result), cfg, start, "A_GC", "train",
        {"gpu_device": "A100", "versions": {},
         "allocator_config": ALLOCATOR_CONFIG})
    (output / "config_hash.txt").write_text(config_hash(cfg))
    for name in B15_RESUME_STATE:
        (checkpoint / name).write_bytes(name.encode())
    history = [
        {"step": step, "rewards/accuracy_reward/mean": 0.5}
        for step in range(1, 101)
    ]
    (checkpoint / "trainer_state.json").write_text(json.dumps({
        "global_step": 100, "max_steps": 500, "log_history": history,
    }))

    binding_path = result / "035_execution_binding.json"
    tracking = result / B15_WANDB_RUNS
    tracking.mkdir()
    run_id = "run_lora_drive"
    (tracking / f"{run_id}.json").write_text(json.dumps({
        "completed": True, "experiment": "035b", "scale": "B15",
        "seed": 0, "parameter_mode": "lora", "rung": "A_GC",
        "project": cfg.log.wandb_project, "run_id": run_id,
        "run_name": f"{result.name}_grpo", "entity": "test",
        "config_hash": config_hash(cfg),
        "execution_binding_sha256": sha256(str(binding_path)),
        "first_recorded_step": 1,
    }))
    status = {
        "completed": False, "task": "train", "rung": "A_GC",
        "execution": PILOT_RUNGS_035["A_GC"],
        "allocator_config": ALLOCATOR_CONFIG, "cuda_oom": False,
        "error_type": "FailedPreconditionError",
        "error_message": (
            f"/content/drive/MyDrive/inm363/checkpoints/{result.name}/grpo/runs/"
            "events.out.tfevents; Transport endpoint is not connected"),
        "start_step": 0, "end_step": 100, "completed_updates": 100,
    }
    (result / "worker_status.json").write_text(json.dumps(status))
    attempt = {**status, "label": "035b_B15_seed0_lora_A_GC", "returncode": 1}
    (result / "035b_attempts.jsonl").write_text(json.dumps(attempt) + "\n")
    return result, output, checkpoint, status


def test_b15_worker_requires_the_scale_check_record(tmp_path):
    cfg = load_config("configs/035_colab_b15_full_seed0.yaml")
    cfg.output_root = str(tmp_path)
    with pytest.raises(IntegrityError, match="scale-check record is missing"):
        check_agc_authorisation(cfg, "A_GC", "pilot")


def test_b15_wandb_record_binds_the_active_run(monkeypatch, tmp_path):
    run = SimpleNamespace(id="run_abc", project="inm363-grpo",
                          name="expected_grpo", entity="test")
    monkeypatch.setitem(sys.modules, "wandb", _module("wandb", run=run))
    context = {
        "scale": "B15", "seed": 0, "parameter_mode": "full",
        "rung": "A_GC", "project": "inm363-grpo",
        "run_name": "expected_grpo", "config_hash": "config",
        "execution_binding_sha256": "binding",
    }
    first = capture_wandb_run(str(tmp_path), context, 1)
    second = capture_wandb_run(str(tmp_path), context, 2)
    assert first == second
    assert first["run_id"] == "run_abc"
    assert first["first_recorded_step"] == 1
    run.name = "different"
    with pytest.raises(IntegrityError, match="identity differs"):
        capture_wandb_run(str(tmp_path), context, 3)
    with open("scripts/s35_worker.py") as f:
        worker = f.read()
    assert "wandb_record_dir=wandb_record_dir" in worker
    assert "wandb_context=wandb_context" in worker


def test_b15_scale_check_binds_the_closed_b3_package(tmp_path):
    record, _ = _b15_registered_state(tmp_path)
    assert record["fixed_roster"] == ["full_seed0", "lora_seed0"]
    assert record["execution"] == PILOT_RUNGS_035["A_GC"]
    assert (tmp_path / "results" / "035_technical" / B15_SCALE_CHECK).exists()

    b15 = load_config("configs/035_colab_b15_full_seed0.yaml")
    b15.output_root = str(tmp_path)
    assert check_agc_authorisation(b15, "A_GC", "pilot") == record
    changed = copy.deepcopy(b15)
    changed.train.grpo_lr = 2.0e-5
    with pytest.raises(IntegrityError, match="worker config changed"):
        check_agc_authorisation(changed, "A_GC", "pilot")
    b3 = load_config("configs/035_colab_b3_lora_seed0.yaml")
    b3.output_root = str(tmp_path)
    with pytest.raises(IntegrityError, match="B3 package is closed"):
        check_agc_authorisation(b3, "A_GC", "train")
    with pytest.raises(IntegrityError, match="only authorises A_GC"):
        check_agc_authorisation(b15, "C", "pilot")


def test_b15_pair_runs_in_order_and_a1_arms_get_no_endpoint(tmp_path):
    _b15_registered_state(tmp_path)
    full = load_config("configs/035_colab_b15_full_seed0.yaml")
    full.output_root = str(tmp_path)
    lora = load_config("configs/035_colab_b15_lora_seed0.yaml")
    lora.output_root = str(tmp_path)

    with pytest.raises(IntegrityError, match="pilot report is missing"):
        check_agc_authorisation(full, "A_GC", "train")
    _passed_b15_pilot(tmp_path)
    with pytest.raises(IntegrityError, match="pair parity is missing"):
        check_agc_authorisation(full, "A_GC", "train")
    parity = _b15_parity(tmp_path)
    assert check_pair_parity(
        str(tmp_path / "results" / "035_technical" / "parity" / "B15"
            / "seed0" / "A_GC" / "pair_parity.json"), "B15") == parity
    assert check_agc_authorisation(full, "A_GC", "train")["scale"] == "B15"
    with pytest.raises(IntegrityError, match="full outcome must settle"):
        check_agc_authorisation(lora, "A_GC", "train")

    _write_b15_a1(tmp_path, "full")
    assert b15_arm_state(str(tmp_path), "full")["state"] == "a1"
    assert check_agc_authorisation(lora, "A_GC", "train")["scale"] == "B15"
    _write_b15_a1(tmp_path, "lora")
    settled = commit_b15_pair_settled(str(tmp_path))
    assert settled["outcomes"]["full"]["state"] == "a1"
    assert settled["outcomes"]["lora"]["state"] == "a1"
    assert settled["outcomes"]["full"]["wandb_runs"][0]["run_id"] == "run_full"
    assert settled["outcomes"]["lora"]["wandb_runs"][0]["run_id"] == "run_lora"
    assert (tmp_path / "results" / "035_technical" / B15_PAIR_SETTLED).exists()
    with pytest.raises(IntegrityError, match="has no endpoint"):
        check_agc_authorisation(full, "A_GC", "eval")


def test_b15_lora_drive_failure_resumes_once_from_complete_checkpoint(tmp_path):
    _b15_registered_state(tmp_path)
    _passed_b15_pilot(tmp_path)
    _b15_parity(tmp_path)
    _write_b15_a1(tmp_path, "full")
    result, _, checkpoint, _ = _write_b15_drive_failure(tmp_path)

    pending = b15_arm_state(str(tmp_path), "lora")
    assert pending["state"] == "drive_interrupted_pending"
    record = commit_b15_drive_resume(str(tmp_path))
    assert record["report_to"] == ["wandb"]
    assert record["source_checkpoint"]["step"] == 100
    assert (tmp_path / "results" / "035_technical" / B15_DRIVE_RESUME).exists()
    resumed = b15_arm_state(str(tmp_path), "lora")
    assert resumed["state"] == "interrupted"
    assert resumed["drive_resume"] == record

    cfg = load_config(config_path("B15", 0, "lora"))
    cfg.output_root = str(tmp_path)
    assert check_agc_authorisation(cfg, "A_GC", "train")["scale"] == "B15"

    os.remove(checkpoint / "optimizer.pt")
    with pytest.raises(IntegrityError, match="continuation state is missing"):
        b15_arm_state(str(tmp_path), "lora")


def test_b15_lora_drive_resume_rejects_a_second_drive_failure(tmp_path):
    _b15_registered_state(tmp_path)
    _passed_b15_pilot(tmp_path)
    _b15_parity(tmp_path)
    _write_b15_a1(tmp_path, "full")
    result, _, _, status = _write_b15_drive_failure(tmp_path)
    commit_b15_drive_resume(str(tmp_path))
    with open(result / "035b_attempts.jsonl", "a") as f:
        f.write(json.dumps({**status, "label": "retry", "returncode": 1}) + "\n")
    with pytest.raises(IntegrityError, match="permits one LoRA Drive interruption"):
        b15_arm_state(str(tmp_path), "lora")


def test_safe_close_writes_the_record_and_removes_only_checkpoints(tmp_path):
    result = tmp_path / "result"
    output = tmp_path / "output"
    checkpoint = output / "checkpoint-500"
    unrelated = output / "final"
    checkpoint.mkdir(parents=True)
    unrelated.mkdir(parents=True)
    (checkpoint / "state.bin").write_bytes(b"checkpoint")
    (unrelated / "keep.txt").write_text("keep")
    training = {"completed": True, "parameter_mode": "lora"}
    result.mkdir()
    (result / "035_training_complete.json").write_text(json.dumps(training))
    _committed_endpoint(result, training)

    # closeout writes its record before it deletes anything
    record = safe_close_checkpoints(str(result), str(output))
    closeout = result / "035_checkpoint_closeout.json"
    assert json.loads(closeout.read_text()) == record
    assert record["removed"] == [{"path": "checkpoint-500", "bytes": 10}]
    assert not checkpoint.exists()
    # sibling directories that arent checkpoints survive the closeout
    assert (unrelated / "keep.txt").read_text() == "keep"
    assert safe_close_checkpoints(str(result), str(output)) == record


def test_observe_only_a2_writes_the_verdict_without_stopping(monkeypatch, tmp_path):
    from obs import make_grpo_sentinel

    monkeypatch.setitem(sys.modules, "transformers", _module(
        "transformers", TrainerCallback=object))
    cfg = RunConfig()
    cfg.experiment = "035"
    cfg.exploratory = True
    cfg.condition = Condition.MATCHED_FULL_GRPO
    cfg.train.grpo_parameter_mode = "full"
    cfg.train.grpo_precision_mode = "fp32_master_bf16"
    cfg.train.a2_action = "observe"
    callback = make_grpo_sentinel(cfg, out_dir=str(tmp_path))
    control = SimpleNamespace(should_training_stop=False)
    for step in range(1, 151):
        state = SimpleNamespace(global_step=step)
        callback.on_log(
            None, state, control,
            logs={"rewards/accuracy_reward/mean": 0.5, "kl": 0.001})
    verdict = json.loads((tmp_path / "a2_verdict.json").read_text())
    assert verdict["would_fire"] is True
    assert control.should_training_stop is False


def test_035_notebook_keeps_the_allocator_retry_record():
    with open("notebooks/035_colab_matched_grpo.ipynb") as f:
        notebook = json.load(f)
    code = "\n".join("".join(cell.get("source", []))
                     for cell in notebook["cells"] if cell["cell_type"] == "code")
    assert "ACTION = 'science_seed0'" in code
    assert "SCIENCE_START_RUNG = 'A'" in code
    assert "commit_execution_override(ROOT, selection, SCIENCE_START_RUNG)" in code
    assert code.count("prepare_allocator_retry(ROOT, selection, override)") == 2
    assert "PYTORCH_CUDA_ALLOC_CONF'] = ALLOCATOR_CONFIG" in code
    assert code.index("PYTORCH_CUDA_ALLOC_CONF'] = ALLOCATOR_CONFIG") < code.index(
        "import shutil, subprocess, torch")
    assert "allow_clean_retry=False" in code
    assert "scripts/s35_worker.py" in code
    assert "'train_steps': status.get('completed_updates', 0)" in code
    assert "code_root = repo_root / '001_Code'" in code
    assert "if not (code_root / 'scripts' / 'run.py').is_file():" in code
    assert "os.chdir(code_root)" in code
    assert "total_gib >= 79" in code
    for cell in notebook["cells"]:
        if cell["cell_type"] == "code":
            ast.parse("".join(cell.get("source", [])))


def test_035_checkpointing_pilot_is_isolated():
    with open("notebooks/035_1_colab_checkpointing_pilot.ipynb") as f:
        notebook = json.load(f)
    code = "\n".join("".join(cell.get("source", []))
                     for cell in notebook["cells"] if cell["cell_type"] == "code")
    assert "PILOT_RUNG = 'A_GC'" in code
    assert "--task', 'pilot'" in code
    assert "cu_before =" in code and "cu_after = None" in code
    assert "unit_rate =" in code
    assert "run_eval" not in code and "--task', 'eval'" not in code
    assert "commit_scale_selection" not in code
    assert "shutil.rmtree(local_out)" in code
    assert "report.get('execution') == PILOT_RUNGS_035[PILOT_RUNG]" in code
    assert "report.get('allocator_config') == ALLOCATOR_CONFIG" in code
    for cell in notebook["cells"]:
        if cell["cell_type"] == "code":
            ast.parse("".join(cell.get("source", [])))


def test_035_agc_science_runner_records_the_seed0_full_attempt():
    with open("notebooks/035_2_colab_agc_science.ipynb") as f:
        notebook = json.load(f)
    code = "\n".join("".join(cell.get("source", []))
                     for cell in notebook["cells"] if cell["cell_type"] == "code")
    assert "prepare_agc_science(ROOT, selection, override, retry)" in code
    assert "run_parity_pair()" in code
    assert "run_training()" in code
    assert "commit_execution_lock(ROOT, 'B3', 'A_GC', training)" in code
    assert "config_path('B3', 0, 'full')" in code
    assert "authorised rung:" in code
    assert "active_training_workers(cfg_path)" in code
    assert "previous.get('cuda_oom') is True" in code
    assert "fill cu_before and unit_rate in the units cell" in code
    assert "allow_clean_retry" not in code
    assert "branch='start'" not in code
    assert "cu_before =" in code and "cu_after = None" in code
    assert "unit_rate =" in code
    for cell in notebook["cells"]:
        assert re.fullmatch(r"[0-9a-f]{8}", cell["id"])
        if cell["cell_type"] == "code":
            ast.parse("".join(cell.get("source", [])))


def test_035_agc_lora_control_runner_records_the_completed_control():
    with open("notebooks/035_3_colab_agc_lora_control.ipynb") as f:
        notebook = json.load(f)
    code = "\n".join("".join(cell.get("source", []))
                     for cell in notebook["cells"] if cell["cell_type"] == "code")
    assert "prepare_agc_lora_control(ROOT)" in code
    assert "config_path('B3', 0, 'lora')" in code
    assert "branch='lora'" in code
    assert "verify_start_probe()" in code
    assert "verify_pair_parity()" in code
    assert "run_training()" in code and "run_endpoint(training)" in code
    assert "commit_execution_lock" not in code
    assert "branch='full'" not in code
    assert "allow_clean_retry" not in code
    assert "cu_before =" in code
    assert re.search(r"cu_after = (None|[\d.]+)", code)
    assert "unit_rate =" in code
    assert "035_3_colab_agc_lora_control" in code
    assert "closeout_path = os.path.join(result_dir, '035_checkpoint_closeout.json')" in code
    assert "checkpoint_path = os.path.join(output_dir, name)" in code
    assert "os.replace(part_path, closeout_path)" in code
    for cell in notebook["cells"]:
        assert re.fullmatch(r"[0-9a-f]{8}", cell["id"])
        if cell["cell_type"] == "code":
            ast.parse("".join(cell.get("source", [])))


def test_035b_pilot_is_zero_contact_and_b15_only():
    with open("notebooks/035_5_colab_b15_pilot.ipynb") as f:
        notebook = json.load(f)
    code = "\n".join("".join(cell.get("source", []))
                     for cell in notebook["cells"] if cell["cell_type"] == "code")
    assert "NOTEBOOK_NAME = '035_5_colab_b15_pilot'" in code
    assert "config_path('B15', 0, 'full')" in code
    assert "prepare_b15_scale_check(ROOT, START_PATH)" in code
    assert "PILOT_RUNG = 'A_GC'" in code
    assert "--task', 'pilot'" in code
    assert "pilot_passes(report)" in code
    assert "shutil.rmtree(local_out)" in code
    assert "run_eval" not in code and "--task', 'eval'" not in code
    assert re.search(r"cu_before = (None|[\d.]+)", code)
    assert re.search(r"cu_after = (None|[\d.]+)", code)
    assert re.search(r"unit_rate = (None|[\d.]+)", code)
    for item in notebook["cells"]:
        assert re.fullmatch(r"[0-9a-f]{8}", item["id"])
        if item["cell_type"] == "code":
            ast.parse("".join(item.get("source", [])))


def test_035b_runner_settles_both_training_outcomes_before_evaluation():
    with open("notebooks/035_6_colab_b15_scale_check.ipynb") as f:
        notebook = json.load(f)
    code = "\n".join("".join(cell.get("source", []))
                     for cell in notebook["cells"] if cell["cell_type"] == "code")
    assert "NOTEBOOK_NAME = '035_6_colab_b15_scale_check'" in code
    assert "prepare_b15_scale_check(ROOT, START_PATH)" in code
    assert "run_pair_parity()" in code
    assert code.index("full_outcome = run_arm('full')") < code.index(
        "lora_outcome = run_arm('lora')")
    assert code.index("settled = commit_b15_pair_settled(ROOT)") < code.index(
        "start_probe = verify_start_probe(completed_modes)")
    assert "status.get('error_message') == AGC_FULL_A1_ERROR" in code
    assert "active_training_workers(cfg_path)" in code
    assert "commit_b15_drive_resume(ROOT)" in code
    assert "report_to = ['wandb'] if resume else ['tensorboard', 'wandb']" in code
    assert "keep_status=resume" in code
    assert "package='035b_scale_check'" in code
    assert "os.environ['WANDB_PROJECT'] = full_cfg.log.wandb_project" in code
    assert "commit_execution_lock" not in code
    assert re.search(r"cu_before = (None|[\d.]+)", code)
    assert re.search(r"cu_after = (None|[\d.]+)", code)
    assert re.search(r"unit_rate = (None|[\d.]+)", code)
    for item in notebook["cells"]:
        assert re.fullmatch(r"[0-9a-f]{8}", item["id"])
        if item["cell_type"] == "code":
            ast.parse("".join(item.get("source", [])))
