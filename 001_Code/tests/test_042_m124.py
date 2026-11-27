import copy
import json
import os
from dataclasses import asdict
from types import SimpleNamespace

import pytest
import yaml

from checkpoint import config_hash, write_artifact_type
from config import load_config
from scripts.s12_eval import IntegrityError, sha256
from scripts.s42_worker import (A1_ERROR, CONFIGS_042, ENDPOINT_FILES,
                                _completion_length_gate, _starvation_summary,
                                _verify_grpo_runtime, a1_state, arm_state,
                                build_length_audit,
                                commit_a1_conclusion, commit_arm_complete,
                                commit_endpoint, commit_full_artifact,
                                config_path, endpoint_state, pilot_passes,
                                verify_artifact_record)


def make_full_model(path, salt="model"):
    # build the smallest typed full artifact accepted by the shared loader
    os.makedirs(path, exist_ok=True)
    for name in ("config.json", "tokenizer.json", "tokenizer_config.json",
                 "model.safetensors"):
        with open(os.path.join(path, name), "w") as handle:
            handle.write(f"{salt}-{name}")
    with open(os.path.join(path, "model_base.json"), "w") as handle:
        json.dump({"kind": "hf_id", "value": "gpt2"}, handle)
    write_artifact_type(path, "full", family="scratch_gpt", scale="M124")
    return path


def make_endpoint(path, branch):
    # mirror the aligned generative and probe rows written by the real evaluator
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


def test_042_configs_are_the_fixed_three_seed_roster():
    # pin the planned training settings before any seed produces a result
    rows = []
    for seed, path in enumerate(CONFIGS_042):
        cfg = load_config(path)
        assert cfg.experiment == "042"
        assert cfg.scale.name == "M124"
        assert cfg.train.seed == seed
        assert cfg.train.sft_parameter_mode == "full"
        assert cfg.train.grpo_parameter_mode == "full"
        assert cfg.train.sft_precision_mode == "fp32_master_bf16"
        assert cfg.train.grpo_precision_mode == "fp32_master_bf16"
        assert cfg.train.a2_action == "observe"
        assert cfg.train.max_steps == 500
        assert cfg.grpo.max_completion_length == cfg.eval.max_new_tokens == 512
        assert cfg.exec.allow_oom_fallback is False
        row = asdict(cfg)
        row["train"].pop("seed")
        rows.append(row)
    assert rows[0] == rows[1] == rows[2]


def test_042_yaml_files_differ_only_by_seed():
    rows = []
    for seed, path in enumerate(CONFIGS_042):
        with open(path) as handle:
            row = yaml.safe_load(handle)
        assert row["train"].pop("seed") == seed
        rows.append(row)
    assert rows[0] == rows[1] == rows[2]


def test_existing_token_audit_uses_the_registered_config_cap(tmp_path):
    cfg = load_config(config_path(0))
    cfg.grpo.max_completion_length = 384
    path = tmp_path / "token_audit.json"
    path.write_text(json.dumps({"completed": True, "completion_cap": 384}))
    assert build_length_audit(cfg, str(path))["completion_cap"] == 384

    cfg.grpo.max_completion_length = 512
    with pytest.raises(IntegrityError, match="token audit changed"):
        build_length_audit(cfg, str(path))


def test_scratch_names_reach_every_precision_family():
    from scripts.s35_matched_grpo import module_family

    names = {
        "tok_emb.weight": "embeddings",
        "pos_emb.weight": "embeddings",
        "trf_blocks.0.att.W_query.weight": "attention",
        "trf_blocks.0.ff.layers.0.weight": "mlp",
        "trf_blocks.0.norm1.scale": "normalization",
        "out_head.weight": "output_head",
    }
    assert {name: module_family(name) for name in names} == names


def test_grpo_runtime_checks_policy_and_reference_dropout():
    torch = pytest.importorskip("torch")

    policy = torch.nn.Sequential(torch.nn.Linear(2, 2), torch.nn.Dropout(0.0))
    reference = torch.nn.Sequential(torch.nn.Linear(2, 2), torch.nn.Dropout(0.0))
    for parameter in reference.parameters():
        parameter.requires_grad_(False)
    trainer = SimpleNamespace(model=policy, ref_model=reference)
    _verify_grpo_runtime(trainer)

    reference[1].p = 0.1
    with pytest.raises(IntegrityError, match="reference dropout"):
        _verify_grpo_runtime(trainer)
    with pytest.raises(IntegrityError, match="no frozen reference"):
        _verify_grpo_runtime(SimpleNamespace(model=policy, ref_model=None))


def test_completion_length_gate_fails_on_the_first_training_log():
    seen = []
    gate = _completion_length_gate(seen)
    state = SimpleNamespace(global_step=1)
    control = SimpleNamespace()
    with pytest.raises(IntegrityError, match="missing completion length"):
        gate.on_log(
            None, state, control,
            logs={"rewards/accuracy_reward/mean": 0.0})

    gate = _completion_length_gate(seen)
    gate.on_log(
        None, state, control,
        logs={"rewards/accuracy_reward/mean": 0.0,
              "completions/max_length": 12.0})
    assert seen == [12.0]


def test_full_artifact_commit_is_write_once_and_hash_bound(tmp_path):
    source = make_full_model(str(tmp_path / "source"))
    destination = str(tmp_path / "destination")
    record_path = str(tmp_path / "base_manifest.json")
    first = commit_full_artifact(
        source, destination, record_path, "immutable_base", {"source": "gpt2"})
    second = commit_full_artifact(
        source, destination, record_path, "immutable_base", {"source": "gpt2"})
    assert first == second == verify_artifact_record(record_path)

    with open(os.path.join(destination, "model.safetensors"), "a") as handle:
        handle.write("changed")
    with pytest.raises(IntegrityError, match="hashes changed"):
        verify_artifact_record(record_path)


def test_endpoint_commit_is_manifest_last_and_tamper_stops(tmp_path):
    result_dir = str(tmp_path / "results")
    scratch = make_endpoint(str(tmp_path / "scratch"), "sft")
    binding = {"completed": True, "artifact_sha256": {"model": "abc"}}
    assert endpoint_state(result_dir, "sft", binding) == "absent"
    manifest = commit_endpoint(result_dir, scratch, "sft", binding)
    assert endpoint_state(result_dir, "sft", binding) == "committed"
    assert manifest["binding"] == binding

    with open(os.path.join(result_dir, "sft_eval.json"), "a") as handle:
        handle.write("changed")
    assert endpoint_state(result_dir, "sft", binding) == "tampered"
    with pytest.raises(IntegrityError, match="endpoint changed"):
        commit_endpoint(result_dir, scratch, "sft", binding)


def test_endpoint_rejects_metrics_that_disagree_with_predictions(tmp_path):
    scratch = make_endpoint(str(tmp_path / "scratch"), "grpo")
    with open(os.path.join(scratch, "grpo_eval.json"), "w") as handle:
        json.dump({"n_items": 1884, "per_tier_acc": {"easy": 0.5}}, handle)
    with pytest.raises(IntegrityError, match="differs from predictions"):
        commit_endpoint(str(tmp_path / "results"), scratch, "grpo", {"x": 1})


def test_a1_conclusion_accepts_only_the_exact_sentinel(tmp_path):
    # bind the stopped arm to its SFT start, online run and exact A1 status
    result_dir = str(tmp_path)
    sft_path = tmp_path / "042_sft_training.json"
    sft_path.write_text(json.dumps({"completed": True, "stage": "sft"}))
    status_path = tmp_path / "grpo_worker_status.json"
    status = {
        "completed": False,
        "seed": 0,
        "cuda_oom": False,
        "error_type": "RuntimeError",
        "error_message": A1_ERROR,
    }
    status_path.write_text(json.dumps(status))
    wandb_dir = tmp_path / "042_wandb_runs"
    wandb_dir.mkdir()
    cfg = load_config(config_path(0))
    run_id = "run042"
    (wandb_dir / f"{run_id}.json").write_text(json.dumps({
        "completed": True,
        "experiment": "042",
        "seed": 0,
        "project": cfg.log.wandb_project,
        "run_id": run_id,
        "run_name": f"{tmp_path.name}_grpo",
        "entity": None,
        "config_hash": config_hash(cfg),
        "first_recorded_step": 1,
    }))
    assert a1_state(result_dir)[0] == "pending"
    record = commit_a1_conclusion(result_dir)
    assert record["outcome"] == "a1"
    assert arm_state(result_dir)["state"] == "a1"

    os.remove(tmp_path / "042_a1_conclusion.json")
    status["error_message"] += "."
    status_path.write_text(json.dumps(status))
    assert a1_state(result_dir)[0] == "failed"
    with pytest.raises(IntegrityError, match="no exact A1"):
        commit_a1_conclusion(result_dir)


def test_arm_completion_rechecks_both_endpoints(tmp_path):
    result_dir = str(tmp_path / "results")
    os.makedirs(result_dir)
    records = {}
    for branch in ("sft", "grpo"):
        record = {"completed": True, "stage": branch, "artifact_sha256": {"x": branch}}
        record_path = os.path.join(result_dir, f"042_{branch}_training.json")
        with open(record_path, "w") as handle:
            json.dump(record, handle)
        commit_endpoint(
            result_dir, make_endpoint(str(tmp_path / f"scratch_{branch}"), branch),
            branch, record)
        records[branch] = record
    complete = commit_arm_complete(result_dir)
    assert complete["completed"] is True
    assert arm_state(result_dir)["state"] == "committed"

    with open(os.path.join(result_dir, "grpo_eval.json"), "a") as handle:
        handle.write("changed")
    with pytest.raises(IntegrityError):
        arm_state(result_dir)


def test_starvation_summary_uses_the_registered_three_fields():
    rows = [{
        "rewards/accuracy_reward/mean": 0.01,
        "rewards/format_reward/mean": 0.0,
        "frac_reward_zero_std": 0.98,
    } for _ in range(500)]
    summary = _starvation_summary(rows)
    assert summary["severely_reward_starved"] is True
    assert summary["first_50"] == pytest.approx(summary["last_50"])
    assert summary["last_50"] == pytest.approx(summary["complete"])

    changed = copy.deepcopy(rows)
    for row in changed[:25]:
        row["rewards/accuracy_reward/mean"] = 1.0
    assert _starvation_summary(changed)["severely_reward_starved"] is False


def test_pilot_requires_memory_precision_reference_and_deleted_weights():
    # keep a fitting run from passing when its weights never moved
    changes = {
        name: {"changed_elements": 1}
        for name in ("embeddings", "attention", "mlp", "normalization")
    }
    report = {
        "completed_updates": 30,
        "peak_reserved_fraction": 0.8,
        "projected_500_updates_hours": 3.0,
        "checkpoint_bytes": 100,
        "precision_audit": {
            "gradient_dtypes": ["float32"],
            "optimizer_state_dtypes": ["float32"],
            "reference_gate": {"verified": True},
            "module_changes": changes,
        },
    }
    closeout = {"weights_deleted": True}
    assert pilot_passes(report, closeout)
    bad = copy.deepcopy(report)
    bad["precision_audit"]["module_changes"]["attention"]["changed_elements"] = 0
    assert not pilot_passes(bad, closeout)


def test_worker_keeps_the_pilot_out_of_the_evaluation_branch():
    source = open("scripts/s42_worker.py").read()
    pilot_block = source[source.index('elif args.task in ("pilot", "grpo")'):
                         source.index("else:\n            binding = read_json", source.index(
                             'elif args.task in ("pilot", "grpo")'))]
    assert "_run_eval" not in pilot_block
    assert "AutoModelForCausalLM" not in source
