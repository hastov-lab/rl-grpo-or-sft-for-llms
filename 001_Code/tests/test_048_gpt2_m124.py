import copy
import json
import os
from dataclasses import asdict

import pytest
import yaml

from checkpoint import write_artifact_type
from config import load_config
from scripts.s12_eval import IntegrityError, sha256
from scripts.s42_worker import ENDPOINT_FILES, commit_endpoint, commit_full_artifact
from scripts.s48_gpt2_lora_scales import (CONFIGS_048, CONFIGS_048_BY_SCALE,
                                          SEEDS_048, arm_tag,
                                          arm_state,
                                          base_eval_state,
                                          check_card_vram,
                                          check_registered_config,
                                          commit_sft_training, config_path,
                                          training_record,
                                          verify_base_reuse,
                                          verify_technical_records,
                                          verify_training_base)


def make_full_model(path, scale="M124"):
    os.makedirs(path, exist_ok=True)
    for name in ("config.json", "tokenizer.json", "tokenizer_config.json",
                 "model.safetensors"):
        with open(os.path.join(path, name), "w") as handle:
            handle.write(name)
    with open(os.path.join(path, "model_base.json"), "w") as handle:
        json.dump({"kind": "hf_id", "value": "gpt2"}, handle)
    write_artifact_type(path, "full", family="scratch_gpt", scale=scale)
    return path


def make_adapter(path, scale="M124"):
    os.makedirs(path, exist_ok=True)
    for name in ("adapter_model.safetensors", "adapter_config.json",
                 "config_hash.txt"):
        with open(os.path.join(path, name), "w") as handle:
            handle.write(name)
    with open(os.path.join(path, "adapter_base.json"), "w") as handle:
        json.dump({"kind": "local", "value": "base"}, handle)
    write_artifact_type(path, "adapter", family="scratch_gpt", scale=scale)
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


def test_048_configs_fix_the_m124_three_seed_roster():
    # remove the seed before comparing the fixed scientific settings
    rows = []
    for seed, path in zip(SEEDS_048, CONFIGS_048):
        cfg = check_registered_config(load_config(path))
        assert cfg.train.seed == seed
        assert cfg.scale.name == "M124"
        assert cfg.train.sft_parameter_mode == "lora"
        assert cfg.train.grpo_parameter_mode == "lora"
        assert cfg.exec.gradient_checkpointing is False
        row = asdict(cfg)
        row["train"].pop("seed")
        rows.append(row)
    assert rows[0] == rows[1] == rows[2]


def test_048_configs_add_the_m355_three_seed_roster():
    rows = []
    for seed, path in zip(SEEDS_048, CONFIGS_048_BY_SCALE["M355"]):
        cfg = check_registered_config(load_config(path))
        assert cfg.train.seed == seed
        assert cfg.scale.name == "M355"
        assert cfg.train.sft_parameter_mode == "lora"
        assert cfg.train.grpo_parameter_mode == "lora"
        assert cfg.exec.gradient_checkpointing is False
        row = asdict(cfg)
        row["train"].pop("seed")
        rows.append(row)
    assert rows[0] == rows[1] == rows[2]


def test_048_configs_add_the_m774_three_seed_roster():
    rows = []
    for seed, path in zip(SEEDS_048, CONFIGS_048_BY_SCALE["M774"]):
        cfg = check_registered_config(load_config(path))
        assert cfg.train.seed == seed
        assert cfg.scale.name == "M774"
        assert cfg.train.sft_parameter_mode == "lora"
        assert cfg.train.grpo_parameter_mode == "lora"
        assert cfg.exec.gradient_checkpointing is False
        row = asdict(cfg)
        row["train"].pop("seed")
        rows.append(row)
    assert rows[0] == rows[1] == rows[2]


def test_048_configs_reuse_the_046_scientific_values():
    for seed, path in zip(SEEDS_048, CONFIGS_048):
        current = asdict(load_config(path))
        source = asdict(load_config(f"configs/046_colab_gpt2_b15_seed{seed}.yaml"))
        assert current.pop("experiment") == "048"
        assert source.pop("experiment") == "046"
        assert current.pop("scale").name == "M124"
        assert source.pop("scale").name == "GPT2_B15"
        assert current == source

    for seed, path in zip(SEEDS_048, CONFIGS_048_BY_SCALE["M355"]):
        current = asdict(load_config(path))
        source = asdict(load_config(f"configs/048_colab_gpt2_m124_seed{seed}.yaml"))
        assert current.pop("scale").name == "M355"
        assert source.pop("scale").name == "M124"
        assert current == source

    for seed, path in zip(SEEDS_048, CONFIGS_048_BY_SCALE["M774"]):
        current = asdict(load_config(path))
        source = asdict(load_config(f"configs/048_colab_gpt2_m355_seed{seed}.yaml"))
        assert current.pop("scale").name == "M774"
        assert source.pop("scale").name == "M355"
        assert current == source


def test_048_yaml_files_differ_only_by_seed():
    rows = []
    for seed, path in zip(SEEDS_048, CONFIGS_048):
        with open(path) as handle:
            row = yaml.safe_load(handle)
        assert row["train"].pop("seed") == seed
        rows.append(row)
    assert rows[0] == rows[1] == rows[2]


def test_048_observe_allowance_stays_narrow(tmp_path):
    cfg = load_config(config_path(0))
    changed = copy.deepcopy(cfg)
    changed.experiment = "049"
    with pytest.raises(IntegrityError, match="registered package"):
        check_registered_config(changed)

    text = open(config_path(0)).read().replace(
        'experiment: "048"', 'experiment: "049"')
    path = tmp_path / "bad.yaml"
    path.write_text(text)
    with pytest.raises(ValueError, match="observe mode"):
        load_config(path)


def test_048_paths_and_card_are_fixed():
    assert config_path(2) == "configs/048_colab_gpt2_m124_seed2.yaml"
    assert arm_tag(1) == "048_COLAB_FULL_GPT2_M124_sft_grpo_seed1"
    assert check_card_vram(39.5) == "A100_40GB"
    with pytest.raises(IntegrityError, match="40GB"):
        check_card_vram(79.3)
    with pytest.raises(ValueError):
        arm_tag(3)
    assert config_path("M355", 2) == "configs/048_colab_gpt2_m355_seed2.yaml"
    assert arm_tag("M355", 1) == "048_COLAB_FULL_GPT2_M355_sft_grpo_seed1"
    assert check_card_vram(39.5, "M355") == "A100_40GB"
    assert config_path("M774", 2) == "configs/048_colab_gpt2_m774_seed2.yaml"
    assert arm_tag("M774", 1) == "048_COLAB_FULL_GPT2_M774_sft_grpo_seed1"
    assert check_card_vram(79.3, "M774") == "A100_80GB"
    with pytest.raises(IntegrityError, match="80GB"):
        check_card_vram(39.5, "M774")
    with pytest.raises(IntegrityError, match="40GB"):
        check_card_vram(79.3, "M355")


def test_048_reuses_only_the_committed_042_base(tmp_path):
    # the old base endpoint must verify before it enters the new package
    source = make_full_model(str(tmp_path / "source"))
    artifact = str(tmp_path / "base")
    result = str(tmp_path / "result")
    record_path = str(tmp_path / "base_manifest.json")
    record = commit_full_artifact(
        source, artifact, record_path, "pretrained_base",
        extra={"experiment": "042"})
    scratch = make_endpoint(str(tmp_path / "scratch"), "base")
    commit_endpoint(result, scratch, "base", record, package="042")
    audit_path = tmp_path / "token_audit.json"
    audit_path.write_text(json.dumps({
        "completed": True, "experiment": "042", "completion_cap": 512,
    }))
    assert base_eval_state(result, record) == "committed"
    assert verify_base_reuse(record_path, str(audit_path), result) == record

    changed = json.loads(audit_path.read_text())
    changed["completion_cap"] = 511
    audit_path.write_text(json.dumps(changed))
    with pytest.raises(IntegrityError, match="cannot reuse"):
        verify_base_reuse(record_path, str(audit_path), result)


def test_048_m355_base_uses_its_own_committed_endpoint(tmp_path):
    source = make_full_model(str(tmp_path / "source"), scale="M355")
    artifact = str(tmp_path / "base")
    result = str(tmp_path / "result")
    record_path = str(tmp_path / "base_manifest.json")
    record = commit_full_artifact(
        source, artifact, record_path, "immutable_base",
        extra={"experiment": "048"})
    scratch = make_endpoint(str(tmp_path / "scratch"), "base")
    commit_endpoint(result, scratch, "base", record, package="048")
    audit_path = tmp_path / "token_audit.json"
    audit_path.write_text(json.dumps({
        "completed": True, "experiment": "048", "completion_cap": 512,
    }))
    assert base_eval_state(result, record, "M355") == "committed"
    assert verify_base_reuse(
        record_path, str(audit_path), result, "M355") == record


def test_048_training_base_matches_the_live_scale(tmp_path):
    # the worker must not train a medium adapter over a base from another scale
    source = make_full_model(str(tmp_path / "source"), scale="M355")
    artifact = str(tmp_path / "base")
    record_path = str(tmp_path / "base_manifest.json")
    record = commit_full_artifact(
        source, artifact, record_path, "immutable_base",
        extra={"experiment": "048", "source": "gpt2-medium"})
    cfg = load_config(config_path("M355", 0))
    assert verify_training_base(cfg, record_path) == record

    changed = json.loads((tmp_path / "base" / "artifact_type.json").read_text())
    changed["scale"] = "M124"
    (tmp_path / "base" / "artifact_type.json").write_text(json.dumps(changed))
    with pytest.raises(IntegrityError):
        verify_training_base(cfg, record_path)


def test_048_m774_training_base_matches_the_large_checkpoint(tmp_path):
    source = make_full_model(str(tmp_path / "source"), scale="M774")
    artifact = str(tmp_path / "base")
    record_path = str(tmp_path / "base_manifest.json")
    record = commit_full_artifact(
        source, artifact, record_path, "immutable_base",
        extra={"experiment": "048", "source": "gpt2-large"})
    cfg = load_config(config_path("M774", 0))
    assert verify_training_base(cfg, record_path) == record


def test_048_m355_conversion_uses_the_registered_checkpoint(tmp_path, monkeypatch):
    # keep conversion on the pinned medium checkpoint and its recorded model scale
    import checkpoint
    import scripts.s42_worker as records
    import train.model_utils as model_utils
    from scripts.s46_worker import _convert

    monkeypatch.syspath_prepend(os.path.abspath("../002_GPT2_From_Scratch"))
    cfg = load_config(config_path("M355", 0))
    captured = {}

    class Saved:
        def save_pretrained(self, path):
            captured.setdefault("saved", []).append(path)

    monkeypatch.setattr(
        model_utils, "load_scratch_model_and_tokenizer",
        lambda cfg_arg, source, dtype: (Saved(), Saved()))
    monkeypatch.setattr(model_utils, "write_model_base", lambda *args: None)
    monkeypatch.setattr(checkpoint, "write_artifact_type", lambda *args, **kwargs:
                        captured.update({"artifact_scale": kwargs["scale"]}))

    def commit(scratch, out, record, role, extra):
        captured.update({"role": role, **extra})
        return captured

    monkeypatch.setattr(records, "commit_full_artifact", commit)
    result = _convert(
        cfg, str(tmp_path / "scratch"), str(tmp_path / "out"),
        str(tmp_path / "record.json"), package="048")
    assert result["artifact_scale"] == "M355"
    assert result["experiment"] == "048"
    assert result["source"] == "gpt2-medium"
    assert result["source_revision"] == (
        "6dcaa7a952f72f9298047fd5137cd6e4f05f41da")


def test_048_m355_technical_records_are_bound_to_the_new_base(tmp_path, monkeypatch):
    # bind parity and token audit records to the same immutable base
    monkeypatch.syspath_prepend(os.path.abspath("../002_GPT2_From_Scratch"))
    source = make_full_model(str(tmp_path / "source"), scale="M355")
    artifact = str(tmp_path / "base")
    base_path = str(tmp_path / "base_manifest.json")
    commit_full_artifact(
        source, artifact, base_path, "immutable_base",
        extra={"experiment": "048"})
    source_audit = tmp_path / "source_audit.json"
    source_audit.write_text("source")
    parity = tmp_path / "parity.json"
    parity.write_text(json.dumps({
        "completed": True, "experiment": "048",
        "source": "openai-community/gpt2-medium",
        "source_revision": "6dcaa7a952f72f9298047fd5137cd6e4f05f41da",
        "base_record_sha256": sha256(base_path),
        "max_abs_unpadded_logit_error": 0.0,
        "max_abs_padded_logit_error": 0.0,
        "abs_loss_error": 0.0,
        "max_abs_cached_logit_error": 0.0,
        "reference_cache_error": 0.0,
        "scratch_cache_error": 0.0,
        "reload_logit_error": 0.0,
        "max_new_tokens": 32,
        "generation_tokens_equal": True,
        "generation_diagnostic": {"first_divergence": None},
        "tied_parameter_and_storage": True,
    }))
    audit = tmp_path / "audit.json"
    audit.write_text(json.dumps({
        "completed": True, "experiment": "048",
        "base_record_sha256": sha256(base_path),
        "completion_cap": 512,
        "source_audit": str(source_audit),
        "source_audit_sha256": sha256(source_audit),
    }))
    records = verify_technical_records(
        load_config(config_path("M355", 0)), base_path, str(parity), str(audit))
    assert records["base"]["artifact"] == artifact


def test_048_sft_record_binds_the_adapter_and_042_base(tmp_path):
    cfg = load_config(config_path(0))
    result = tmp_path / "result"
    result.mkdir()
    adapter = make_adapter(str(tmp_path / "adapter"))
    history = [{"step": step, "loss": 1.0 / step} for step in range(1, 501)]
    (result / "sft_trainlog.json").write_text(json.dumps(history))
    base_path = tmp_path / "base.json"
    base_path.write_text(json.dumps({"artifact_sha256": {"model": "abc"}}))
    wandb_path = tmp_path / "wandb.json"
    wandb_path.write_text(json.dumps({"completed": True}))
    wandb = [{"path": str(wandb_path), "sha256": sha256(wandb_path)}]

    record = commit_sft_training(
        cfg, str(result), adapter, str(base_path), wandb)
    assert record["experiment"] == "048"
    assert record["base_record_sha256"] == sha256(base_path)
    assert training_record(str(result), "sft") == record


def test_048_worker_reuses_the_046_merge_and_evaluation_paths():
    # merge and evaluation stay on the functions that completed experiment 046
    import scripts.s46_worker as source
    import scripts.s48_worker as current

    assert current._merge is source._merge
    assert current._run_eval is source._run_eval


def test_048_archives_the_seed2_zero_gradient_attempt_before_retry(tmp_path):
    # only the failed GRPO branch moves; the SFT record remains the retry start
    root = tmp_path
    tag = arm_tag("M355", 2)
    result = root / "results" / tag
    checkpoint = root / "checkpoints" / tag
    result.mkdir(parents=True)
    grpo = checkpoint / "grpo"
    (grpo / "checkpoint-500").mkdir(parents=True)
    (grpo / "checkpoint-500" / "trainer_state.json").write_text("old checkpoint")
    wandb_dir = result / "048_grpo_wandb_runs"
    wandb_dir.mkdir()
    (wandb_dir / "oldrun.json").write_text("old run")
    (result / "048_attempts.jsonl").write_text("failed attempt\n")

    cfg = load_config(config_path("M355", 2))
    adapter = make_adapter(str(checkpoint / "sft"), scale="M355")
    history = [{"step": step, "loss": 1.0 / step} for step in range(1, 501)]
    (result / "sft_trainlog.json").write_text(json.dumps(history))
    base_path = result / "base.json"
    base_path.write_text(json.dumps({"artifact_sha256": {"model": "abc"}}))
    wandb_path = result / "sft_wandb.json"
    wandb_path.write_text(json.dumps({"completed": True}))
    commit_sft_training(cfg, str(result), adapter, str(base_path), [{
        "path": str(wandb_path), "sha256": sha256(wandb_path)}])
    status = {
        "completed": False,
        "error_type": "IntegrityError",
        "error_message": "matched GRPO first-step gradients are invalid",
        "cuda_oom": False,
        "start_step": 0,
        "end_step": 500,
        "completed_updates": 500,
    }
    (result / "grpo_worker_status.json").write_text(json.dumps(status))

    assert arm_state(str(result)) == {"state": "absent"}
    archive = root / "failed_attempts" / "048" / "M355_seed2_initial_zero_gradient"
    assert (archive / "archive_manifest.json").exists()
    assert (archive / "checkpoints" / "grpo" / "checkpoint-500"
            / "trainer_state.json").read_text() == "old checkpoint"
    assert (archive / "results" / "grpo_worker_status.json").exists()
    assert not (result / "grpo_worker_status.json").exists()
    assert not grpo.exists()
    assert (result / "048_sft_training.json").exists()
    assert (result / "048_attempts.jsonl").read_text() == "failed attempt\n"
    assert arm_state(str(result)) == {"state": "absent"}

    (archive / "results" / "grpo_worker_status.json").write_text("changed")
    with pytest.raises(IntegrityError, match="archive changed"):
        arm_state(str(result))
