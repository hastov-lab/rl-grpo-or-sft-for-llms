import json
import os
import shutil

import pytest
import yaml

from config import load_config
from enums import Condition
from scripts.s12_eval import ADAPTER_FILES, IntegrityError
from scripts.s28_compose import (ARCHIVED_MERGE_RECORD, BINDING_NAME,
                                 archived_merge_state, ensure_bound_merge,
                                 commit_training_complete, merged_hashes,
                                 stage_archived_merge, training_state,
                                 verify_archived_merge, verify_historical_merge)


def make_adapter(path):
    os.makedirs(path, exist_ok=True)
    for name in ADAPTER_FILES:
        with open(os.path.join(path, name), "w") as f:
            f.write("adapter-" + name)
    return str(path)


def make_merged(path):
    os.makedirs(path, exist_ok=True)
    for name in ("config.json", "tokenizer.json", "tokenizer_config.json",
                 "model.safetensors"):
        with open(os.path.join(path, name), "w") as f:
            f.write("merged-" + name)
    return str(path)


@pytest.mark.parametrize("seed", (0, 1, 2))
def test_030_config_differs_from_013_only_as_registered(seed):
    new = yaml.safe_load(open(f"configs/030_colab_b7_n4_seed{seed}.yaml"))
    old = yaml.safe_load(open(f"configs/013_colab_full_b7_seed{seed}.yaml"))

    assert new.pop("experiment") == "030" and old.pop("experiment") == "013"
    assert new.pop("exploratory") is True
    assert new.pop("condition") == "EXTERNAL_SFT_GRPO"
    assert old.pop("condition") == "SFT_GRPO"

    new_train, old_train = new.pop("train"), old.pop("train")
    assert new_train.pop("num_generations") == 4
    assert old_train.pop("num_generations") == 8
    assert new_train.pop("grpo_from_adapter") == \
        f"checkpoints/013_COLAB_FULL_B7_sft_grpo_seed{seed}/sft"
    assert "grpo_from_adapter" not in old_train
    assert new_train == old_train
    assert new == old


def test_030_configs_differ_only_by_seed_and_source():
    configs = [yaml.safe_load(open(f"configs/030_colab_b7_n4_seed{seed}.yaml"))
               for seed in (0, 1, 2)]
    for seed, cfg in enumerate(configs):
        assert cfg["train"].pop("seed") == seed
        assert cfg["train"].pop("grpo_from_adapter") == \
            f"checkpoints/013_COLAB_FULL_B7_sft_grpo_seed{seed}/sft"
    assert configs[0] == configs[1] == configs[2]


def test_030_loads_with_the_registered_batch_and_stages():
    from scripts.run import plan_stages

    cfg = load_config("configs/030_colab_b7_n4_seed0.yaml")
    assert cfg.exploratory is True
    assert cfg.condition == Condition.EXTERNAL_SFT_GRPO
    assert cfg.train.num_generations == 4
    per_device = cfg.exec.grpo_batch_size or cfg.train.num_generations
    assert per_device == 4 and per_device * cfg.exec.grpo_grad_accum == 4
    assert cfg.exec.gradient_checkpointing is False
    assert cfg.exec.allow_oom_fallback is False
    assert plan_stages(cfg.condition) == ["base_eval", "grpo", "grpo_eval"]


def test_archived_merge_commits_and_never_rebuilds(tmp_path):
    source = make_merged(tmp_path / "013" / "sft_merged")
    adapter = make_adapter(tmp_path / "013" / "sft")
    merged = str(tmp_path / "030" / "sft_merged")
    grpo = str(tmp_path / "030" / "grpo")

    record = stage_archived_merge(source, merged, adapter, grpo, log=lambda *a: None)
    assert record["completed"] is True
    assert archived_merge_state(source, merged, adapter) == "committed"
    assert os.path.exists(os.path.join(merged, ARCHIVED_MERGE_RECORD))
    assert os.path.exists(os.path.join(merged, BINDING_NAME))

    def must_not_rebuild():
        raise AssertionError("the archived base was rebuilt")

    assert ensure_bound_merge(adapter, merged, grpo, must_not_rebuild,
                              log=lambda *a: None) == merged


def test_interrupted_copy_repeats_without_rebuilding(tmp_path):
    source = make_merged(tmp_path / "013" / "sft_merged")
    adapter = make_adapter(tmp_path / "013" / "sft")
    merged = str(tmp_path / "030" / "sft_merged")
    grpo = str(tmp_path / "030" / "grpo")
    calls = []

    def interrupted_copy(src, dst):
        calls.append(src)
        if len(calls) == 2:
            raise OSError("runtime stopped during copy")
        return shutil.copyfile(src, dst)

    with pytest.raises(OSError):
        stage_archived_merge(source, merged, adapter, grpo,
                             copy_fn=interrupted_copy, log=lambda *a: None)
    assert not os.path.exists(os.path.join(merged, ARCHIVED_MERGE_RECORD))

    stage_archived_merge(source, merged, adapter, grpo, log=lambda *a: None)
    assert archived_merge_state(source, merged, adapter) == "committed"
    assert any(p.name.startswith("sft_merged_unbound_")
               for p in (tmp_path / "030").iterdir())


def test_incomplete_copy_beside_training_halts(tmp_path):
    source = make_merged(tmp_path / "013" / "sft_merged")
    adapter = make_adapter(tmp_path / "013" / "sft")
    merged = str(tmp_path / "030" / "sft_merged")
    grpo = str(tmp_path / "030" / "grpo")
    os.makedirs(merged)
    with open(os.path.join(merged, "config.json"), "w") as f:
        f.write("partial")
    os.makedirs(os.path.join(grpo, "checkpoint-50"))

    with pytest.raises(IntegrityError):
        stage_archived_merge(source, merged, adapter, grpo, log=lambda *a: None)


def test_committed_copy_halts_on_source_destination_or_binding_drift(tmp_path):
    source = make_merged(tmp_path / "013" / "sft_merged")
    adapter = make_adapter(tmp_path / "013" / "sft")
    merged = str(tmp_path / "030" / "sft_merged")
    grpo = str(tmp_path / "030" / "grpo")
    stage_archived_merge(source, merged, adapter, grpo, log=lambda *a: None)

    for path in (os.path.join(source, "tokenizer.json"),
                 os.path.join(merged, "tokenizer.json"),
                 os.path.join(merged, BINDING_NAME)):
        original = open(path).read()
        with open(path, "a") as f:
            f.write("drift")
        with pytest.raises(IntegrityError):
            verify_archived_merge(source, merged, adapter)
        with open(path, "w") as f:
            f.write(original)
    assert archived_merge_state(source, merged, adapter) == "committed"


def test_historical_merge_record_anchors_the_013_archive(tmp_path):
    source = make_merged(tmp_path / "013" / "sft_merged")
    source_run = "013_COLAB_FULL_B7_sft_grpo_seed0"
    manifest = tmp_path / "022_arm_manifest.json"
    json.dump({"source_run": source_run,
               "hashes": {f"sft_merged/{name}": value
                          for name, value in merged_hashes(source).items()}},
              open(manifest, "w"))

    record = verify_historical_merge(source, str(manifest), source_run)
    assert record["source_sha256"] == merged_hashes(source)

    with open(os.path.join(source, "model.safetensors"), "a") as f:
        f.write("drift")
    with pytest.raises(IntegrityError):
        verify_historical_merge(source, str(manifest), source_run)


def test_historical_merge_missing_anchor_halts(tmp_path):
    source = make_merged(tmp_path / "013" / "sft_merged")
    missing = tmp_path / "missing_arm_manifest.json"
    with pytest.raises(IntegrityError):
        verify_historical_merge(
            source, str(missing), "013_COLAB_FULL_B7_sft_grpo_seed1")


def test_seed12_historical_anchors_are_present_and_distinct():
    weight_hashes = []
    for seed in (1, 2):
        path = f"results/022_smooth_B7_seed{seed}_step1000/arm_manifest.json"
        record = json.load(open(path))
        assert record["source_run"] == f"013_COLAB_FULL_B7_sft_grpo_seed{seed}"
        merged = {name: value for name, value in record["hashes"].items()
                  if name.startswith("sft_merged/")}
        assert len(merged) == 6
        weight_hashes.append(merged["sft_merged/model.safetensors"])
    assert len(set(weight_hashes)) == 2


def make_completed_training(tmp_path):
    from checkpoint import config_hash

    cfg = load_config("configs/030_colab_b7_n4_seed0.yaml")
    cfg.output_root = str(tmp_path)
    rdir = tmp_path / "results" / "030"
    out_dir = tmp_path / "checkpoints" / "030" / "grpo"
    merged_dir = tmp_path / "checkpoints" / "030" / "sft_merged"
    rdir.mkdir(parents=True)
    make_adapter(out_dir)
    with open(out_dir / "config_hash.txt", "w") as f:
        f.write(config_hash(cfg))
    json.dump({"kind": "local",
               "value": os.path.relpath(merged_dir, cfg.output_root)},
              open(out_dir / "adapter_base.json", "w"))
    history = [{"step": step,
                "rewards/accuracy_reward/mean": 0.5,
                "learning_rate": cfg.train.grpo_lr
                * (cfg.train.max_steps - step + 1) / cfg.train.max_steps}
               for step in range(1, cfg.train.max_steps + 1)]
    json.dump(history, open(rdir / "grpo_trainlog.json", "w"))
    checkpoint = out_dir / f"checkpoint-{cfg.train.max_steps}"
    checkpoint.mkdir()
    json.dump({"global_step": cfg.train.max_steps},
              open(checkpoint / "trainer_state.json", "w"))
    return cfg, str(rdir), str(out_dir), str(merged_dir)


def test_training_completion_record_binds_steps_schedule_and_artifacts(tmp_path):
    cfg, rdir, out_dir, merged_dir = make_completed_training(tmp_path)
    record = commit_training_complete(rdir, out_dir, merged_dir, cfg)
    assert record["steps"] == 500
    assert training_state(rdir, out_dir, merged_dir, cfg) == "committed"

    with open(os.path.join(rdir, "grpo_trainlog.json"), "a") as f:
        f.write(" ")
    with pytest.raises(IntegrityError):
        training_state(rdir, out_dir, merged_dir, cfg)


@pytest.mark.parametrize("change", ("missing_step", "duplicate_step", "increasing_lr",
                                     "wrong_global_step", "wrong_base"))
def test_training_completion_rejects_incomplete_treatment(tmp_path, change):
    cfg, rdir, out_dir, merged_dir = make_completed_training(tmp_path)
    history_p = os.path.join(rdir, "grpo_trainlog.json")
    history = json.load(open(history_p))
    if change == "missing_step":
        history.pop()
    elif change == "duplicate_step":
        history.append(dict(history[-1]))
    elif change == "increasing_lr":
        history[200]["learning_rate"] = history[199]["learning_rate"] + 1e-6
    elif change == "wrong_global_step":
        state_p = os.path.join(out_dir, "checkpoint-500", "trainer_state.json")
        json.dump({"global_step": 499}, open(state_p, "w"))
    elif change == "wrong_base":
        json.dump({"kind": "local", "value": "checkpoints/other/sft_merged"},
                  open(os.path.join(out_dir, "adapter_base.json"), "w"))
    if change in ("missing_step", "duplicate_step", "increasing_lr"):
        json.dump(history, open(history_p, "w"))
    with pytest.raises(IntegrityError):
        commit_training_complete(rdir, out_dir, merged_dir, cfg)


def test_external_sft_summary_uses_the_gsm8k_parent_label(tmp_path):
    from scripts.run import build_summary

    cfg = load_config("configs/030_colab_b7_n4_seed0.yaml")
    cfg.output_root = str(tmp_path)
    tag = "013_COLAB_FULL_B7_sft_grpo_seed0"
    parent = tmp_path / "results" / tag
    parent.mkdir(parents=True)
    json.dump({"per_tier_acc": {"hard": 0.6}}, open(parent / "sft_eval.json", "w"))
    json.dump({"per_tier_acc": {"hard": 0.55}}, open(parent / "sft_logprob.json", "w"))
    json.dump({"config": {"train": {"sft_data": "gsm8k"}}},
              open(parent / "manifest.json", "w"))
    grpo = {"per_tier_acc": {"hard": 0.64}}
    probe = {"per_tier_acc": {"hard": 0.55}}

    summary = build_summary(cfg, None, None, grpo, None, None, probe)
    assert summary["reference"] == f"gsm8k_sft_parent:{tag}"
    assert summary["sft_acc"] == {"hard": 0.6}
    assert abs(summary["grpo_delta"]["hard"] - 0.04) < 1e-9


def test_030_notebook_carries_the_registered_guards():
    notebook = json.load(open("notebooks/030_colab_b7_n4_ablation.ipynb"))
    source = "\n".join("".join(cell.get("source", [])) for cell in notebook["cells"])

    assert "cfg.condition.name == 'EXTERNAL_SFT_GRPO'" in source
    assert "cfg.train.num_generations == 4" in source
    assert "stage_archived_merge(ARCHIVE_MERGED" in source
    assert source.index("verify_archived_merge(ARCHIVE_MERGED") < \
        source.index("scripts/grpo_worker.py")
    assert "results/022_smooth_B7_seed0_step1000/arm_manifest.json" in source
    assert "verify_historical_merge(" in source
    assert "commit_training_complete(RES_DIR, OUT_DIR, MERGED_DIR, cfg)" in source
    assert source.index("commit_training_complete(RES_DIR, OUT_DIR, MERGED_DIR, cfg)") < \
        source.index("ep_state = endpoint_state(RES_DIR, OUT_DIR)")
    assert "'training_complete': json.load(open(" in source
    assert "package='030_n4'" in source
    assert "gsm8k_sft_parent:{PARENT_TAG}" in source
    assert "--no-ledger" in source
    assert "os.environ['WANDB_PROJECT'] = 'inm363-grpo'" in source
    assert "allow_oom_fallback is False" in source
    assert "per_device x accum" not in source
    assert "paired_delta_ci" in source and "mcnemar_exact" in source
    assert "primary_endpoint': 'hard'" in source


def test_030_seed12_notebook_carries_the_registered_guards():
    # notebook ran on 12-13 aug, so unit fields now hold session values
    # and asserts for blank fields moved to 031 nb test
    notebook = json.load(open("notebooks/030_colab_b7_n4_seeds12.ipynb"))
    source = "\n".join("".join(cell.get("source", [])) for cell in notebook["cells"])
    assert "SEEDS_TO_RUN = (1, 2)" in source
    assert "configs/030_colab_b7_n4_seed1.yaml" in source
    assert "configs/030_colab_b7_n4_seed2.yaml" in source
    assert "results/022_smooth_B7_seed1_step1000/arm_manifest.json" in source
    assert "results/022_smooth_B7_seed2_step1000/arm_manifest.json" in source
    assert "f'030b_preflight_seed{seed}.json'" in source
    assert "030_preflight.json" not in source
    for name in ("sft_eval.json", "sft_preds.jsonl", "sft_logprob.json",
                 "sft_logprob_preds.jsonl", "grpo_eval.json", "grpo_preds.jsonl",
                 "grpo_logprob.json", "grpo_logprob_preds.jsonl",
                 "grpo_trainlog.json", "manifest.json"):
        assert name in source
    assert source.index("## 6. Preflight") < source.index("scripts/build_data.py")
    assert source.index("verify_historical_merge(\n        ARCHIVE_MERGED[seed]") < \
        source.index("record = stage_archived_merge(")
    assert source.index("record = stage_archived_merge(") < source.index("worker_cmd =")
    assert "if status_is_a2(status_now):" in source
    assert "else:\n                    raise RuntimeError(" in source
    assert "blocked_resume_reason" in source
    assert "commit_training_complete(res_dir, out_dir, merged_dir, cfg)" in source
    assert source.index("commit_training_complete(res_dir, out_dir, merged_dir, cfg)") < \
        source.index("ep_state = endpoint_state(res_dir, out_dir)")
    assert "package='030_n4'" in source
    assert "gsm8k_sft_parent:{parent_tag}" in source
    assert "os.environ['WANDB_PROJECT'] = 'inm363-grpo'" in source
    assert "allow_oom_fallback is False" in source
    assert "030_family_summary.json" in source
    assert "030_comparison_summary.json" not in source
    assert "holm([row['hard_raw_p'] for row in completed])" in source
    assert "formal_equivalence_possible': False" in source


