import json

import yaml

from config import load_config


def _raw(path):
    with open(path) as f:
        return yaml.safe_load(f)


def test_004_seed_configs_match_except_registered():
    # s4 rule: new seeds equal the seed-0 cell plus only the seed, the explicit
    # ladder switch, and seed 2's registered exploratory status
    base = _raw("configs/004_colab_full_b3_base_grpo_seed0.yaml")
    assert base["train"]["seed"] == 0
    assert "allow_oom_fallback" not in base.get("exec", {})
    assert "exploratory" not in base
    for s in (1, 2):
        new = _raw(f"configs/004_colab_full_b3_base_grpo_seed{s}.yaml")
        assert new["train"].pop("seed") == s
        if s == 2:
            assert new.pop("exploratory") is True
        else:
            assert "exploratory" not in new
        ref = dict(base)
        ref_train = dict(base["train"]); ref_train.pop("seed")
        ref["train"] = ref_train
        ne = dict(new.pop("exec"))
        assert ne.pop("allow_oom_fallback") is False
        re_ = dict(ref.pop("exec"))
        assert ne == re_, f"seed {s}: exec differs beyond the ladder switch"
        assert new == ref, f"seed {s}: differs from seed 0 beyond registered changes"


def test_manifest_records_exploratory_status(tmp_path):
    from obs import write_manifest

    cfg = load_config("configs/004_colab_full_b3_base_grpo_seed2.yaml")
    cfg.output_root = str(tmp_path)
    write_manifest(cfg, str(tmp_path))
    man = json.load(open(tmp_path / "manifest.json"))
    assert man["exploratory"] is True
    cfg1 = load_config("configs/004_colab_full_b3_base_grpo_seed1.yaml")
    cfg1.output_root = str(tmp_path)
    write_manifest(cfg1, str(tmp_path))
    man1 = json.load(open(tmp_path / "manifest.json"))
    assert man1["exploratory"] is False


def test_s4_stage1_notebook_carries_the_registered_guards():
    with open("notebooks/004_colab_3b_base_grpo_seeds12.ipynb") as f:
        notebook = json.load(f)
    source = "\n".join("".join(cell.get("source", [])) for cell in notebook["cells"])
    assert "004_colab_full_b3_base_grpo_seed1" in source
    assert "004_colab_full_b3_base_grpo_seed2" in source
    assert "exploratory" in source
    assert "total_gb >= 79" in source
    assert "base_preds.jsonl" in source and "base_logprob_preds.jsonl" in source
    assert "results/<run_tag> AND checkpoints/<run_tag>" in source
    # executed notebooks legitimately carry filled unit values
    import re as _re
    assert _re.search(r"cu_before = (None|[\d.]+)", source)
    assert _re.search(r"unit_rate = (None|[\d.]+)", source)
    assert _re.search(r"cu_after = (None|[\d.]+)", source)
    assert "_exit_code" in source
    assert "WANDB_PROJECT" not in source
    assert "recorded_cfg = yaml.safe_load(old.stdout)" in source
    assert "recorded_cfg == current_cfg" in source
    assert "seed-0 config values drifted since its recorded run" in source


def test_005_seed_configs_match_except_registered():
    # both new 0.5b seeds are exploratory arms of the adaptive extension
    base = _raw("configs/005_colab_full_b05_base_grpo_seed0.yaml")
    assert base["train"]["seed"] == 0
    assert "allow_oom_fallback" not in base.get("exec", {})
    assert "exploratory" not in base
    for s in (1, 2):
        new = _raw(f"configs/005_colab_full_b05_base_grpo_seed{s}.yaml")
        assert new["train"].pop("seed") == s
        assert new.pop("exploratory") is True
        ref = dict(base)
        ref_train = dict(base["train"]); ref_train.pop("seed")
        ref["train"] = ref_train
        ne = dict(new.pop("exec"))
        assert ne.pop("allow_oom_fallback") is False
        re_ = dict(ref.pop("exec"))
        assert ne == re_, f"seed {s}: exec differs beyond the ladder switch"
        assert new == ref, f"seed {s}: differs from seed 0 beyond registered changes"


def test_s4_stage2_notebook_carries_the_registered_guards():
    with open("notebooks/005_colab_0p5b_base_grpo_seeds12.ipynb") as f:
        notebook = json.load(f)
    source = "\n".join("".join(cell.get("source", [])) for cell in notebook["cells"])
    assert "005_colab_full_b05_base_grpo_seed1" in source
    assert "005_colab_full_b05_base_grpo_seed2" in source
    assert "39 <= total_gb < 45" in source
    assert "base_preds.jsonl" in source and "base_logprob_preds.jsonl" in source
    assert "results/<run_tag> AND checkpoints/<run_tag>" in source
    assert "seed-0 config values drifted since its recorded run" in source
    import re as _re
    assert _re.search(r"cu_before = (None|[\d.]+)", source)
    assert _re.search(r"unit_rate = (None|[\d.]+)", source)
    assert _re.search(r"cu_after = (None|[\d.]+)", source)
    assert "_exit_code" in source
    assert "WANDB_PROJECT" not in source


def test_012_seed_configs_match_except_registered():
    base = _raw("configs/012_colab_full_b15_base_grpo_seed0.yaml")
    assert base["train"]["seed"] == 0
    assert "allow_oom_fallback" not in base.get("exec", {})
    assert "exploratory" not in base
    for s in (1, 2):
        new = _raw(f"configs/012_colab_full_b15_base_grpo_seed{s}.yaml")
        assert new["train"].pop("seed") == s
        assert new.pop("exploratory") is True
        ref = dict(base)
        ref_train = dict(base["train"]); ref_train.pop("seed")
        ref["train"] = ref_train
        ne = dict(new.pop("exec"))
        assert ne.pop("allow_oom_fallback") is False
        re_ = dict(ref.pop("exec"))
        assert ne == re_, f"seed {s}: exec differs beyond the ladder switch"
        assert new == ref, f"seed {s}: differs from seed 0 beyond registered changes"


def test_s4_stage3_notebook_carries_the_registered_guards():
    with open("notebooks/012_colab_1p5b_base_grpo_seeds12.ipynb") as f:
        notebook = json.load(f)
    source = "\n".join("".join(cell.get("source", [])) for cell in notebook["cells"])
    assert "012_colab_full_b15_base_grpo_seed1" in source
    assert "012_colab_full_b15_base_grpo_seed2" in source
    assert "A100 with 80GB required" in source
    assert "total_gb >= 79" in source
    assert "base_preds.jsonl" in source and "base_logprob_preds.jsonl" in source
    assert "results/<run_tag> AND checkpoints/<run_tag>" in source
    assert "ref['resolved_dtype'] == 'bfloat16'" in source
    assert "ref['gpu']['device'] == 'NVIDIA A100-SXM4-80GB'" in source
    assert "seed-0 config values drifted since its recorded run" in source
    import re as _re
    assert _re.search(r"cu_before = (None|[\d.]+)", source)
    assert _re.search(r"unit_rate = (None|[\d.]+)", source)
    assert _re.search(r"cu_after = (None|[\d.]+)", source)
    assert "_exit_code" in source
    assert "WANDB_PROJECT" not in source


def test_013b_seed_configs_match_except_registered():
    # the 7b cell already had the ladder off, so only the seed and the
    # exploratory status may differ
    base = _raw("configs/013_colab_full_b7_base_grpo_seed0.yaml")
    assert base["train"]["seed"] == 0
    assert base["exec"]["allow_oom_fallback"] is False
    assert "exploratory" not in base
    for s in (1, 2):
        new = _raw(f"configs/013_colab_full_b7_base_grpo_seed{s}.yaml")
        assert new["train"].pop("seed") == s
        assert new.pop("exploratory") is True
        ref = dict(base)
        ref_train = dict(base["train"]); ref_train.pop("seed")
        ref["train"] = ref_train
        assert new == ref, f"seed {s}: differs from seed 0 beyond registered changes"


def test_s4_stage4_notebook_carries_the_registered_guards():
    with open("notebooks/013_colab_7b_base_grpo_seeds12.ipynb") as f:
        notebook = json.load(f)
    source = "\n".join("".join(cell.get("source", [])) for cell in notebook["cells"])
    assert "013_colab_full_b7_base_grpo_seed1" in source
    assert "013_colab_full_b7_base_grpo_seed2" in source
    assert "total_gb >= 79" in source
    assert "ref['versions']['cais/mmlu@']" in source
    assert "base_preds.jsonl" in source and "base_logprob_preds.jsonl" in source
    assert "results/<run_tag> AND checkpoints/<run_tag>" in source
    assert "seed-0 config values drifted since its recorded run" in source
    import re as _re
    assert _re.search(r"cu_before = (None|[\d.]+)", source)
    assert _re.search(r"unit_rate = (None|[\d.]+)", source)
    assert _re.search(r"cu_after = (None|[\d.]+)", source)
    assert "_exit_code" in source
    assert "WANDB_PROJECT" not in source
    assert "def base_grpo_arm_complete" in source
    assert "manifest.get('config') == expected" in source
    assert "grpo_preds.jsonl" in source and "grpo_logprob_preds.jsonl" in source
    assert "training and evaluations complete, skipped" in source
