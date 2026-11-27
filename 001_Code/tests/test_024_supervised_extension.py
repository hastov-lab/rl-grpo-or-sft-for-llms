import json
from pathlib import Path

import yaml

from config import load_config, run_tag
from enums import Condition, Scale
from scripts.run import base_eval_dir


def _raw(path):
    with open(path) as f:
        return yaml.safe_load(f)


def test_024_configs_declare_the_registered_design():
    for seed in (0, 1, 2):
        for prefix in ("024_p16", "024_wsft_p16"):
            path = f"configs/{prefix}_colab_b7_seed{seed}.yaml"
            cfg = load_config(path)
            assert cfg.experiment == prefix and cfg.exploratory is True
            assert cfg.scale == Scale.B7 and cfg.condition == Condition.SFT_ONLY
            assert cfg.train.seed == seed and cfg.train.max_steps == 245
            assert cfg.train.sft_data == "mmlu_mc" and cfg.train.sft_val_examples == 0
            assert cfg.train.sft_batch_size * cfg.train.sft_grad_accum == 16
            assert cfg.eval.skip_generative is True
            assert cfg.eval.max_new_tokens == 512 and cfg.eval.batch_size == 16
            assert cfg.exec.gradient_checkpointing is False
            assert cfg.exec.allow_oom_fallback is False
            assert cfg.log.wandb_project == "inm363-grpo"
            assert run_tag(cfg) == f"{prefix}_COLAB_FULL_B7_sft_only_seed{seed}"


def test_024_base_configs_match_014_except_registered_fields():
    for seed in (0, 1, 2):
        old = _raw("configs/014_p8_colab_b7.yaml")
        new = _raw(f"configs/024_p16_colab_b7_seed{seed}.yaml")
        assert old.pop("experiment") == "014_p8"
        assert new.pop("experiment") == "024_p16"
        assert old["train"].pop("max_steps") == 123
        assert new["train"].pop("max_steps") == 245
        assert old["train"].pop("seed") == 0
        assert new["train"].pop("seed") == seed
        assert old == new


def test_024_warm_configs_match_015_with_seed_matched_sources():
    for seed in (0, 1, 2):
        old = _raw("configs/015_wsft_colab_b7.yaml")
        new = _raw(f"configs/024_wsft_p16_colab_b7_seed{seed}.yaml")
        assert old.pop("experiment") == "015_wsft"
        assert new.pop("experiment") == "024_wsft_p16"
        assert old["train"].pop("max_steps") == 123
        assert new["train"].pop("max_steps") == 245
        assert old["train"].pop("seed") == 0
        assert new["train"].pop("seed") == seed
        assert old["train"].pop("sft_from_adapter").endswith("seed0/sft")
        source = new["train"].pop("sft_from_adapter")
        assert source == f"checkpoints/013_COLAB_FULL_B7_sft_grpo_seed{seed}/sft"
        assert old == new


def test_024_notebook_carries_the_registered_guards():
    with open("notebooks/024_colab_b7_mc_sft_p16.ipynb") as f:
        notebook = json.load(f)
    source = "\n".join("".join(cell.get("source", [])) for cell in notebook["cells"])
    assert "SEEDS_TO_RUN = (0,)" in source
    assert "SEEDS_TO_RUN = (0, 1, 2)" in source
    assert "VARIANTS = ('base', 'warm')" in source
    assert "cfg.train.max_steps == 245" in source
    assert "cfg.exec.allow_oom_fallback is False" in source
    assert "cfg.log.wandb_project == 'inm363-grpo'" in source
    assert "source_binding.json" in source
    assert "024_execution_binding.json" in source
    assert "notebook_protocol_sha256" in source
    assert "scientific_file_hashes" in source
    assert "execution_protocol_sha256" in source
    assert "sft_eval_binding.json" in source
    assert "source base file drifted" in source
    assert "set(range(1, 246))" in source
    assert "024_comparison_summary.json" in source
    assert "warm_vs_smooth_grpo" in source
    assert "exact_preparation_match" in source
    assert "rows == 1884" in source
    assert "total_gb >= 79" in source


def test_024_completed_arms_skip_before_the_full_runner():
    with open("notebooks/024_colab_b7_mc_sft_p16.ipynb") as f:
        notebook = json.load(f)
    cells = ["".join(cell.get("source", [])) for cell in notebook["cells"]]
    cell = next(source for source in cells if "--stage all" in source)
    check = "if training_probe_complete(cfg_path, tag):"
    assert check in cell
    assert "continue" in cell
    assert cell.index(check) < cell.index("continue") < cell.index("--stage all")
    for name in ("sft_trainlog.json", "sft_logprob.json", "sft_logprob_preds.jsonl",
                 "manifest.json", "summary.json"):
        assert name in cell


def test_024_common_reading_rejects_inconclusive_consensus():
    with open("notebooks/024_colab_b7_mc_sft_p16.ipynb") as f:
        notebook = json.load(f)
    source = "\n".join("".join(cell.get("source", [])) for cell in notebook["cells"])
    allowed = "categories[0] in ('equivalent', 'material improvement', 'material decline')"
    assert allowed in source
    assert "categories[0] if len(set(categories)) == 1" not in source


def test_024_staged_passes_bind_scientific_code_but_ignore_the_seed_selector():
    with open("notebooks/024_colab_b7_mc_sft_p16.ipynb") as f:
        notebook = json.load(f)
    source = "\n".join("".join(cell.get("source", [])) for cell in notebook["cells"])
    assert "if 'SEEDS_TO_RUN =' not in line" in source
    assert "024 scientific code changed between staged passes" in source
    assert "'analysis', 'data', 'eval', 'train'" in source
    for path in ("scripts/build_data.py", "scripts/env_switch.py", "scripts/run.py",
                 "scripts/eval_adapter.py"):
        assert path in source


def test_024_reuses_the_014_base_probe_before_the_runner_starts():
    with open("notebooks/024_colab_b7_mc_sft_p16.ipynb") as f:
        notebook = json.load(f)
    source = "\n".join("".join(cell.get("source", [])) for cell in notebook["cells"])
    assert "results', 'base_evals', '014_p8_COLAB_FULL_B7'" in source
    assert "probe_files = ('base_logprob.json', 'base_logprob_preds.jsonl')" in source
    assert "for experiment in ('024_p16', '024_wsft_p16')" in source
    assert "base_probe_reuse.json" in source
    assert "source_hashes[fname] = sha(src)" in source
    assert "generative base eval not reused" in source
    assert source.index("base_probe_reuse.json") < source.index("## 9. Wiring check")

    for experiment in ("024_p16", "024_wsft_p16"):
        cfg = load_config(f"configs/{experiment}_colab_b7_seed0.yaml")
        expected = Path(cfg.output_root) / "results" / "base_evals" / f"{experiment}_COLAB_FULL_B7"
        assert Path(base_eval_dir(cfg)) == expected


def test_025_configs_identical_except_seed():
    for prefix in ("025_p2", "025_p32"):
        loaded = []
        for seed in (0, 1, 2):
            cfg = _raw(f"configs/{prefix}_colab_b7_seed{seed}.yaml")
            assert cfg["train"].pop("seed") == seed
            loaded.append(cfg)
        assert loaded[0] == loaded[1] == loaded[2]


def test_025_configs_match_014_except_registered_fields():
    for prefix, steps in (("025_p2", 31), ("025_p32", 490)):
        for seed in (0, 1, 2):
            old = _raw("configs/014_p8_colab_b7.yaml")
            new = _raw(f"configs/{prefix}_colab_b7_seed{seed}.yaml")
            assert old.pop("experiment") == "014_p8"
            assert new.pop("experiment") == prefix
            assert old["train"].pop("max_steps") == 123
            assert new["train"].pop("max_steps") == steps
            assert old["train"].pop("seed") == 0
            assert new["train"].pop("seed") == seed
            assert old == new


def test_025_notebook_carries_the_registered_guards():
    with open("notebooks/025_colab_b7_exposure_ends.ipynb") as f:
        notebook = json.load(f)
    source = "\n".join("".join(cell.get("source", [])) for cell in notebook["cells"])
    assert "SEEDS_TO_RUN = (0,)" in source
    assert "SEEDS_TO_RUN = (0, 1, 2)" in source
    # executed notebooks legitimately carry filled unit values, so the guard
    # checks the fields exist rather than demanding blanks
    import re as _re
    assert _re.search(r"cu_before = (None|[\d.]+)", source)
    assert _re.search(r"unit_rate = (None|[\d.]+)", source)
    assert _re.search(r"cu_after = (None|[\d.]+)", source)
    assert "STEPS = {'p2': 31, 'p32': 490}" in source
    assert "025_execution_binding.json" in source
    assert "config_file_hashes" in source
    assert "execution_protocol_sha256" in source
    assert "base_probe_reuse.json" in source
    assert "base_source, 'base_logprob_preds.jsonl')) as f" in source
    assert "training_probe_complete" in source
    assert "sft_eval_binding.json" in source
    assert "rows == 1884" in source
    assert "hard_probe_ci" in source
    assert "'hard_probe', 'hard_margin')" in source
    assert "cfg.log.wandb_project == 'inm363-grpo'" in source
    assert "cfg.exec.allow_oom_fallback is False" in source
    assert "'material improvement', 'material decline')" in source
