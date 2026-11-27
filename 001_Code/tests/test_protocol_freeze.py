# The freeze guard: every experiment config from 003 onward must match the frozen
# protocol in all frozen sections. If this test fails, either the config drifted
# (fix the config) or the protocol is being changed (dont, its frozen)
import glob
import os

import yaml

FROZEN = "configs/protocol_frozen_v1.yaml"
FROZEN_SECTIONS = ["lora", "reward", "grpo", "eval"]
# Per device batch splits are execution, not science: what is frozen is the
# effective sft batch (16). The exec section is free by design (memory knobs)
FROZEN_TRAIN_KEYS = ["sft_lr", "grpo_lr", "max_steps", "num_generations", "max_seq_len",
                     "sft_completion_only", "sft_val_examples", "dtype"]
EFFECTIVE_SFT_BATCH = 16
EXPERIMENTS_049 = {
    "049_p2", "049_p8", "049_p32", "049_wsft_p8", "049_base_grpo",
}


def _load(path):
    with open(path) as f:
        return yaml.safe_load(f)


def _experiment_configs():
    out = []
    for p in sorted(glob.glob("configs/0*.yaml")):
        exp = str(_load(p).get("experiment", "000"))
        if exp >= "003":
            out.append(p)
    return out


def _eval_without_skip(section):
    # skip_generative decides whether the gen eval RUNS, not what it measures,
    # so it stays outside the frozen comparison
    d = dict(section or {})
    d.pop("skip_generative", None)
    return d


def test_frozen_sections_match():
    frozen = _load(FROZEN)
    configs = _experiment_configs()
    assert configs, "no experiment configs from 003 onward yet"
    for p in configs:
        cfg = _load(p)
        if cfg.get("exploratory"):
            # exploratory arms may vary WHAT they train on and for how long, nothing
            # else: dataset, step count, val slice and seed are free, every other
            # frozen value still binds, otherwise their numbers arent comparable
            free = {"max_steps", "sft_data", "sft_val_examples", "seed"}
            action = cfg["train"].get("a2_action", "abort")
            if cfg.get("experiment") in ("030", "031"):
                assert cfg.get("condition") == "EXTERNAL_SFT_GRPO"
                assert cfg["train"].get("num_generations") == 4
                free.add("num_generations")
            elif cfg["train"].get("num_generations") != frozen["train"].get("num_generations"):
                raise AssertionError(f"{p}: only experiments 030 and 031 may change num_generations")
            if cfg.get("experiment") == "031":
                assert action == "record", f"{p}: experiment 031 must record a2"
            elif cfg.get("experiment") == "035":
                assert action == "observe", f"{p}: experiment 035 must observe a2"
                assert cfg.get("condition") in ("MATCHED_LORA_GRPO", "MATCHED_FULL_GRPO")
                assert cfg["train"].get("grpo_precision_mode") == "fp32_master_bf16"
                expected = ("lora" if cfg.get("condition") == "MATCHED_LORA_GRPO" else "full")
                assert cfg["train"].get("grpo_parameter_mode") == expected
            elif cfg.get("experiment") == "046":
                assert action == "observe", f"{p}: experiment 046 must observe a2"
                assert cfg.get("scale") == "GPT2_B15"
                assert cfg.get("condition") == "SFT_GRPO"
                assert cfg["train"].get("sft_parameter_mode") == "lora"
                assert cfg["train"].get("grpo_parameter_mode") == "lora"
                assert cfg["train"].get("sft_precision_mode") == "fp32_master_bf16"
                assert cfg["train"].get("grpo_precision_mode") == "fp32_master_bf16"
            elif cfg.get("experiment") == "048":
                assert action == "observe", f"{p}: experiment 048 must observe a2"
                assert cfg.get("scale") in ("M124", "M355", "M774")
                assert cfg.get("condition") == "SFT_GRPO"
                assert cfg["train"].get("sft_parameter_mode") == "lora"
                assert cfg["train"].get("grpo_parameter_mode") == "lora"
                assert cfg["train"].get("sft_precision_mode") == "fp32_master_bf16"
                assert cfg["train"].get("grpo_precision_mode") == "fp32_master_bf16"
            elif cfg.get("experiment") in {*EXPERIMENTS_049, "051"}:
                exposure_steps = {
                    "049_p2": 31, "049_p8": 123, "049_p32": 490,
                    "049_wsft_p8": 123,
                }
                assert cfg.get("scale") == "GPT2_B15"
                assert cfg["train"].get("sft_parameter_mode") == "lora"
                assert cfg["train"].get("grpo_parameter_mode") == "lora"
                assert cfg["train"].get("sft_precision_mode") == "fp32_master_bf16"
                assert cfg["train"].get("grpo_precision_mode") == "fp32_master_bf16"
                if cfg.get("experiment") in ("049_base_grpo", "051"):
                    assert action == "observe"
                    assert cfg.get("condition") == "BASE_GRPO"
                    assert cfg["train"].get("max_steps") == 500
                    assert not cfg["train"].get("sft_from_adapter")
                    assert not cfg["train"].get("grpo_from_adapter")
                else:
                    assert action == "abort"
                    assert cfg.get("condition") == "SFT_ONLY"
                    assert cfg["train"].get("max_steps") == exposure_steps[
                        cfg.get("experiment")]
                    assert cfg["train"].get("sft_data") == "mmlu_mc"
                    assert cfg["train"].get("sft_val_examples") == 0
                    source = cfg["train"].get("sft_from_adapter")
                    if cfg.get("experiment") == "049_wsft_p8":
                        assert source == (
                            "checkpoints/046_COLAB_FULL_GPT2_B15_sft_grpo_"
                            f"seed{cfg['train']['seed']}/sft")
                    else:
                        assert not source
                    assert not cfg["train"].get("grpo_from_adapter")
            else:
                assert action == "abort", f"{p}: non-default a2 action is not allowed here"
            if cfg.get("experiment") in {"046", "048", *EXPERIMENTS_049, "051"}:
                current = dict(cfg.get("lora") or {})
                original = dict(frozen.get("lora") or {})
                current.pop("target_modules", None)
                original.pop("target_modules", None)
                assert current == original, f"{p}: exploratory lora drifted"
            else:
                assert cfg.get("lora") == frozen.get("lora"), f"{p}: exploratory lora drifted"
            assert cfg.get("reward") == frozen.get("reward"), f"{p}: exploratory reward drifted"
            assert cfg.get("grpo") == frozen.get("grpo"), f"{p}: exploratory grpo section drifted"
            assert _eval_without_skip(cfg.get("eval")) == _eval_without_skip(frozen.get("eval")), \
                f"{p}: exploratory eval protocol drifted"
            for k in FROZEN_TRAIN_KEYS:
                if k in free:
                    continue
                assert cfg["train"].get(k) == frozen["train"].get(k), f"{p}: exploratory train.{k} drifted"
            eff = cfg["train"].get("sft_batch_size", 1) * cfg["train"].get("sft_grad_accum", 1)
            assert eff == EFFECTIVE_SFT_BATCH, f"{p}: exploratory effective sft batch {eff}"
            ex = cfg.get("exec") or {}
            per_device = ex.get("grpo_batch_size") or cfg["train"]["num_generations"]
            assert per_device * ex.get("grpo_grad_accum", 1) == cfg["train"]["num_generations"], \
                f"{p}: exploratory grpo batch x accum must equal num_generations"
            continue
        # confirmatory runs never skip the generative eval, thats the primary metric
        assert not (cfg.get("eval") or {}).get("skip_generative"), f"{p}: confirmatory run skips gen eval"
        # And they never chain sft from an adapter, that would change the frozen
        # pipeline silently. Warm-start sft is an 008 exploratory device only
        assert not cfg["train"].get("sft_from_adapter"), f"{p}: confirmatory run chains sft"
        assert not cfg["train"].get("grpo_from_adapter"), f"{p}: confirmatory run chains grpo"
        scratch_boundary = cfg.get("experiment") == "042"
        if scratch_boundary:
            # check the registered GPT-2 settings against the original protocol
            assert cfg.get("scale") == "M124"
            assert cfg["train"].get("a2_action") == "observe"
            assert cfg["train"].get("sft_parameter_mode") == "full"
            assert cfg["train"].get("grpo_parameter_mode") == "full"
            assert cfg["train"].get("sft_precision_mode") == "fp32_master_bf16"
            assert cfg["train"].get("grpo_precision_mode") == "fp32_master_bf16"
            assert cfg.get("lora", {}).get("target_modules") is None
        else:
            assert cfg["train"].get("a2_action", "abort") == "abort", \
                f"{p}: confirmatory run changes the a2 action"
        for sec in FROZEN_SECTIONS:
            # n=8 is frozen except in the designated ablation, which will declare itself
            if scratch_boundary and sec == "lora":
                current = dict(cfg.get(sec) or {})
                original = dict(frozen.get(sec) or {})
                current.pop("target_modules", None)
                original.pop("target_modules", None)
                assert current == original, f"{p}: section '{sec}' drifted from the frozen protocol"
            else:
                assert _eval_without_skip(cfg.get(sec)) == _eval_without_skip(frozen.get(sec)) if sec == "eval" \
                    else cfg.get(sec) == frozen.get(sec), f"{p}: section '{sec}' drifted from the frozen protocol"
        for k in FROZEN_TRAIN_KEYS:
            assert cfg["train"].get(k) == frozen["train"].get(k), f"{p}: train.{k} drifted"
        eff = cfg["train"].get("sft_batch_size", 1) * cfg["train"].get("sft_grad_accum", 1)
        assert eff == EFFECTIVE_SFT_BATCH, f"{p}: effective sft batch {eff} != {EFFECTIVE_SFT_BATCH}"
        ex = cfg.get("exec") or {}
        per_device = ex.get("grpo_batch_size") or cfg["train"]["num_generations"]
        assert per_device * ex.get("grpo_grad_accum", 1) == cfg["train"]["num_generations"], \
            f"{p}: grpo batch x accum must equal num_generations"


def test_exploratory_configs_exist_and_declare_themselves():
    # the 007 mc-sft sweep must be marked exploratory and keep the frozen eval caps
    ps = sorted(glob.glob("configs/007_*.yaml"))
    assert len(ps) == 6
    for p in ps:
        cfg = _load(p)
        assert cfg.get("exploratory") is True, f"{p} not marked exploratory"
        assert cfg["train"]["sft_data"] == "mmlu_mc"
        assert cfg["eval"]["max_new_tokens"] == 512 and cfg["eval"]["batch_size"] == 16


def test_008_configs_declare_themselves():
    # warm-start mc-sft: exploratory, chained from the exact stage-1 adapters that
    # fed grpo in 006 and 003, frozen eval caps kept
    ps = sorted(glob.glob("configs/008_*.yaml"))
    assert len(ps) == 2
    chains = {}
    for p in ps:
        cfg = _load(p)
        assert cfg.get("exploratory") is True, f"{p} not marked exploratory"
        assert cfg["train"]["sft_data"] == "mmlu_mc"
        assert cfg["train"]["sft_from_adapter"], f"{p} missing the chain adapter"
        assert cfg["eval"]["max_new_tokens"] == 512 and cfg["eval"]["batch_size"] == 16
        chains[cfg["scale"]] = cfg["train"]["sft_from_adapter"]
    assert chains["B05"] == "checkpoints/006_COLAB_FULL_B05_sft_grpo_seed0/sft"
    assert chains["B3"] == "checkpoints/003_COLAB_FULL_B3_sft_grpo_seed0/sft"


def test_1p5b_mirror_configs_declare_themselves():
    # Experiments 010-012 mirror the 007/008/005 designs at 1.5b. The supervised
    # arms stay exploratory with the frozen eval caps, base-grpo is confirmatory
    ps = sorted(glob.glob("configs/010_*.yaml"))
    assert len(ps) == 3
    steps = set()
    for p in ps:
        cfg = _load(p)
        assert cfg.get("exploratory") is True and cfg["scale"] == "B15"
        assert cfg["train"]["sft_data"] == "mmlu_mc"
        assert cfg["eval"]["max_new_tokens"] == 512 and cfg["eval"]["batch_size"] == 16
        steps.add(cfg["train"]["max_steps"])
    assert steps == {31, 123, 490}  # 2, 8 and 32 passes over the 245 prompts
    w = _load("configs/011_wsft_colab_b15.yaml")
    assert w.get("exploratory") is True and w["scale"] == "B15"
    assert w["train"]["sft_from_adapter"] == "checkpoints/009_COLAB_FULL_B15_sft_grpo_seed0/sft"
    b = _load("configs/012_colab_full_b15_base_grpo_seed0.yaml")
    assert not b.get("exploratory") and b["condition"] == "BASE_GRPO" and b["scale"] == "B15"


def test_seed_is_free():
    # seed is deliberately not a frozen key, so each run can pick its own
    frozen = _load(FROZEN)
    assert "seed" not in FROZEN_TRAIN_KEYS
    assert "seed" not in (frozen.get("train") or {})


def test_seed_configs_differ_only_in_seed():
    # Per-seed configs of one experiment must be the same run apart from the seed,
    # otherwise the seeds are not replicates. Covers 003, 006 and 009
    for exp, pattern in (("003", "configs/003_colab_full_b3_seed*.yaml"),
                         ("006", "configs/006_colab_full_b05_seed*.yaml"),
                         ("009", "configs/009_colab_full_b15_seed*.yaml"),
                         ("013", "configs/013_colab_full_b7_seed*.yaml")):
        ps = sorted(glob.glob(pattern))
        assert len(ps) >= 2, f"{exp}: expected at least two seed configs"
        stripped = []
        for p in ps:
            cfg = _load(p)
            cfg["train"].pop("seed")
            stripped.append((p, cfg))
        first_path, first = stripped[0]
        for p, cfg in stripped[1:]:
            assert cfg == first, f"{p} differs from {first_path} beyond the seed"
