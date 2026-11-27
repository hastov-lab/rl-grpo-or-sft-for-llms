import copy
import json
import os
import re
from dataclasses import asdict

import pytest

from checkpoint import config_hash, write_artifact_type
from config import load_config, run_tag
from file_utils import read_json
from obs import a2_verdict_payload
from scripts.s12_eval import ADAPTER_FILES, IntegrityError, atomic_write_json, sha256
from scripts.s28_compose import adapter_hashes_of, merged_hashes
import scripts.s49_gpt2_xl_symmetry as s49


# keep completed 046 checkpoints compatible with the new config allowance
HISTORICAL_046_HASHES = {
    0: "dfaeee6ff04814805397dab6fd6a3fd4a63fd33d56ff2ef7f72c9582aa9cc1f5",
    1: "a8e3d207e1e04f3eeb8760aa6d1ef2828bc47f8b132dd7615ef5411d08c12939",
    2: "8fdd99a87f7da1c2a9c494e5c292f48c5397be390b449beedda511250df5593c",
}


def _changed_fields(left, right, prefix=""):
    if isinstance(left, dict) and isinstance(right, dict):
        changed = set()
        for key in set(left) | set(right):
            name = f"{prefix}.{key}" if prefix else key
            if key not in left or key not in right:
                changed.add(name)
            else:
                changed.update(_changed_fields(left[key], right[key], name))
        return changed
    return set() if left == right else {prefix}


def _write_full_model(path):
    os.makedirs(path, exist_ok=True)
    for name in ("config.json", "tokenizer.json", "tokenizer_config.json",
                 "model.safetensors"):
        with open(os.path.join(path, name), "w") as handle:
            handle.write(name)
    with open(os.path.join(path, "model_base.json"), "w") as handle:
        json.dump({"kind": "hf_id", "value": "gpt2-xl"}, handle)
    write_artifact_type(path, "full", family="scratch_gpt", scale="GPT2_B15")
    return path


def _write_adapter(path, base):
    os.makedirs(path, exist_ok=True)
    for name in ("adapter_model.safetensors", "adapter_config.json",
                 "config_hash.txt"):
        with open(os.path.join(path, name), "w") as handle:
            handle.write(name)
    with open(os.path.join(path, "adapter_base.json"), "w") as handle:
        json.dump({"kind": "local", "value": str(base)}, handle)
    write_artifact_type(path, "adapter", family="scratch_gpt", scale="GPT2_B15")
    return path


def _write_merge_binding(adapter, merged):
    payload = {
        "ts": "2026-08-31T12:00:00",
        "adapter_dir": str(adapter),
        "adapter_sha256": {
            name: sha256(os.path.join(adapter, name)) for name in ADAPTER_FILES
        },
        "merged_sha256": merged_hashes(str(merged)),
    }
    atomic_write_json(os.path.join(merged, "merge_binding.json"), payload)
    return payload


def test_049_roster_has_fifteen_unique_configs_and_tags():
    configs = s49.registered_configs()
    assert set(configs) == {
        (family, seed)
        for family in s49.FAMILY_SPECS_049
        for seed in s49.SEEDS_049
    }
    assert len(configs) == 15
    paths = [s49.config_path(family, seed) for family, seed in configs]
    tags = [run_tag(cfg) for cfg in configs.values()]
    assert len(paths) == len(set(paths)) == 15
    assert len(tags) == len(set(tags)) == 15
    assert all(cfg.scale.name == "GPT2_B15" for cfg in configs.values())
    assert all(cfg.exploratory is True for cfg in configs.values())


def test_049_configs_change_only_the_registered_fields_from_046():
    # compare every nested field instead of maintaining a second expected config
    expected = {
        "p2": {"experiment", "condition", "train.max_steps", "train.sft_data",
               "train.sft_val_examples", "train.a2_action"},
        "p8": {"experiment", "condition", "train.max_steps", "train.sft_data",
               "train.sft_val_examples", "train.a2_action"},
        "p32": {"experiment", "condition", "train.max_steps", "train.sft_data",
                "train.sft_val_examples", "train.a2_action"},
        "wsft_p8": {"experiment", "condition", "train.max_steps", "train.sft_data",
                    "train.sft_val_examples", "train.a2_action",
                    "train.sft_from_adapter"},
        "base_grpo": {"experiment", "condition"},
    }
    for family in s49.FAMILY_SPECS_049:
        for seed in s49.SEEDS_049:
            current = asdict(load_config(s49.config_path(family, seed)))
            source = asdict(load_config(
                f"configs/046_colab_gpt2_b15_seed{seed}.yaml"))
            assert _changed_fields(current, source) == expected[family]


def test_049_seed_replicates_differ_only_by_seed_and_parent_path():
    for family in s49.FAMILY_SPECS_049:
        rows = []
        for seed in s49.SEEDS_049:
            row = asdict(load_config(s49.config_path(family, seed)))
            assert row["train"].pop("seed") == seed
            parent = row["train"].get("sft_from_adapter", "")
            if parent:
                # the warm parent changes with the seed while the recipe stays fixed
                row["train"]["sft_from_adapter"] = parent.replace(
                    f"seed{seed}", "seedX")
            rows.append(row)
        assert rows[0] == rows[1] == rows[2]


def test_049_exposure_and_start_values_are_fixed():
    configs = s49.registered_configs()
    assert {family: configs[(family, 0)].train.max_steps
            for family in ("p2", "p8", "p32", "wsft_p8")} == {
                "p2": 31, "p8": 123, "p32": 490, "wsft_p8": 123,
            }
    for family in ("p2", "p8", "p32", "wsft_p8"):
        cfg = configs[(family, 0)]
        assert cfg.condition.name == "SFT_ONLY"
        assert cfg.train.sft_data == "mmlu_mc"
        assert cfg.train.sft_val_examples == 0
        assert cfg.train.a2_action == "abort"
        assert cfg.eval.skip_generative is False
    base_grpo = configs[("base_grpo", 0)]
    assert base_grpo.condition.name == "BASE_GRPO"
    assert base_grpo.train.max_steps == 500
    assert base_grpo.train.a2_action == "observe"


@pytest.mark.parametrize("old,new", [
    ('experiment: "049_base_grpo"', 'experiment: "049_p8"'),
    ("condition: BASE_GRPO", "condition: SFT_ONLY"),
    ("exploratory: true", "exploratory: false"),
    ("scale: GPT2_B15", "scale: M774"),
])
def test_049_observe_allowance_rejects_near_misses(tmp_path, old, new):
    source = s49.config_path("base_grpo", 0)
    with open(source) as handle:
        text = handle.read()
    assert old in text
    path = tmp_path / "changed.yaml"
    path.write_text(text.replace(old, new))
    with pytest.raises(ValueError, match="observe mode"):
        load_config(path)


def test_049_schema_change_keeps_046_hashes():
    for seed, expected in HISTORICAL_046_HASHES.items():
        cfg = load_config(f"configs/046_colab_gpt2_b15_seed{seed}.yaml")
        assert config_hash(cfg) == expected


def test_049_card_rules_match_the_two_runners():
    assert s49.check_card_vram(39.5, "p8") == "A100_40GB"
    assert s49.check_card_vram(79.3, "base_grpo") == "A100_80GB"
    with pytest.raises(IntegrityError, match="40GB"):
        s49.check_card_vram(79.3, "wsft_p8")
    with pytest.raises(IntegrityError, match="80GB"):
        s49.check_card_vram(39.5, "base_grpo")
    with pytest.raises(IntegrityError, match="not registered"):
        s49.check_card_vram(39.5, "unknown")


def test_049_worker_rejects_a_training_task_from_another_family():
    from scripts.s49_worker import check_task

    sft = load_config(s49.config_path("p8", 0))
    grpo = load_config(s49.config_path("base_grpo", 0))
    assert check_task(sft, "sft") == "p8"
    assert check_task(grpo, "grpo") == "base_grpo"
    assert check_task(sft, "eval") == "p8"
    with pytest.raises(IntegrityError, match="task differs"):
        check_task(sft, "grpo")
    with pytest.raises(IntegrityError, match="task differs"):
        check_task(grpo, "sft")


def test_049_grpo_worker_blocks_terminal_states(tmp_path, monkeypatch):
    from scripts.s49_worker import check_grpo_launch_state

    cfg = load_config(s49.config_path("base_grpo", 0))
    monkeypatch.setattr(
        s49, "arm_state", lambda *args: {"state": "interrupted"})
    assert check_grpo_launch_state(
        cfg, str(tmp_path), "preflight.json") == "interrupted"

    monkeypatch.setattr(s49, "arm_state", lambda *args: {"state": "a1"})
    with pytest.raises(IntegrityError, match="cannot start from state a1"):
        check_grpo_launch_state(cfg, str(tmp_path), "preflight.json")

    monkeypatch.setattr(s49, "arm_state", lambda *args: {"state": "absent"})
    (tmp_path / "049_grpo_training.json").write_text("committed")
    with pytest.raises(IntegrityError, match="cannot start from state absent"):
        check_grpo_launch_state(cfg, str(tmp_path), "preflight.json")


def test_049_grpo_worker_allows_initial_zero_gradient():
    with open("scripts/s49_worker.py") as handle:
        source = handle.read()
    assert "PrecisionAudit(allow_initial_zero_gradient=True)" in source


def test_049_launch_gate_commits_once_for_later_sessions(tmp_path, monkeypatch):
    # later sessions reuse the original funding decision at their lower balance
    preflight_path = tmp_path / "preflight.json"
    preflight_path.write_text(json.dumps({"completed": True}))
    roster = ["arm-a", "arm-b"]
    monkeypatch.setattr(s49, "verify_preflight", lambda path: {"roster": roster})
    launch_path = tmp_path / "launch.json"
    with pytest.raises(IntegrityError, match="at least 350"):
        s49.commit_launch(str(launch_path), str(preflight_path), 349.99)
    record = s49.commit_launch(
        str(launch_path), str(preflight_path), s49.MINIMUM_LAUNCH_UNITS)
    assert record["initial_available_units"] == 350.0
    assert record["roster"] == roster
    assert s49.commit_launch(
        str(launch_path), str(preflight_path), 100.0) == record


def test_049_warm_start_binds_the_seed_matched_046_merge(tmp_path, monkeypatch):
    # the warm branch must use the merge that the matching 046 GRPO arm loaded
    cfg = copy.deepcopy(load_config(s49.config_path("wsft_p8", 1)))
    cfg.output_root = str(tmp_path)
    cfg.train.sft_from_adapter = (
        "checkpoints/046_COLAB_FULL_GPT2_B15_sft_grpo_seed1/sft")
    monkeypatch.setattr(s49, "check_registered_config", lambda value: value)

    adapter = tmp_path / cfg.train.sft_from_adapter
    merged = adapter.parent / "sft_merged"
    _write_adapter(str(adapter), tmp_path / "base")
    _write_full_model(str(merged))
    binding = _write_merge_binding(str(adapter), merged)
    parent = tmp_path / "results" / "046_parent_seed1"
    parent.mkdir(parents=True)
    sft_path = parent / "046_sft_training.json"
    grpo_path = parent / "046_grpo_training.json"
    sft = {"seed": 1, "artifact": {"path": str(adapter)}}
    sft_path.write_text(json.dumps(sft))
    grpo = {
        "seed": 1,
        "sft_record_sha256": sha256(str(sft_path)),
        "merged_path": str(merged),
        "merge_binding": binding,
    }
    grpo_path.write_text(json.dumps(grpo))

    import scripts.s46_gpt2_xl as source
    monkeypatch.setattr(source, "arm_state", lambda path: {"state": "committed"})
    monkeypatch.setattr(
        source, "training_record",
        lambda path, branch: sft if branch == "sft" else grpo)
    record = s49.verify_warm_start(cfg, str(parent))
    assert record["seed"] == 1
    assert record["path"] == os.path.realpath(merged)
    assert record["merge_binding"] == binding

    grpo["seed"] = 2
    with pytest.raises(IntegrityError, match="differs from its 046 parent"):
        s49.verify_warm_start(cfg, str(parent))


def test_049_sft_training_record_binds_start_and_adapter(tmp_path, monkeypatch):
    # completion binds the exact start and adapter bytes before evaluation
    cfg = load_config(s49.config_path("p2", 0))
    result_dir = tmp_path / "result"
    output_dir = tmp_path / "adapter"
    start_dir = tmp_path / "base"
    result_dir.mkdir()
    start_dir.mkdir()
    _write_adapter(str(output_dir), start_dir)
    rows = [{"step": step, "loss": 1.0 / step}
            for step in range(1, cfg.train.max_steps + 1)]
    (result_dir / "sft_trainlog.json").write_text(json.dumps(rows))
    preflight_path = tmp_path / "preflight.json"
    preflight_path.write_text("{}")
    wandb_path = tmp_path / "wandb.json"
    wandb_path.write_text("{}")
    wandb = [{"path": str(wandb_path), "sha256": sha256(str(wandb_path))}]
    start = {"kind": "immutable_base", "path": str(start_dir)}
    monkeypatch.setattr(s49, "start_for", lambda cfg_arg, path: start)
    monkeypatch.setattr(s49, "wandb_records", lambda *args, **kwargs: wandb)

    record = s49.commit_sft_training(
        cfg, str(result_dir), str(output_dir), str(preflight_path))
    assert record["family"] == "p2"
    assert record["updates"] == 31
    assert record["start"] == start
    assert record["artifact"]["sha256"] == adapter_hashes_of(str(output_dir))
    assert s49.training_record(cfg, str(result_dir)) == record

    trainlog_path = result_dir / "sft_trainlog.json"
    trainlog = trainlog_path.read_text()
    trainlog_path.write_text(trainlog.replace('"loss": 1.0', '"loss": 2.0'))
    with pytest.raises(IntegrityError, match="training record changed"):
        s49.training_record(cfg, str(result_dir))
    trainlog_path.write_text(trainlog)

    monkeypatch.setattr(s49, "wandb_records", lambda *args, **kwargs: [])
    with pytest.raises(IntegrityError, match="W&B records"):
        s49.training_record(cfg, str(result_dir))
    monkeypatch.setattr(s49, "wandb_records", lambda *args, **kwargs: wandb)

    with open(output_dir / "adapter_model.safetensors", "a") as handle:
        handle.write("changed")
    with pytest.raises(IntegrityError, match="training record changed"):
        s49.training_record(cfg, str(result_dir))


def test_049_grpo_record_rechecks_precision_and_a2_files(tmp_path, monkeypatch):
    # the GRPO endpoint stays tied to the precision and sentinel evidence
    cfg = load_config(s49.config_path("base_grpo", 0))
    result_dir = tmp_path / "result"
    output_dir = tmp_path / "adapter"
    start_dir = tmp_path / "base"
    result_dir.mkdir()
    start_dir.mkdir()
    _write_adapter(str(output_dir), start_dir)
    rows = [{
        "step": step, "loss": 1.0 / step,
        "rewards/accuracy_reward/mean": 0.1,
        "rewards/format_reward/mean": 0.2,
        "frac_reward_zero_std": 0.5,
        "kl": 0.01,
    } for step in range(1, 501)]
    (result_dir / "grpo_trainlog.json").write_text(json.dumps(rows))
    preflight_path = tmp_path / "preflight.json"
    preflight_path.write_text("{}")
    wandb_path = tmp_path / "wandb.json"
    wandb_path.write_text("{}")
    wandb = [{"path": str(wandb_path), "sha256": sha256(str(wandb_path))}]
    start = {"kind": "immutable_base", "path": str(start_dir)}
    monkeypatch.setattr(s49, "start_for", lambda cfg_arg, path: start)
    monkeypatch.setattr(s49, "wandb_records", lambda *args, **kwargs: wandb)
    adapter_base = s49.verify_adapter_base(cfg, str(output_dir), start)
    precision_path = result_dir / "precision_report.json"
    precision = {
        "gradient_dtypes": ["float32"],
        "optimizer_state_dtypes": ["float32"],
        "first_nonzero_gradient_update": 1,
        "bf16_enabled": True,
        "accelerator_mixed_precision": "bf16",
        "reference_gate": {"reference_kind": "adapter_disabled"},
        "reference_behaviour": {"verified": True},
        "adapter_base": adapter_base,
        "lora_setup": {"verified": True,
                       "target_module_counts": {"W_query": 48}},
    }
    precision_path.write_text(json.dumps(precision))
    verdict_path = output_dir / "a2_verdict.json"
    verdict_path.write_text(json.dumps({"completed": True}))
    with pytest.raises(IntegrityError, match="A2 verdict is invalid"):
        s49.commit_grpo_training(
            cfg, str(result_dir), str(output_dir), str(precision_path),
            str(preflight_path))
    verdict_path.write_text(json.dumps(a2_verdict_payload(cfg, rows[:150])))

    record = s49.commit_grpo_training(
        cfg, str(result_dir), str(output_dir), str(precision_path),
        str(preflight_path))
    assert s49.training_record(cfg, str(result_dir)) == record

    record_path = result_dir / "049_grpo_training.json"
    changed_record = dict(record)
    changed_record["starvation"] = {"changed": True}
    atomic_write_json(str(record_path), changed_record)
    with pytest.raises(IntegrityError, match="GRPO evidence changed"):
        s49.training_record(cfg, str(result_dir))
    atomic_write_json(str(record_path), record)

    precision_path.write_text(json.dumps({**precision, "changed": True}))
    with pytest.raises(IntegrityError, match="GRPO evidence changed"):
        s49.training_record(cfg, str(result_dir))


def test_049_a1_conclusion_is_exact_and_forbids_late_endpoint(tmp_path, monkeypatch):
    # a stopped base GRPO arm cannot gain a result after its conclusion commits
    cfg = load_config(s49.config_path("base_grpo", 0))
    result_dir = tmp_path / "result"
    result_dir.mkdir()
    preflight_path = tmp_path / "preflight.json"
    preflight_path.write_text("{}")
    start = {"kind": "immutable_base", "path": "/bound/base"}
    monkeypatch.setattr(s49, "start_for", lambda cfg_arg, path: start)
    wandb = [{"run_id": "run-1", "run_name": "base_grpo"}]
    monkeypatch.setattr(s49, "wandb_records", lambda *args, **kwargs: wandb)
    status = {
        "completed": False,
        "cuda_oom": False,
        "task": "grpo",
        "seed": 0,
        "family": "base_grpo",
        "error_type": "RuntimeError",
        "error_message": s49.A1_ERROR,
    }
    status_path = result_dir / "grpo_worker_status.json"
    for field, value in (("task", "sft"), ("seed", 1), ("family", "p8")):
        changed = dict(status)
        changed[field] = value
        status_path.write_text(json.dumps(changed))
        assert s49.a1_state(
            cfg, str(result_dir), str(preflight_path))[0] == "failed"
    status_path.write_text(json.dumps(status))

    assert s49.a1_state(cfg, str(result_dir), str(preflight_path))[0] == "pending"
    monkeypatch.setattr(s49, "wandb_records", lambda *args, **kwargs: [])
    with pytest.raises(IntegrityError, match="W&B record is missing"):
        s49.commit_a1_conclusion(cfg, str(result_dir), str(preflight_path))
    monkeypatch.setattr(s49, "wandb_records", lambda *args, **kwargs: wandb)
    record = s49.commit_a1_conclusion(
        cfg, str(result_dir), str(preflight_path))
    assert record["outcome"] == "a1"
    assert s49.a1_state(
        cfg, str(result_dir), str(preflight_path))[0] == "committed"

    record_path = result_dir / "049_a1_conclusion.json"
    changed_record = dict(record)
    changed_record["family"] = "p8"
    atomic_write_json(str(record_path), changed_record)
    with pytest.raises(IntegrityError, match="A1 conclusion changed"):
        s49.a1_state(cfg, str(result_dir), str(preflight_path))
    atomic_write_json(str(record_path), record)

    (result_dir / "049_grpo_eval_manifest.json").write_text("{}")
    with pytest.raises(IntegrityError, match="gained"):
        s49.a1_state(cfg, str(result_dir), str(preflight_path))


def test_049_arm_completion_halts_after_manifest_drift(tmp_path, monkeypatch):
    cfg = load_config(s49.config_path("p8", 0))
    result_dir = tmp_path / "result"
    result_dir.mkdir()
    preflight_path = tmp_path / "preflight.json"
    preflight_path.write_text("{}")
    training_path = result_dir / "049_sft_training.json"
    manifest_path = result_dir / "049_sft_eval_manifest.json"
    training_path.write_text("training")
    manifest_path.write_text("manifest")
    binding = {"completed": True, "family": "p8"}
    monkeypatch.setattr(s49, "training_record", lambda *args: binding)
    monkeypatch.setattr(s49, "eval_state", lambda *args: "committed")
    record = s49.commit_arm_complete(
        cfg, str(result_dir), str(preflight_path))
    assert s49.arm_state(
        cfg, str(result_dir), str(preflight_path))["state"] == "committed"

    manifest_path.write_text("changed")
    with pytest.raises(IntegrityError, match="changed after completion"):
        s49.arm_state(cfg, str(result_dir), str(preflight_path))


def test_049_analysis_requires_all_three_fixed_pairs(tmp_path, monkeypatch):
    # every registered family needs all three fixed pairs for formal inference
    monkeypatch.setattr(s49, "verify_preflight", lambda path: {"starts": {}})
    monkeypatch.setattr(
        s49, "arm_state", lambda *args: {"state": "committed"})
    record = s49.analysis_inputs("F2a", str(tmp_path), "preflight.json")
    assert record["pair_count"] == 3
    assert record["formal_analysis_allowed"] is True
    assert all(set(pair["endpoints"]) == {"p2", "p8"}
               for pair in record["pairs"])

    def one_missing(cfg, *args):
        state = "absent" if cfg.train.seed == 2 else "committed"
        return {"state": state}

    monkeypatch.setattr(s49, "arm_state", one_missing)
    with pytest.raises(IntegrityError, match="needs three completed pairs"):
        s49.analysis_inputs("F2a", str(tmp_path), "preflight.json")


def test_049_f4_analysis_rechecks_historical_046_endpoints(tmp_path, monkeypatch):
    import scripts.s46_gpt2_xl as source

    # the historical side is reverified before it enters a new family
    monkeypatch.setattr(s49, "verify_preflight", lambda path: {"starts": {}})
    monkeypatch.setattr(
        s49, "arm_state", lambda *args: {"state": "committed"})
    monkeypatch.setattr(
        source, "arm_state", lambda path: {"state": "committed"})
    monkeypatch.setattr(
        source, "training_record", lambda path, branch: {"stage": branch})
    checked = []

    def committed_endpoint(path, branch, binding):
        checked.append((path, branch, binding))
        return "committed"

    monkeypatch.setattr(source, "eval_state", committed_endpoint)
    record = s49.analysis_inputs("F4", str(tmp_path), "preflight.json")
    assert record["formal_analysis_allowed"] is True
    assert len(checked) == 3
    assert all("historical_046_grpo" in pair for pair in record["pairs"])


def test_049_notebooks_match_launch_or_closeout():
    paths = (
        "notebooks/049_colab_gpt2_xl_mc_sft.ipynb",
        "notebooks/049_colab_gpt2_xl_base_grpo.ipynb",
    )
    for path in paths:
        with open(path) as handle:
            notebook = json.load(handle)
        text = "\n".join("".join(cell.get("source", []))
                         for cell in notebook["cells"])
        before = re.search(r"^cu_before =\s*(None|[\d.]+)$", text, re.MULTILINE)
        after = re.search(r"^cu_after =\s*(None|[\d.]+)$", text, re.MULTILINE)
        assert before is not None and after is not None
        outputs = [output for cell in notebook["cells"]
                   for output in cell.get("outputs", [])]
        if not outputs:
            assert before.group(1) == "None"
            assert after.group(1) == "None"
            assert "SEEDS_TO_RUN = (0,)" in text
        else:
            assert before.group(1) != "None"
            if after.group(1) != "None":
                assert float(before.group(1)) >= float(after.group(1)) >= 0
            assert ("SEEDS_TO_RUN = (0,)" in text
                    or "SEEDS_TO_RUN = (1, 2)" in text)
            assert all(output.get("output_type") != "error" for output in outputs)
        assert "commit_preflight(" in text
        assert "scripts/s49_worker.py" in text
        assert text.index("commit_preflight(") < text.index("def run_arm")
        for index, cell in enumerate(notebook["cells"]):
            if cell["cell_type"] == "code":
                assert (cell.get("execution_count") is None
                        or isinstance(cell.get("execution_count"), int))
                compile("".join(cell.get("source", [])), f"{path}:{index}", "exec")


@pytest.mark.parametrize("path", (
    "notebooks/049_colab_gpt2_xl_mc_sft.ipynb",
    "notebooks/049_colab_gpt2_xl_base_grpo.ipynb",
))
def test_049_notebook_records_interrupts_and_preserves_failed_eval(tmp_path, path):
    with open(path) as handle:
        notebook = json.load(handle)
    helper = next(
        "".join(cell.get("source", [])) for cell in notebook["cells"]
        if "def run_worker" in "".join(cell.get("source", [])))
    arm = next(
        "".join(cell.get("source", [])) for cell in notebook["cells"]
        if "def run_arm" in "".join(cell.get("source", [])))
    namespace = {
        "os": os,
        "json": json,
        "shutil": __import__("shutil"),
        "time": __import__("time"),
        "read_json": read_json,
        "IntegrityError": IntegrityError,
        "RESULTS": str(tmp_path),
        "session_id": "test_session",
    }
    exec(helper, namespace)
    namespace["active_workers"] = lambda: []
    result_dir = tmp_path / "result"
    result_dir.mkdir()
    status_path = result_dir / "eval_status.json"
    scratch = tmp_path / "partial_eval"
    scratch.mkdir()
    rows = "".join(json.dumps({
        "subject": "abstract_algebra", "tier": "hard", "gold": "A",
        "row": index,
    }) + "\n" for index in range(1884))
    # a complete generative pass counts even when the later probe never starts
    (scratch / "sft_preds.jsonl").write_text(rows)
    values = ["worker", "--scratch-dir", str(scratch)]

    def interrupt(_):
        status_path.write_text(json.dumps({
            "completed": False, "error_type": "KeyboardInterrupt"}))
        raise KeyboardInterrupt

    namespace["stream_subprocess"] = interrupt
    with pytest.raises(KeyboardInterrupt):
        namespace["run_worker"](
            values, str(status_path), str(result_dir), "eval", 2)
    attempts = (result_dir / "049_attempts.jsonl").read_text().splitlines()
    ledger = (tmp_path / "compute_ledger.jsonl").read_text().splitlines()
    assert len(attempts) == 1
    assert len(ledger) == 1
    attempt = json.loads(attempts[0])
    assert attempt["mmlu_test_passes"] == 1
    assert json.loads(ledger[0])["mmlu_test_passes"] == 1
    archive = attempt["eval_scratch_archive"]
    assert os.path.isfile(os.path.join(archive, "sft_preds.jsonl"))

    assert namespace["observed_test_passes"](values) == 1
    # two complete prediction files count both passes before metric writes
    (scratch / "sft_logprob_preds.jsonl").write_text(rows)
    assert namespace["observed_test_passes"](values) == 2

    status_path.write_text(json.dumps({"completed": True}))
    with pytest.raises(IntegrityError, match="no committed result"):
        namespace["run_worker"](
            ["worker"], str(status_path), str(result_dir), "eval", 2)
    assert "completed evaluation has no manifest" in arm
    assert arm.index("completed evaluation has no manifest") < arm.index(
        "scratch = fresh_scratch")
    assert "if status.get('completed') is True:\n            shutil.rmtree" in arm
    assert "evaluation scratch kept:" in arm
