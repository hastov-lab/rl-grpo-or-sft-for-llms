# the 7b pilot config and the ladder escape it depends on
from config import load_config, MODEL_IDS
from enums import Scale


def test_b7_scale_registered():
    assert MODEL_IDS[Scale.B7] == "Qwen/Qwen2.5-7B"


def test_pilot_a_config():
    cfg = load_config("configs/013_pilot_a.yaml")
    assert cfg.scale == Scale.B7
    assert cfg.exploratory is True
    assert cfg.train.max_steps == 30   # exact, per the plan
    # the whole point of pilot A: the untouched original exec settings
    assert cfg.exec.grpo_batch_size is None
    assert cfg.exec.grpo_grad_accum == 1
    assert cfg.exec.gradient_checkpointing is False


def test_run_grpo_accepts_fallback_flag():
    import inspect
    from train.grpo import run_grpo
    sig = inspect.signature(run_grpo)
    assert sig.parameters["allow_oom_fallback"].default is True


def _fake_heavy_modules(monkeypatch, tmp_path, called):
    # run_grpo imports trl/datasets and a few torch touching helpers at call
    # time, so stub just enough of them to walk the control flow offline
    import sys, types
    from types import SimpleNamespace

    class FakeTrainer:
        def __init__(self, **kw):
            self.state = SimpleNamespace(log_history=[])
        def add_callback(self, cb):
            pass
        def train(self, resume_from_checkpoint=None):
            called.append("train")
        def save_model(self, d):
            pass

    trl = types.ModuleType("trl")
    trl.GRPOConfig = lambda **kw: SimpleNamespace(**kw)
    trl.GRPOTrainer = lambda **kw: FakeTrainer(**kw)
    ds = types.ModuleType("datasets")
    ds.Dataset = SimpleNamespace(from_list=lambda rows: rows)
    monkeypatch.setitem(sys.modules, "trl", trl)
    monkeypatch.setitem(sys.modules, "datasets", ds)

    import train.grpo as g
    monkeypatch.setattr(g, "build_grpo_split", lambda limit_per_subject=None: [
        {"prompt": "q\n<think>", "gold": "A", "tier": "easy"}])
    monkeypatch.setattr(g, "set_all_seeds", lambda s: None)
    monkeypatch.setattr(g, "resolve_dtype", lambda c: "float32")
    monkeypatch.setattr(g, "write_adapter_base", lambda *a, **k: None)
    monkeypatch.setattr(g, "make_lora", lambda c: None)  # peft import lives in there

    import obs
    monkeypatch.setattr(obs, "stamp_trainer_logs", lambda t: t)
    monkeypatch.setattr(obs, "make_grpo_sentinel", lambda c: None)

    def fake_ladder(attempt, per_device, accum, checkpointing):
        called.append("ladder")
        return attempt(per_device, accum, checkpointing)
    monkeypatch.setattr(obs, "oom_ladder", fake_ladder)


def test_default_path_uses_the_ladder(monkeypatch, tmp_path):
    called = []
    _fake_heavy_modules(monkeypatch, tmp_path, called)
    from train.grpo import run_grpo
    cfg = load_config("configs/013_pilot_a.yaml")
    cfg.output_root = str(tmp_path)
    run_grpo(cfg, "cpu", out_dir=str(tmp_path / "grpo"), report_to=[])
    assert "ladder" in called and "train" in called


def test_no_fallback_bypasses_the_ladder(monkeypatch, tmp_path):
    called = []
    _fake_heavy_modules(monkeypatch, tmp_path, called)
    from train.grpo import run_grpo
    cfg = load_config("configs/013_pilot_a.yaml")
    cfg.output_root = str(tmp_path)
    run_grpo(cfg, "cpu", out_dir=str(tmp_path / "grpo"), report_to=[], allow_oom_fallback=False)
    assert "ladder" not in called and "train" in called


def test_pilot_notebook_has_no_eval_path():
    # the pilot must never touch the test set, so its code cells may not call
    # any evaluation entrypoint
    import json
    nb = json.load(open("notebooks/013_colab_7b_pilot_a.ipynb"))
    for c in nb["cells"]:
        if c["cell_type"] == "code":
            src = "".join(c["source"])
            # the clone cell globs for scripts/run.py to find the repo root, thats
            # fine, only an actual runner invocation is forbidden
            for bad in ("run_eval", "eval_adapter", "logprob", "python scripts/run.py"):
                assert bad not in src, f"eval path {bad!r} in the pilot notebook"
            assert not src.lstrip().startswith("%%bash")


def test_phase1_config():
    # the reportable 7b seed 0 run: frozen protocol, pilot-locked original exec,
    # ladder off per the no-mutation rule
    cfg = load_config("configs/013_colab_full_b7_seed0.yaml")
    assert cfg.scale == Scale.B7 and not cfg.exploratory
    assert cfg.train.seed == 0 and cfg.train.max_steps == 500
    assert cfg.exec.grpo_batch_size is None
    assert cfg.exec.grpo_grad_accum == 1
    assert cfg.exec.gradient_checkpointing is False
    assert cfg.exec.allow_oom_fallback is False
    # and the default stays on for everything historical
    assert load_config("configs/009_colab_full_b15_seed0.yaml").exec.allow_oom_fallback is True


def _fake_sft_modules(monkeypatch, called):
    # same stubbing idea as the grpo ladder tests, sft flavour
    import sys, types
    from types import SimpleNamespace

    class FakeTrainer:
        def __init__(self, **kw):
            self.state = SimpleNamespace(log_history=[])
        def train(self, resume_from_checkpoint=None):
            called.append("train")
        def save_model(self, d):
            pass

    trl = types.ModuleType("trl")
    trl.SFTConfig = lambda **kw: SimpleNamespace(**kw)
    trl.SFTTrainer = lambda **kw: FakeTrainer(**kw)
    ds = types.ModuleType("datasets")
    ds.Dataset = SimpleNamespace(from_list=lambda rows: rows)
    tf = types.ModuleType("transformers")
    tf.AutoTokenizer = SimpleNamespace(from_pretrained=lambda mid: SimpleNamespace(eos_token="<eos>"))
    monkeypatch.setitem(sys.modules, "trl", trl)
    monkeypatch.setitem(sys.modules, "datasets", ds)
    monkeypatch.setitem(sys.modules, "transformers", tf)

    import train.sft as s
    monkeypatch.setattr(s, "load_sft", lambda split="train", limit=None: [
        {"prompt": "q\n<think>", "target": "t</think>\n<answer>1</answer>"}])
    monkeypatch.setattr(s, "set_all_seeds", lambda x: None)
    monkeypatch.setattr(s, "resolve_dtype", lambda c: "float32")
    monkeypatch.setattr(s, "write_adapter_base", lambda *a, **k: None)
    monkeypatch.setattr(s, "make_lora", lambda c: None)

    import obs
    monkeypatch.setattr(obs, "stamp_trainer_logs", lambda t: t)
    def fake_ladder(attempt, per_device, accum, checkpointing):
        called.append("ladder")
        return attempt(per_device, accum, checkpointing)
    monkeypatch.setattr(obs, "oom_ladder", fake_ladder)


def test_sft_default_path_uses_the_ladder(monkeypatch, tmp_path):
    called = []
    _fake_sft_modules(monkeypatch, called)
    from train.sft import run_sft
    cfg = load_config("configs/013_colab_full_b7_seed0.yaml")
    cfg.output_root = str(tmp_path)
    run_sft(cfg, "cpu", out_dir=str(tmp_path / "sft"), report_to=[])
    assert "ladder" in called and "train" in called


def test_sft_no_fallback_bypasses_the_ladder(monkeypatch, tmp_path):
    called = []
    _fake_sft_modules(monkeypatch, called)
    from train.sft import run_sft
    cfg = load_config("configs/013_colab_full_b7_seed0.yaml")
    cfg.output_root = str(tmp_path)
    run_sft(cfg, "cpu", out_dir=str(tmp_path / "sft"), report_to=[], allow_oom_fallback=False)
    assert "ladder" not in called and "train" in called


def test_013_base_evals_shared_across_conditions():
    # the regrouping decision: both 013 conditions share one base eval dir while
    # their run tags stay distinct, so the expensive 7b base generative eval is
    # a one-time cost paid by phase 1
    from config import run_tag
    from scripts.run import base_eval_dir
    a = load_config("configs/013_colab_full_b7_seed0.yaml")
    b = load_config("configs/013_colab_full_b7_base_grpo_seed0.yaml")
    assert base_eval_dir(a) == base_eval_dir(b)
    assert run_tag(a) != run_tag(b)


def test_phase2_configs_keep_the_no_mutation_rule():
    for f in ("013_colab_full_b7_seed1.yaml", "013_colab_full_b7_base_grpo_seed0.yaml"):
        cfg = load_config(f"configs/{f}")
        assert cfg.scale == Scale.B7 and not cfg.exploratory
        assert cfg.exec.allow_oom_fallback is False, f


def test_base_grpo_differs_from_phase1_only_by_condition():
    # guards against config drift between the two 013 conditions, they must be
    # the same run apart from what is being trained on top of
    import yaml
    a = yaml.safe_load(open("configs/013_colab_full_b7_seed0.yaml"))
    b = yaml.safe_load(open("configs/013_colab_full_b7_base_grpo_seed0.yaml"))
    assert a.pop("condition") == "SFT_GRPO" and b.pop("condition") == "BASE_GRPO"
    assert a == b


def test_phase3_configs_declare_themselves():
    # the 7b supervised control: every field that defines the design is pinned
    # here so config drift cannot pass silently
    from enums import Condition
    for f in ("configs/014_p8_colab_b7.yaml", "configs/015_wsft_colab_b7.yaml"):
        cfg = load_config(f)
        assert cfg.scale == Scale.B7 and cfg.exploratory is True, f
        assert cfg.condition == Condition.SFT_ONLY, f
        assert cfg.train.sft_data == "mmlu_mc", f
        assert cfg.train.max_steps == 123, f
        assert cfg.train.sft_val_examples == 0, f
        assert cfg.eval.max_new_tokens == 512 and cfg.eval.batch_size == 16, f
        assert cfg.eval.skip_generative is True, f
        assert cfg.exec.gradient_checkpointing is False, f
        assert cfg.exec.allow_oom_fallback is False, f
    a = load_config("configs/014_p8_colab_b7.yaml")
    assert not a.train.sft_from_adapter   # base start, no chain
    b = load_config("configs/015_wsft_colab_b7.yaml")
    assert b.train.sft_from_adapter == "checkpoints/013_COLAB_FULL_B7_sft_grpo_seed0/sft"


def test_018_config():
    # the held recovery test: max_steps isnt frozen there (exploratory),
    # doubled budget, seed 1, everything else the frozen values
    cfg = load_config("configs/018_ext_colab_b05.yaml")
    assert cfg.exploratory is True
    assert cfg.train.max_steps == 1000 and cfg.train.seed == 1
    assert cfg.scale.name == "B05"


def test_019_config_matches_018_except_identity():
    # seed 2 recovery must be 018 run with only registered identity
    # changes: experiment id, seed, and ladder switch that now comes from config
    import yaml
    a = yaml.safe_load(open("configs/018_ext_colab_b05.yaml"))
    b = yaml.safe_load(open("configs/019_ext_colab_b05.yaml"))
    assert a.pop("experiment") == "018_ext" and b.pop("experiment") == "019_ext"
    assert a["train"].pop("seed") == 1 and b["train"].pop("seed") == 2
    ea, eb = a.pop("exec", {}), b.pop("exec", {})
    # the ladder binding is the only permitted exec difference, everything else
    # in exec and beyond must match exactly
    assert eb.pop("allow_oom_fallback") is False
    assert ea == eb, "exec differs beyond the ladder binding"
    assert a == b, "019 differs from 018 beyond the registered identity changes"
    cfg = load_config("configs/019_ext_colab_b05.yaml")
    assert cfg.exploratory is True and cfg.train.max_steps == 1000
    assert cfg.exec.allow_oom_fallback is False


def test_020_config_matches_019_except_identity():
    # the final recovery arm must be the 019 run with only the identity moved
    import yaml
    a = yaml.safe_load(open("configs/019_ext_colab_b05.yaml"))
    b = yaml.safe_load(open("configs/020_ext_colab_b05.yaml"))
    assert a.pop("experiment") == "019_ext" and b.pop("experiment") == "020_ext"
    assert a["train"].pop("seed") == 2 and b["train"].pop("seed") == 0
    assert a == b, "020 differs from 019 beyond the registered identity changes"
    cfg = load_config("configs/020_ext_colab_b05.yaml")
    assert cfg.exec.allow_oom_fallback is False and cfg.train.max_steps == 1000


def test_021_configs_identical_except_seed():
    import yaml
    loaded = []
    for s in (0, 1, 2):
        cfg = yaml.safe_load(open(f"configs/021_scratch_colab_b05_seed{s}.yaml"))
        assert cfg["train"].pop("seed") == s
        loaded.append(cfg)
    assert loaded[0] == loaded[1] == loaded[2]


def test_021_configs_match_006_counterparts():
    # each arm must be its 006 counterpart plus exactly the registered
    # differences: experiment id, doubled steps, exploratory flag, ladder off
    import yaml
    for s in (0, 1, 2):
        a = yaml.safe_load(open(f"configs/006_colab_full_b05_seed{s}.yaml"))
        b = yaml.safe_load(open(f"configs/021_scratch_colab_b05_seed{s}.yaml"))
        assert a.pop("experiment") == "006" and b.pop("experiment") == "021_scratch"
        assert b.pop("exploratory") is True and "exploratory" not in a
        assert a["train"].pop("max_steps") == 500 and b["train"].pop("max_steps") == 1000
        ea, eb = a.pop("exec", {}), b.pop("exec", {})
        assert eb.pop("allow_oom_fallback") is False
        assert ea == eb, f"seed {s}: exec differs beyond the ladder binding"
        assert a == b, f"seed {s}: differs from the 006 counterpart beyond registered changes"


def test_013_seed2_config_identical_except_seed():
    import yaml
    loaded = []
    for s in (0, 1, 2):
        cfg = yaml.safe_load(open(f"configs/013_colab_full_b7_seed{s}.yaml"))
        assert cfg["train"].pop("seed") == s
        loaded.append(cfg)
    assert loaded[0] == loaded[1] == loaded[2]


def test_022_configs_identical_except_seed():
    import yaml
    loaded = []
    for s in (0, 1, 2):
        cfg = yaml.safe_load(open(f"configs/022_smooth_colab_b7_seed{s}.yaml"))
        assert cfg["train"].pop("seed") == s
        loaded.append(cfg)
    assert loaded[0] == loaded[1] == loaded[2]


def test_022_configs_match_013_counterparts():
    # each arm must be its 013 counterpart plus exactly the registered
    # differences: experiment id, doubled steps, exploratory flag, grid cadence
    import yaml
    for s in (0, 1, 2):
        a = yaml.safe_load(open(f"configs/013_colab_full_b7_seed{s}.yaml"))
        b = yaml.safe_load(open(f"configs/022_smooth_colab_b7_seed{s}.yaml"))
        assert a.pop("experiment") == "013" and b.pop("experiment") == "022_smooth"
        assert b.pop("exploratory") is True and "exploratory" not in a
        assert a["train"].pop("max_steps") == 500 and b["train"].pop("max_steps") == 1000
        la, lb = a.pop("log", {}), b.pop("log", {})
        assert lb.pop("save_steps") == 100
        assert la.pop("save_total_limit") == 10 and lb.pop("save_total_limit") == 12
        assert la == lb, f"seed {s}: log section differs beyond the grid cadence"
        assert a == b, f"seed {s}: differs from the 013 counterpart beyond registered changes"


def test_022_training_uses_disposable_worker_and_eval_gate():
    import json

    nb = json.load(open("notebooks/022_colab_b7_smooth1000.ipynb"))
    source = "\n".join("".join(c.get("source", [])) for c in nb["cells"])
    assert "scripts/grpo_worker.py" in source
    assert "stream_subprocess(worker_cmd" in source
    assert "active_worker_processes(config_path)" in source
    assert "--query-gpu=memory.free" in source
    assert "free_vram_gib >= 70" in source
    assert "blocked_resume_reason" in source
    assert "worker_returncode" in source
    assert "'resume_blocked': explicit_failure" in source
    assert "worker_status.get('completed') is True" in source
    assert "allowed_infra_files" in source
    assert "resume_compatibility" in source
    assert "'git', 'diff', '--name-only'" in source


def test_003_seed_configs_identical_except_seed():
    import yaml
    loaded = []
    for s in (0, 1, 2):
        cfg = yaml.safe_load(open(f"configs/003_colab_full_b3_seed{s}.yaml"))
        assert cfg["train"].pop("seed") == s
        loaded.append(cfg)
    assert loaded[0] == loaded[1] == loaded[2]


def test_009_seed_configs_identical_except_seed():
    import yaml
    loaded = []
    for s in (0, 1, 2):
        cfg = yaml.safe_load(open(f"configs/009_colab_full_b15_seed{s}.yaml"))
        assert cfg["train"].pop("seed") == s
        loaded.append(cfg)
    assert loaded[0] == loaded[1] == loaded[2]
