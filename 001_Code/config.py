# all params come from here or a yaml, nothing hard coded in the pipeline
from dataclasses import dataclass, field
import yaml

from enums import Env, Mode, Scale, Condition

SMOKE_MODEL_ID = "sshleifer/tiny-gpt2"  # tiny stand in for local smoke wiring

# Map each scale to its hf model. The 124m model loads into the from scratch gpt2 path
MODEL_IDS = {
    Scale.M124: "gpt2",
    Scale.M355: "gpt2-medium",
    Scale.M774: "gpt2-large",
    Scale.GPT2_B15: "gpt2-xl",
    Scale.B05: "Qwen/Qwen2.5-0.5B",
    Scale.B15: "Qwen/Qwen2.5-1.5B",
    Scale.B7: "Qwen/Qwen2.5-7B",
    Scale.B3: "Qwen/Qwen2.5-3B",
}

SCRATCH_GPT_SCALES = frozenset({Scale.M124, Scale.M355, Scale.M774, Scale.GPT2_B15})


@dataclass
class LoraConfig:
    r: int = 16
    alpha: int = 32
    dropout: float = 0.05
    target_modules: list | None = None   # none lets peft pick its defaults


@dataclass
class RewardConfig:
    correct_weight: float = 1.0
    format_weight: float = 0.2
    min_think_chars: int = 20   # format reward needs a real think block, not an empty one


@dataclass
class TrainConfig:
    sft_lr: float = 1.0e-4      # lora sft convention, decided 10 jul
    grpo_lr: float = 1.0e-5
    max_steps: int = 500
    num_generations: int = 8    # this is N in grpo
    max_seq_len: int = 1024
    sft_batch_size: int = 4
    sft_grad_accum: int = 4     # effective batch 16, so 500 steps is about one gsm8k epoch
    sft_completion_only: bool = True   # loss on the tagged response only, not the question
    sft_data: str = "gsm8k"     # mmlu_mc for the control; synthetic_facts only in 034
    sft_val_examples: int = 500  # gsm8k test slice for val loss during sft, 0 turns it off
    # Experiment 008: start sft from an existing adapter instead of the raw base.
    # Path relative to output_root, empty means off. Exploratory runs only, the
    # freeze test rejects it on confirmatory configs
    sft_from_adapter: str = ""
    # Experiment 028: start grpo from an external adapter (merged into the base
    # first). Same rules as sft_from_adapter: relative to output_root, empty
    # means off, rejected on confirmatory configs
    grpo_from_adapter: str = ""
    sft_parameter_mode: str = "lora"
    sft_precision_mode: str = "legacy"
    grpo_parameter_mode: str = "lora"  # lora normally; full is reserved for experiment 035
    grpo_precision_mode: str = "legacy"  # matched fp32 storage is reserved for experiment 035
    a2_action: str = "abort"      # abort normally; record and observe are registered exceptions
    dtype: str = "auto"         # auto, bfloat16, float16 or float32
    seed: int = 0


@dataclass
class GrpoConfig:
    # protocol decisions of 10 jul: vanilla grpo (keeps the
    # length bias diagnostics meaningful) with the original 0.04 kl, all pinned
    # explicitly so a trl version bump cant change the algorithm under me
    loss_type: str = "grpo"
    beta: float = 0.04
    scale_rewards: str = "group"
    temperature: float = 1.0
    top_p: float = 1.0
    max_completion_length: int = 384   # rollout cap, biggest single cost lever


@dataclass
class EvalConfig:
    max_new_tokens: int = 384
    batch_size: int = 16
    skip_generative: bool = False   # skip generation for cheap exploratory probe sweeps


@dataclass
class ExecConfig:
    # Execution knobs, deliberately OUTSIDE the frozen protocol: they exist so a
    # bigger model can fit in memory without changing the science. The effective
    # grpo batch stays num_generations because batch x accum must equal it
    grpo_batch_size: int | None = None   # none = num_generations, the default
    grpo_grad_accum: int = 1
    gradient_checkpointing: bool = False
    max_train_hours: float | None = None  # sentinel a4: abort if the projection blows past this
    # 7B runs keep this false, so each attempt either finishes with its declared settings or fails
    allow_oom_fallback: bool = True


@dataclass
class LogConfig:
    # log as much as is useful: trackers, checkpoint cadence, how many rollouts to keep
    trackers: tuple = ("tensorboard", "wandb")
    save_steps: int = 50            # periodic checkpoints for crash resume on long runs
    save_total_limit: int = 2       # keep the last couple, not every step
    log_completions: bool = True
    num_sample_completions: int = 20
    wandb_project: str = "inm363-grpo"
    wandb_mode: str = "offline"     # flip to online after the wandb login is available for real runs
    wandb_key_file: str = "wandb_api_key.txt"  # gitignored, read at run time so no manual login


@dataclass
class RunConfig:
    experiment: str = "000"   # numbered experiment id, 000 = smoke/infrastructure
    exploratory: bool = False  # exploratory arms dont bind to the frozen training protocol
    env: Env = Env.LOCAL
    mode: Mode = Mode.SMOKE
    scale: Scale = Scale.B05
    condition: Condition = Condition.SFT_GRPO
    lora: LoraConfig = field(default_factory=LoraConfig)
    reward: RewardConfig = field(default_factory=RewardConfig)
    train: TrainConfig = field(default_factory=TrainConfig)
    grpo: GrpoConfig = field(default_factory=GrpoConfig)
    eval: EvalConfig = field(default_factory=EvalConfig)
    exec: ExecConfig = field(default_factory=ExecConfig)
    log: LogConfig = field(default_factory=LogConfig)
    output_root: str = "."   # where results/ and checkpoints/ go; point at drive on colab


def _pick(enum_cls, name):
    # turn the yaml string into an enum, blow up early if its a typo
    try:
        return enum_cls[name]
    except KeyError:
        raise ValueError(f"bad {enum_cls.__name__} value: {name!r}")


def load_config(path):
    with open(path) as f:
        raw = yaml.safe_load(f) or {}
    cfg = RunConfig(
        experiment=str(raw.get("experiment", "000")),
        exploratory=bool(raw.get("exploratory", False)),
        env=_pick(Env, raw.get("env", "LOCAL")),
        mode=_pick(Mode, raw.get("mode", "SMOKE")),
        scale=_pick(Scale, raw.get("scale", "B05")),
        condition=_pick(Condition, raw.get("condition", "SFT_GRPO")),
        lora=LoraConfig(**raw.get("lora", {})),
        reward=RewardConfig(**raw.get("reward", {})),
        train=TrainConfig(**raw.get("train", {})),
        grpo=GrpoConfig(**raw.get("grpo", {})),
        eval=EvalConfig(**raw.get("eval", {})),
        exec=ExecConfig(**raw.get("exec", {})),
        log=LogConfig(**raw.get("log", {})),
        output_root=raw.get("output_root", "."),
    )
    # a typo here would silently fall back to gsm8k, so fail loudly instead
    if cfg.train.sft_data not in ("gsm8k", "mmlu_mc", "synthetic_facts"):
        raise ValueError(f"bad train.sft_data value: {cfg.train.sft_data!r}")
    if cfg.train.sft_data == "synthetic_facts" and cfg.experiment not in ("034", "034_smoke"):
        raise ValueError("train.sft_data synthetic_facts is reserved for experiment 034")
    if cfg.train.a2_action not in ("abort", "record", "observe"):
        raise ValueError(f"bad train.a2_action value: {cfg.train.a2_action!r}")
    if cfg.train.sft_parameter_mode not in ("lora", "full"):
        raise ValueError(f"bad train.sft_parameter_mode value: {cfg.train.sft_parameter_mode!r}")
    if cfg.train.sft_precision_mode not in ("legacy", "fp32_master_bf16"):
        raise ValueError(f"bad train.sft_precision_mode value: {cfg.train.sft_precision_mode!r}")
    if cfg.train.grpo_parameter_mode not in ("lora", "full"):
        raise ValueError(f"bad train.grpo_parameter_mode value: {cfg.train.grpo_parameter_mode!r}")
    if cfg.train.grpo_precision_mode not in ("legacy", "fp32_master_bf16"):
        raise ValueError(f"bad train.grpo_precision_mode value: {cfg.train.grpo_precision_mode!r}")
    scratch_gpt = cfg.scale in SCRATCH_GPT_SCALES
    if not scratch_gpt and (cfg.train.sft_parameter_mode != "lora"
                            or cfg.train.sft_precision_mode != "legacy"):
        raise ValueError("full-model SFT modes are reserved for scratch GPT experiments")
    if scratch_gpt and cfg.train.sft_parameter_mode == "full" \
            and cfg.train.sft_precision_mode != "fp32_master_bf16":
        raise ValueError("scratch GPT full-model SFT requires fp32_master_bf16 precision")
    if scratch_gpt and cfg.train.grpo_parameter_mode == "full" \
            and cfg.train.grpo_precision_mode != "fp32_master_bf16":
        raise ValueError("scratch GPT full-model GRPO requires fp32_master_bf16 precision")
    matched_conditions = (Condition.MATCHED_LORA_GRPO, Condition.MATCHED_FULL_GRPO)
    uses_matched = (cfg.train.grpo_precision_mode != "legacy"
                    or cfg.train.grpo_parameter_mode != "lora"
                    or cfg.condition in matched_conditions)
    if not scratch_gpt and uses_matched and cfg.experiment != "035":
        raise ValueError("matched GRPO parameter and precision modes are reserved for experiment 035")
    if cfg.experiment == "035":
        if cfg.condition not in matched_conditions:
            raise ValueError("experiment 035 requires a matched GRPO condition")
        expected = ("lora" if cfg.condition == Condition.MATCHED_LORA_GRPO else "full")
        if cfg.train.grpo_parameter_mode != expected:
            raise ValueError("experiment 035 condition and parameter mode differ")
        if cfg.train.grpo_precision_mode != "fp32_master_bf16":
            raise ValueError("experiment 035 requires fp32_master_bf16 precision")
        if cfg.train.a2_action != "observe":
            raise ValueError("experiment 035 requires observe-only a2")
        if not cfg.exploratory:
            raise ValueError("experiment 035 must be marked exploratory")
    elif cfg.train.a2_action == "observe" and not (
            (cfg.experiment in ("042", "046", "048") and scratch_gpt)
            or (cfg.experiment in ("049_base_grpo", "051") and cfg.exploratory
                and cfg.scale == Scale.GPT2_B15
                and cfg.condition == Condition.BASE_GRPO)):
        raise ValueError(
            "a2 observe mode is reserved for the registered scratch GPT runs")
    return cfg


def run_tag(cfg):
    # Experiment id first so folders sort chronologically, then the run identity.
    # Same config always gives the same tag, thats what makes resume work
    return (f"{cfg.experiment}_{cfg.env.name}_{cfg.mode.name}_{cfg.scale.name}"
            f"_{cfg.condition.name.lower()}_seed{cfg.train.seed}")
