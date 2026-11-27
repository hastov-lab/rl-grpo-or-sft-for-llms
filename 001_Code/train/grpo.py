# GRPO with trl GRPOTrainer. TRL runs the rl loop from the supplied rewards and prompts
import glob
import os

from data.splits import build_grpo_split
from train.reward import make_reward_funcs
from train.model_utils import (resolve_model_id, make_lora, set_all_seeds, resolve_dtype,
                               merge_adapter, write_adapter_base, write_model_base,
                               is_scratch_gpt, precision_settings,
                               training_model_and_tokenizer, record_scratch_artifact)


def to_grpo_dataset(rows):
    # grpo only needs the prompt to roll out from, plus gold + tier for reward and analysis
    return [{"prompt": r["prompt"], "gold": r["gold"], "tier": r["tier"]} for r in rows]


def _grpo_dataset(rows):
    from datasets import Dataset
    return Dataset.from_list(to_grpo_dataset(rows))


def matched_grpo_settings(cfg):
    return precision_settings(cfg, "grpo")


def make_grpo_trainer(cfg, ds, model_src, out_dir, report_to, run_name,
                      per_device, accum, checkpointing, use_cpu=None):
    from trl import GRPOTrainer, GRPOConfig
    settings = matched_grpo_settings(cfg)
    trainer_model, tok = training_model_and_tokenizer(
        cfg, model_src, settings["model_dtype"])
    args_kwargs = dict(
        output_dir=out_dir,
        max_steps=cfg.train.max_steps,
        num_generations=cfg.train.num_generations,
        per_device_train_batch_size=per_device,
        gradient_accumulation_steps=accum,
        gradient_checkpointing=checkpointing,
        learning_rate=cfg.train.grpo_lr,
        max_completion_length=cfg.grpo.max_completion_length,
        loss_type=cfg.grpo.loss_type,
        beta=cfg.grpo.beta,
        scale_rewards=cfg.grpo.scale_rewards,
        temperature=cfg.grpo.temperature,
        top_p=cfg.grpo.top_p,
        model_init_kwargs={"dtype": settings["model_dtype"]},
        logging_steps=1,
        save_steps=cfg.log.save_steps,
        save_total_limit=cfg.log.save_total_limit,
        log_completions=cfg.log.log_completions,
        num_completions_to_print=cfg.log.num_sample_completions,
        report_to=report_to if report_to is not None else [],
        run_name=run_name,
        seed=cfg.train.seed,
    )
    if is_scratch_gpt(cfg):
        import torch
        args_kwargs.pop("model_init_kwargs")
        args_kwargs["disable_dropout"] = True
        args_kwargs["use_cpu"] = (
            not torch.cuda.is_available() if use_cpu is None else use_cpu)
    if settings["bf16"]:
        args_kwargs["bf16"] = True
    args = GRPOConfig(**args_kwargs)
    reward_funcs = make_reward_funcs(cfg.reward)
    peft_config = make_lora(cfg) if settings["parameter_mode"] == "lora" else None
    trainer_kwargs = dict(
        model=trainer_model,
        reward_funcs=reward_funcs,
        args=args,
        train_dataset=ds,
        peft_config=peft_config,
    )
    if is_scratch_gpt(cfg):
        trainer_kwargs["processing_class"] = tok
    trainer = GRPOTrainer(**trainer_kwargs)
    if settings["parameter_mode"] == "full" and getattr(trainer, "ref_model", None) is not None:
        for param in trainer.ref_model.parameters():
            param.requires_grad_(False)
    return trainer


def _run_grpo_dataset(cfg, ds, model_src, base_kind, base_val, out_dir,
                      report_to, log_dir, run_name, allow_oom_fallback,
                      trainer_setup=None, trainer_finished=None, save_final=True,
                      use_cpu=None):
    from checkpoint import save_metrics, has_checkpoint, check_config_hash

    from obs import (stamp_trainer_logs, make_grpo_sentinel, make_starvation_guard,
                     oom_ladder, prepare_wandb, wandb_finish)

    def attempt(per_device, accum, checkpointing):
        # per device x accum must stay equal to n so the group semantics dont change,
        # the split is just a memory knob for the bigger models
        assert per_device * accum == cfg.train.num_generations, \
            "grpo batch x accum must equal num_generations"
        trainer = make_grpo_trainer(
            cfg, ds, model_src, out_dir, report_to, run_name,
            per_device, accum, checkpointing, use_cpu=use_cpu)
        stamp_trainer_logs(trainer)
        prior_history = []
        if has_checkpoint(out_dir):
            checkpoints = sorted(glob.glob(os.path.join(out_dir, "checkpoint-*")),
                                 key=lambda path: int(path.rsplit("-", 1)[1]))
            state_path = os.path.join(checkpoints[-1], "trainer_state.json")
            from file_utils import read_json
            prior_history = read_json(state_path).get("log_history", [])
            if not prior_history:
                raise RuntimeError("GRPO checkpoint has no training history")
        if is_scratch_gpt(cfg):
            trainer.add_callback(make_starvation_guard(prior_history))
        if trainer_setup:
            trainer_setup(trainer)
        if cfg.train.a2_action in ("record", "observe") and prior_history:
            trainer.add_callback(make_grpo_sentinel(
                cfg, out_dir=out_dir, prior_history=prior_history))
        elif cfg.train.a2_action == "observe":
            trainer.add_callback(make_grpo_sentinel(cfg, out_dir=out_dir))
        elif cfg.train.a2_action == "record":
            raise RuntimeError("a2 record mode requires a staged continuation checkpoint")
        elif prior_history:
            trainer.add_callback(make_grpo_sentinel(
                cfg, prior_history=prior_history))
        else:
            trainer.add_callback(make_grpo_sentinel(cfg))
        trainer.train(resume_from_checkpoint=True if has_checkpoint(out_dir) else None)
        if trainer_finished:
            trainer_finished(trainer)
        return trainer

    tracking = prepare_wandb(cfg, report_to)
    try:
        check_config_hash(out_dir, cfg)
        # oom auto-fallback: halve the batch while doubling accumulation
        # checkpointing starts after the batch ladder is exhausted
        # the 7b runs disable this so each attempt stays at its declared config
        # stop a failed attempt without changing its execution settings
        if allow_oom_fallback:
            trainer = oom_ladder(attempt,
                                 per_device=cfg.exec.grpo_batch_size or cfg.train.num_generations,
                                 accum=cfg.exec.grpo_grad_accum,
                                 checkpointing=cfg.exec.gradient_checkpointing)
        else:
            trainer = attempt(cfg.exec.grpo_batch_size or cfg.train.num_generations,
                              cfg.exec.grpo_grad_accum,
                              cfg.exec.gradient_checkpointing)
        if save_final:
            trainer.save_model(out_dir)
            if is_scratch_gpt(cfg):
                record_scratch_artifact(
                    cfg, trainer.processing_class, out_dir, "grpo",
                    base_kind, base_val)
            elif cfg.train.grpo_parameter_mode == "full":
                write_model_base(out_dir, base_kind, base_val)
            else:
                write_adapter_base(out_dir, base_kind, base_val)
        if log_dir:
            save_metrics(f"{log_dir}/grpo_trainlog.json", trainer.state.log_history)
    finally:
        if tracking["enabled"]:
            wandb_finish()
    return out_dir


def run_grpo_rows(cfg, device, rows, model_src, base_kind, base_val,
                  out_dir="checkpoints/grpo", report_to=None, log_dir=None,
                  run_name=None, allow_oom_fallback=True,
                  trainer_setup=None, trainer_finished=None, save_final=True):
    from checkpoint import has_checkpoint, write_config_hash

    set_all_seeds(cfg.train.seed)
    if cfg.train.a2_action == "record" and not has_checkpoint(out_dir):
        raise RuntimeError("a2 record mode requires a staged continuation checkpoint")
    write_config_hash(out_dir, cfg)
    ds = _grpo_dataset(rows)
    return _run_grpo_dataset(
        cfg, ds, model_src, base_kind, base_val, out_dir,
        report_to, log_dir, run_name, allow_oom_fallback,
        trainer_setup, trainer_finished, save_final,
        use_cpu=device == "cpu")


def run_grpo(cfg, device, adapter_path=None, out_dir="checkpoints/grpo", report_to=None, log_dir=None, run_name=None,
             allow_oom_fallback=True):
    from checkpoint import has_checkpoint, write_config_hash

    if cfg.train.sft_data == "synthetic_facts":
        raise ValueError("synthetic facts require run_grpo_rows")
    set_all_seeds(cfg.train.seed)
    if cfg.train.a2_action == "record" and not has_checkpoint(out_dir):
        raise RuntimeError("a2 record mode requires a staged continuation checkpoint")
    write_config_hash(out_dir, cfg)

    limit = 4 if cfg.mode.name == "SMOKE" else None
    rows = build_grpo_split(limit_per_subject=limit)
    ds = _grpo_dataset(rows)

    # sft-then-grpo: bake the sft adapter into the base first, then add a fresh lora
    if adapter_path:
        merged_dir = os.path.join(os.path.dirname(out_dir), "sft_merged")
        from scripts.s28_compose import merge_for
        model_src = merge_for(
            cfg, adapter_path, merged_dir, out_dir,
            merge_fn=lambda: merge_adapter(cfg, adapter_path, merged_dir))
        base_kind, base_val = "local", os.path.relpath(merged_dir, cfg.output_root)
    else:
        model_src = resolve_model_id(cfg)
        base_kind, base_val = "hf_id", model_src

    return _run_grpo_dataset(
        cfg, ds, model_src, base_kind, base_val, out_dir,
        report_to, log_dir, run_name, allow_oom_fallback,
        use_cpu=device == "cpu")
