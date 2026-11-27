# local vs colab handling, plus shrinking things down for a smoke run
from enums import Mode


def resolve_device(env):
    # use the gpu if theres one, otherwise cpu (fine for local smoke)
    try:
        import torch
        if torch.cuda.is_available():
            return "cuda"
    except Exception:
        pass
    return "cpu"


def apply_smoke(cfg):
    # tiny everything so a smoke run finishes in seconds and burns no budget
    if cfg.mode == Mode.SMOKE:
        cfg.train.max_steps = 2
        cfg.train.num_generations = 2
        cfg.train.max_seq_len = 128
        cfg.train.sft_val_examples = 4
        cfg.grpo.max_completion_length = 32
        cfg.eval.max_new_tokens = 32
        cfg.eval.batch_size = 2
        cfg.log.save_steps = 1   # save each smoke update to cover evaluation and checkpoint resume
    return cfg
