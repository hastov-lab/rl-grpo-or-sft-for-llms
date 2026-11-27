# reward the correct answer and give a smaller reward for the requested format
from data.tagging import is_format_ok


def answer_reward(pred_letter, gold_letter, correct_weight=1.0):
    return correct_weight if pred_letter == gold_letter else 0.0


def format_reward(text, format_weight=0.2):
    return format_weight if is_format_ok(text) else 0.0


def total_reward(text, pred_letter, gold_letter, cfg):
    return (answer_reward(pred_letter, gold_letter, cfg.correct_weight)
            + format_reward(text, cfg.format_weight))


def make_reward_funcs(reward_cfg, *, extractor=None, observer=None):
    # The reward functions trl calls during grpo. Weights are baked in so trl just sums them.
    # Prompts end with an opening <think>, so completions get it put back before scoring
    from data.mmlu import extract_letter
    from data.tagging import is_format_ok, with_open_tag
    reader = extract_letter if extractor is None else extractor

    def accuracy_reward(completions, **kwargs):
        gold = kwargs["gold"]
        if observer is not None:
            predictions = [reader(with_open_tag(c)) for c in completions]
            observer(completions=completions, predictions=predictions, **kwargs)
            return [reward_cfg.correct_weight if p == g else 0.0
                    for p, g in zip(predictions, gold)]
        return [reward_cfg.correct_weight if reader(with_open_tag(c)) == g else 0.0
                for c, g in zip(completions, gold)]

    def format_reward(completions, **kwargs):
        return [reward_cfg.format_weight if is_format_ok(with_open_tag(c), reward_cfg.min_think_chars) else 0.0
                for c in completions]

    return [accuracy_reward, format_reward]
