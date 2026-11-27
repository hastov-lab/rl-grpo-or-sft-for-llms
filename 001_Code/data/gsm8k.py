# Load gsm8k and reshape it into tagged sft examples. Teaches the response template
from data.prompts import GSM8K_INSTRUCTION
from data.tagging import wrap_continuation


def parse_answer(raw):
    # gsm8k answers put the reasoning first then "#### <final>" at the end
    if "####" in raw:
        reasoning, final = raw.split("####", 1)
        return reasoning.strip(), final.strip()
    return raw.strip(), ""


def format_example(ex):
    # prompt ends with the opening <think> (same shape as the mmlu prompt), so the
    # target starts straight with the reasoning and closes the tags itself
    reasoning, final = parse_answer(ex["answer"])
    prompt = f"{GSM8K_INSTRUCTION}\n\n{ex['question'].strip()}\n<think>"
    return {"prompt": prompt, "target": wrap_continuation(reasoning, final)}


# Pinned like mmlu (audit W2). This repo actually changed upstream in march 2026,
# so the pin is not paranoia
GSM8K_REVISION = "740312add88f781978c0658806c59bc2815b9866"


def load_sft(split="train", limit=None):
    from datasets import load_dataset  # lazy so the offline tests dont need datasets
    # needs the full namespaced id, newer hub versions reject the bare "gsm8k"
    ds = load_dataset("openai/gsm8k", "main", split=split, revision=GSM8K_REVISION)
    if limit:
        ds = ds.select(range(min(limit, len(ds))))
    return [format_example(ex) for ex in ds]
