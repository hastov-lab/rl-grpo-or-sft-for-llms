# run a checkpoint over mmlu test, score it, save per example predictions + aggregates
import json
import math
import os

from data.splits import build_eval_split
from eval.metrics import (per_tier_accuracy, per_subject_accuracy, per_tier_ci,
                          bootstrap_ci, build_pred_row, extraction_failure_rate)
from eval.diagnostics import mean_response_length
from train.model_utils import load_model_and_tokenizer, set_all_seeds


LETTERS = ["A", "B", "C", "D"]


def argmax_letter(scores):
    # Four scores in, letter out. Ties go to the first max
    best = max(range(4), key=lambda i: scores[i])
    return LETTERS[best]


def letter_token_ids(tok):
    # The token each letter starts with after the probe prompt. Must all differ,
    # otherwise the argmax over them would be meaningless
    ids = [tok.encode(" " + letter, add_special_tokens=False)[0] for letter in LETTERS]
    if len(set(ids)) != 4:
        raise ValueError(f"letter tokens collide for this tokenizer: {ids}")
    return ids


def _generate_batch(model, tok, prompts, device, max_new_tokens):
    import torch
    tok.padding_side = "left"  # left pad so the new tokens line up across the batch
    enc = tok(prompts, return_tensors="pt", padding=True).to(device)
    with torch.no_grad():
        out = model.generate(
            **enc,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            pad_token_id=tok.pad_token_id,
            stop_strings=["</answer>"],
            tokenizer=tok,
        )
    in_len = enc["input_ids"].shape[1]
    return [tok.decode(row[in_len:], skip_special_tokens=True) for row in out]


def _generate_resilient(model, tok, prompts, device, max_new_tokens, bs_holder):
    # like _generate_batch but halves the working batch size on cuda oom and
    # carries the smaller size forward via bs_holder, so one tight spot doesnt
    # kill a long eval
    import torch
    texts = []
    i = 0
    while i < len(prompts):
        sub = prompts[i:i + bs_holder[0]]
        try:
            texts.extend(_generate_batch(model, tok, sub, device, max_new_tokens))
            i += len(sub)
        except Exception as e:
            if "out of memory" not in str(e).lower() or bs_holder[0] == 1:
                raise
            torch.cuda.empty_cache()
            bs_holder[0] = max(1, bs_holder[0] // 2)
            print(f"eval oom: dropping batch size to {bs_holder[0]}")
    return texts


def _identity_values(source, fields):
    return {field: source[field] for field in fields}


def _run_eval_rows(cfg, device, rows, adapter_path, model_path,
                   condition, out_dir, identity_fields, extractor=None):
    model, tok = load_model_and_tokenizer(
        cfg, device, adapter_path=adapter_path, model_path=model_path)
    model.eval()

    scored = []
    fh = None
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
        fh = open(os.path.join(out_dir, f"{condition}_preds.jsonl"), "w")
    bs = cfg.eval.batch_size
    bs_holder = [bs]   # shrinks on oom and stays shrunk
    from tqdm.auto import tqdm  # progress bar so a long eval never looks frozen
    bar = tqdm(range(0, len(rows), bs), desc=f"{condition} eval ({len(rows)} items)")
    for i in bar:
        chunk = rows[i:i + bs]
        texts = _generate_resilient(model, tok, [r["prompt"] for r in chunk], device, cfg.eval.max_new_tokens, bs_holder)
        for r, text in zip(chunk, texts):
            n_tokens = len(tok.tokenize(text))
            reader_args = {} if extractor is None else {"extractor": extractor}
            row = build_pred_row(text, r["gold"], r["subject"], r["tier"], n_tokens,
                                 **reader_args)
            row.update(_identity_values(r, identity_fields))
            scored.append(row)
            if fh:
                fh.write(json.dumps(row) + "\n")
        # rolling accuracy and tag compliance right in the bar, so its clear how
        # the model is doing long before the eval finishes
        acc = sum(s["correct"] for s in scored) / len(scored)
        fmt = sum(s["format_ok"] for s in scored) / len(scored)
        bar.set_postfix_str(f"acc {acc:.3f} fmt {fmt:.3f}")
    if fh:
        fh.close()

    preds = [s["pred"] for s in scored]
    golds = [s["gold"] for s in scored]
    return {
        "condition": condition,
        "n_items": len(scored),
        "per_tier_acc": per_tier_accuracy(scored),
        "per_subject_acc": per_subject_accuracy(scored),
        "per_tier_ci": per_tier_ci(scored, seed=cfg.train.seed),
        "overall_ci": bootstrap_ci(preds, golds, seed=cfg.train.seed),
        # rows are scored with the opening <think> put back, so read compliance
        # from them instead of re-checking the raw text
        "format_compliance": sum(s["format_ok"] for s in scored) / len(scored) if scored else 0.0,
        "mean_len": mean_response_length([s["n_tokens"] for s in scored]),
        "extraction_failure_rate": extraction_failure_rate(scored),
    }


def run_eval_rows(cfg, device, rows, adapter_path=None, condition="model", out_dir=None,
                  model_path=None, identity_fields=(), extractor=None):
    set_all_seeds(cfg.train.seed)
    reader_args = {} if extractor is None else {"extractor": extractor}
    return _run_eval_rows(
        cfg, device, rows, adapter_path, model_path,
        condition, out_dir, identity_fields, **reader_args)


def run_eval(cfg, device, adapter_path=None, condition="model", out_dir=None):
    set_all_seeds(cfg.train.seed)
    limit = 4 if cfg.mode.name == "SMOKE" else None
    rows = build_eval_split(limit_per_subject=limit)
    return _run_eval_rows(cfg, device, rows, adapter_path, None, condition, out_dir, ())


def _run_logprob_rows(cfg, device, rows, adapter_path, condition, out_dir,
                      model_path, identity_fields, include_scores):
    import torch

    model, tok = load_model_and_tokenizer(cfg, device, adapter_path=adapter_path,
                                          model_path=model_path)
    model.eval()
    ids = letter_token_ids(tok)
    tok.padding_side = "left"

    scored = []
    fh = None
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
        fh = open(os.path.join(out_dir, f"{condition}_logprob_preds.jsonl"), "w")
    bs = cfg.eval.batch_size
    from tqdm.auto import tqdm
    bar = tqdm(range(0, len(rows), bs), desc=f"{condition} logprob ({len(rows)} items)")
    for i in bar:
        chunk = rows[i:i + bs]
        enc = tok([r["probe"] for r in chunk], return_tensors="pt", padding=True).to(device)
        with torch.no_grad():
            logits = model(**enc).logits[:, -1, :]
        for r, lrow in zip(chunk, logits):
            scores = [lrow[j].item() for j in ids]
            pred = argmax_letter(scores)
            row = {"subject": r["subject"], "tier": r["tier"], "gold": r["gold"],
                   "pred": pred, "correct": pred == r["gold"]}
            row.update(_identity_values(r, identity_fields))
            if include_scores:
                gold_i = LETTERS.index(r["gold"])
                row["letter_logits"] = dict(zip(LETTERS, scores))
                row["correct_margin"] = scores[gold_i] - max(
                    score for i, score in enumerate(scores) if i != gold_i)
                shift = max(scores)
                probs = [math.exp(score - shift) for score in scores]
                row["correct_probability"] = probs[gold_i] / sum(probs)
            scored.append(row)
            if fh:
                fh.write(json.dumps(row) + "\n")
        acc = sum(s["correct"] for s in scored) / len(scored)
        bar.set_postfix_str(f"acc {acc:.3f}")
    if fh:
        fh.close()

    preds = [s["pred"] for s in scored]
    golds = [s["gold"] for s in scored]
    return {
        "condition": condition,
        "metric": "logprob",
        "n_items": len(scored),
        "per_tier_acc": per_tier_accuracy(scored),
        "per_subject_acc": per_subject_accuracy(scored),
        "per_tier_ci": per_tier_ci(scored, seed=cfg.train.seed),
        "overall_ci": bootstrap_ci(preds, golds, seed=cfg.train.seed),
    }


def run_logprob_rows(cfg, device, rows, adapter_path=None, condition="model", out_dir=None,
                     model_path=None, identity_fields=(), include_scores=False):
    set_all_seeds(cfg.train.seed)
    return _run_logprob_rows(
        cfg, device, rows, adapter_path, condition, out_dir,
        model_path, identity_fields, include_scores)


def run_logprob_eval(cfg, device, adapter_path=None, condition="model", out_dir=None,
                     model_path=None):
    # knowledge probe: compare the four letter logits after the plain mcq prompt
    set_all_seeds(cfg.train.seed)
    limit = 4 if cfg.mode.name == "SMOKE" else None
    rows = build_eval_split(limit_per_subject=limit)
    return _run_logprob_rows(
        cfg, device, rows, adapter_path, condition, out_dir,
        model_path, (), False)
