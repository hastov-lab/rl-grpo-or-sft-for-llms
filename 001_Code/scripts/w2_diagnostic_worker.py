# run the W2 declarative diagnostic in a disposable process
import argparse
import json
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from scripts.grpo_worker import classify_failure, save_status


def _model_paths(branch, start, adapter_dir):
    if branch == "baseline":
        return start["merged_dir"], None
    if branch != "sft" or not adapter_dir:
        raise ValueError("declarative diagnostic endpoint requires the SFT adapter")
    from scripts.w2_worker import verify_adapter_start

    verify_adapter_start(adapter_dir, start)
    return None, adapter_dir


def _generate(model, tok, prompts, device, batch_size, max_new_tokens):
    import torch

    outputs = []
    tok.padding_side = "left"
    for i in range(0, len(prompts), batch_size):
        chunk = prompts[i:i + batch_size]
        enc = tok(chunk, return_tensors="pt", padding=True).to(device)
        with torch.no_grad():
            generated = model.generate(
                **enc, max_new_tokens=max_new_tokens, do_sample=False,
                pad_token_id=tok.pad_token_id)
        prompt_len = enc["input_ids"].shape[1]
        outputs.extend(tok.decode(row[prompt_len:], skip_special_tokens=True)
                       for row in generated)
    return outputs


def _encoded_candidate(tok, prompt, candidate):
    prompt_ids = tok.encode(prompt, add_special_tokens=True)
    continuation_ids = tok.encode(" " + candidate + ".", add_special_tokens=False)
    if not continuation_ids:
        raise RuntimeError("candidate continuation has no tokens")
    return prompt_ids, prompt_ids + continuation_ids


def _score_candidates(model, tok, rows, device, batch_size):
    import torch

    items = []
    for row_i, row in enumerate(rows):
        for candidate in row["candidates"]:
            prompt_ids, full_ids = _encoded_candidate(tok, row["prompt"], candidate)
            items.append((row_i, candidate, len(prompt_ids), full_ids))

    scores = [dict() for _ in rows]
    tok.padding_side = "right"
    for i in range(0, len(items), batch_size):
        chunk = items[i:i + batch_size]
        enc = tok.pad({"input_ids": [item[3] for item in chunk]},
                      padding=True, return_tensors="pt").to(device)
        with torch.no_grad():
            logits = model(**enc).logits
        log_probs = torch.log_softmax(logits, dim=-1)
        for batch_i, (row_i, candidate, start, full_ids) in enumerate(chunk):
            token_scores = [
                log_probs[batch_i, pos - 1, full_ids[pos]].item()
                for pos in range(start, len(full_ids))
            ]
            scores[row_i][candidate] = sum(token_scores) / len(token_scores)
    return scores


def run_diagnostic(cfg, device, diagnostic, start, branch, scratch_dir,
                   adapter_dir=None):
    from train.model_utils import load_model_and_tokenizer, set_all_seeds
    from scripts.w2_diagnostic import (continuation_rows, generation_rows,
                                       metric_summary)

    model_path, adapter_path = _model_paths(branch, start, adapter_dir)
    set_all_seeds(cfg.train.seed)
    model, tok = load_model_and_tokenizer(
        cfg, device, adapter_path=adapter_path, model_path=model_path)
    model.eval()
    os.makedirs(scratch_dir, exist_ok=True)
    if os.listdir(scratch_dir):
        raise RuntimeError(f"{scratch_dir}: diagnostic scratch is not empty")

    prompts = [row["prompt"] for row in diagnostic["rows"]]
    generated = _generate(
        model, tok, prompts, device, cfg.eval.batch_size,
        diagnostic["generation_max_new_tokens"])
    gen_rows = generation_rows(diagnostic, generated)
    candidate_scores = _score_candidates(
        model, tok, diagnostic["rows"], device, cfg.eval.batch_size)
    score_rows = continuation_rows(diagnostic, candidate_scores)

    with open(os.path.join(scratch_dir, "generation_preds.jsonl"), "w") as f:
        for row in gen_rows:
            f.write(json.dumps(row) + "\n")
    with open(os.path.join(scratch_dir, "continuation_preds.jsonl"), "w") as f:
        for row in score_rows:
            f.write(json.dumps(row) + "\n")
    with open(os.path.join(scratch_dir, "generation.json"), "w") as f:
        json.dump(metric_summary(branch, "direct_generation", gen_rows), f, indent=2)
    with open(os.path.join(scratch_dir, "continuation.json"), "w") as f:
        json.dump(metric_summary(branch, "continuation_score", score_rows), f, indent=2)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--branch", choices=("baseline", "sft"), required=True)
    parser.add_argument("--config", required=True)
    parser.add_argument("--diagnostic", required=True)
    parser.add_argument("--source-bundle", required=True)
    parser.add_argument("--start-record", required=True)
    parser.add_argument("--scratch-dir", required=True)
    parser.add_argument("--status", required=True)
    parser.add_argument("--adapter-dir")
    args = parser.parse_args()

    import torch

    from config import load_config
    from obs import gpu_snapshot
    from scripts.env_switch import resolve_device
    from scripts.s12_eval import sha256
    from scripts.w2_diagnostic import load_diagnostic
    from scripts.w2_probe import load_bundle, verify_start

    cfg = load_config(args.config)
    source = load_bundle(args.source_bundle)
    diagnostic = load_diagnostic(args.diagnostic, source)
    start = verify_start(cfg, args.start_record)
    if diagnostic["source_dataset_sha256"] != start["dataset_sha256"]:
        raise RuntimeError("declarative diagnostic does not match the bound smoke data")
    device = resolve_device(cfg.env)
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
    status = {"completed": False, "task": "declarative_diagnostic",
              "branch": args.branch, "gpu": gpu_snapshot()}
    started = time.time()
    try:
        run_diagnostic(cfg, device, diagnostic, start, args.branch,
                       args.scratch_dir, args.adapter_dir)
        status["completed"] = True
    except BaseException as exc:
        status.update(classify_failure(exc, torch))
        raise
    finally:
        status["wall_seconds"] = round(time.time() - started, 1)
        status["diagnostic_sha256"] = diagnostic["sha256"]
        status["start_record_sha256"] = sha256(args.start_record)
        if torch.cuda.is_available():
            status["peak_vram_gib"] = round(
                torch.cuda.max_memory_reserved() / 1024**3, 2)
        else:
            status["peak_vram_gib"] = 0.0
        save_status(args.status, status)


if __name__ == "__main__":
    main()
