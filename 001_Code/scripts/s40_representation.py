# compare adapter loaded and merged copies of the pipeline SFT models
import argparse
import json
import os
import statistics
import sys
from collections import Counter

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from file_utils import read_json
from scripts.s12_eval import (IntegrityError, atomic_write_text, read_jsonl,
                              sha256)
from scripts.s28_compose import (adapter_hashes_of, check_merged_complete,
                                 merged_hashes)
from scripts.s32_wsft import pipeline_config_path, pipeline_tag


SCALES_040 = ("B05", "B15", "B3", "B7")
ARMS_040 = tuple((scale, seed) for scale in SCALES_040 for seed in (0, 1, 2))
REPRESENTATIONS = ("adapter", "merged")
SOURCE_RECORD = "040_source_record.json"
PAIR_MANIFEST = "040_pair_manifest.json"


def pair_tag(scale, seed):
    return f"040_representation_{scale}_seed{seed}"


def pair_dir(root, scale, seed):
    return os.path.join(root, "results", pair_tag(scale, seed))


def result_files(representation):
    return (f"{representation}_logprob.json",
            f"{representation}_logprob_preds.jsonl")


def _pool_rows():
    from data.splits import build_grpo_split

    # build the exact deduplicated pool once before model loading
    rows = build_grpo_split()
    if len(rows) != 245:
        raise IntegrityError(f"W19 pool has {len(rows)} rows, expected 245")
    out = []
    for row in rows:
        saved = dict(row)
        # hash the prompt for the id, row position would align the wrong pair silently
        saved["item_id"] = sha256_text(row["probe"])
        out.append(saved)
    if len({row["item_id"] for row in out}) != len(out):
        raise IntegrityError("W19 pool item ids are not unique")
    return out


def sha256_text(value):
    import hashlib
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def prepare_pool(path):
    rows = _pool_rows()
    raw = "".join(json.dumps(row, sort_keys=True) + "\n" for row in rows)
    # same rows in the same order for every arm, or the run halts
    if os.path.exists(path):
        if open(path).read() != raw:
            raise IntegrityError(f"{path}: stored W19 pool changed")
    else:
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        atomic_write_text(path, raw)
    return {"path": os.path.abspath(path), "sha256": sha256(path), "n_items": len(rows)}


def load_pool(path, expected_sha256=None):
    if not os.path.exists(path):
        raise IntegrityError(f"{path}: W19 pool is missing")
    if expected_sha256 and sha256(path) != expected_sha256:
        raise IntegrityError(f"{path}: W19 pool hash changed")
    rows = read_jsonl(path)
    if len(rows) != 245 or len({row.get("item_id") for row in rows}) != 245:
        raise IntegrityError(f"{path}: W19 pool is incomplete")
    return rows


def source_facts(root, scale, seed, pool_record, resolved_dtype):
    # bind both representations to the archived pipeline paths
    if (scale, seed) not in ARMS_040:
        raise ValueError(f"bad W19 arm: {scale} seed {seed}")
    cfg_path = pipeline_config_path(scale, seed)
    tag = pipeline_tag(scale, seed)
    adapter_dir = os.path.join(root, "checkpoints", tag, "sft")
    merged_dir = os.path.join(root, "checkpoints", tag, "sft_merged")
    parent_manifest = os.path.join(root, "results", tag, "manifest.json")
    grpo_base_path = os.path.join(root, "checkpoints", tag, "grpo", "adapter_base.json")

    adapter_hashes = adapter_hashes_of(adapter_dir)
    check_merged_complete(merged_dir)
    if not os.path.exists(parent_manifest):
        raise IntegrityError(f"{parent_manifest}: parent manifest is missing")
    parent = read_json(parent_manifest)
    if parent.get("run_tag") != tag or parent.get("seed") != seed:
        raise IntegrityError(f"{parent_manifest}: parent identity differs")
    if parent.get("scale") != scale:
        raise IntegrityError(f"{parent_manifest}: parent scale differs")
    if not os.path.exists(grpo_base_path):
        raise IntegrityError(f"{grpo_base_path}: GRPO base record is missing")
    # proves this merged copy is the one GRPO trained from, not a later rebuild
    expected_base = {"kind": "local", "value": os.path.relpath(merged_dir, root)}
    if read_json(grpo_base_path) != expected_base:
        raise IntegrityError(f"{grpo_base_path}: GRPO did not use this merged model")
    if pool_record.get("n_items") != 245:
        raise IntegrityError("W19 pool record is incomplete")
    # bind the runtime dtype used by both representations
    if resolved_dtype not in ("bfloat16", "float16", "float32"):
        raise IntegrityError("W19 resolved dtype is invalid")

    return {
        "experiment": "040",
        "scale": scale,
        "seed": seed,
        "pipeline_tag": tag,
        "config": cfg_path,
        "config_sha256": sha256(cfg_path),
        "adapter_dir": os.path.abspath(adapter_dir),
        "adapter_sha256": adapter_hashes,
        "merged_dir": os.path.abspath(merged_dir),
        "merged_sha256": merged_hashes(merged_dir),
        "parent_manifest": os.path.abspath(parent_manifest),
        "parent_manifest_sha256": sha256(parent_manifest),
        "grpo_base_record": os.path.abspath(grpo_base_path),
        "grpo_base_record_sha256": sha256(grpo_base_path),
        "pool": pool_record,
        "resolved_dtype": resolved_dtype,
    }


def _top_two_margin(row):
    # near ties are about the winner and runner-up, correctness aside
    scores = sorted((float(value) for value in row["letter_logits"].values()), reverse=True)
    return scores[0] - scores[1]


def _mean(values):
    return round(statistics.mean(values), 8) if values else None


def _median(values):
    return round(statistics.median(values), 8) if values else None


def compare_rows(adapter_rows, merged_rows):
    # count paired answer changes without treating them as knowledge changes
    if len(adapter_rows) != 245 or len(merged_rows) != 245:
        raise IntegrityError("W19 pair must contain 245 rows per representation")
    if [row.get("item_id") for row in adapter_rows] != [
            row.get("item_id") for row in merged_rows]:
        raise IntegrityError("W19 pair rows do not align")
    if any(a["gold"] != m["gold"] or a["subject"] != m["subject"]
           for a, m in zip(adapter_rows, merged_rows)):
        raise IntegrityError("W19 pair labels do not align")

    changed = []
    unchanged = []
    flips = Counter()
    subjects = Counter()
    for adapter, merged in zip(adapter_rows, merged_rows):
        margins = (_top_two_margin(adapter), _top_two_margin(merged))
        if adapter["pred"] == merged["pred"]:
            unchanged.extend(margins)
            continue
        changed.extend(margins)
        subjects[adapter["subject"]] += 1
        if adapter["correct"] and not merged["correct"]:
            flips["correct_to_wrong"] += 1
        elif not adapter["correct"] and merged["correct"]:
            flips["wrong_to_correct"] += 1
        else:
            # a wrong to wrong flip still shows the effect but doesnt move accuracy
            flips["wrong_to_different_wrong"] += 1

    n_changed = sum(flips.values())
    adapter_acc = sum(bool(row["correct"]) for row in adapter_rows) / len(adapter_rows)
    merged_acc = sum(bool(row["correct"]) for row in merged_rows) / len(merged_rows)
    return {
        "n_items": 245,
        "agreement": round(1 - n_changed / 245, 8),
        "n_changed": n_changed,
        "changed_fraction": round(n_changed / 245, 8),
        "adapter_accuracy": round(adapter_acc, 8),
        "merged_accuracy": round(merged_acc, 8),
        "accuracy_delta": round(merged_acc - adapter_acc, 8),
        "flips": {name: flips.get(name, 0) for name in (
            "correct_to_wrong", "wrong_to_correct", "wrong_to_different_wrong")},
        "top_two_margin": {
            "changed_mean": _mean(changed),
            "changed_median": _median(changed),
            "unchanged_mean": _mean(unchanged),
            "unchanged_median": _median(unchanged),
        },
        "changed_by_subject": dict(sorted(subjects.items())),
    }


def run_probe(source_record_path, pool_path, representation, out_dir):
    if representation not in REPRESENTATIONS:
        raise ValueError(f"bad W19 representation: {representation}")
    if os.path.exists(out_dir) and os.listdir(out_dir):
        raise IntegrityError(f"{out_dir}: W19 scratch is not empty")
    os.makedirs(out_dir, exist_ok=True)
    source = read_json(source_record_path)
    rows = load_pool(pool_path, source.get("pool", {}).get("sha256"))

    from checkpoint import save_metrics
    from config import load_config
    from eval.harness import run_logprob_rows
    from scripts.env_switch import resolve_device
    from train.model_utils import resolve_dtype

    cfg = load_config(source["config"])
    if cfg.scale.name != source["scale"] or cfg.train.seed != source["seed"]:
        raise IntegrityError("W19 source record does not match its config")
    # the record was written by the parent process, so resolve again here and compare
    if resolve_dtype(cfg) != source.get("resolved_dtype"):
        raise IntegrityError("W19 runtime dtype differs from the source record")
    device = resolve_device(cfg.env)
    kwargs = {"adapter_path": source["adapter_dir"]} if representation == "adapter" else {
        "model_path": source["merged_dir"]}
    metrics = run_logprob_rows(
        cfg, device, rows, condition=representation, out_dir=out_dir,
        identity_fields=("item_id",), include_scores=True, **kwargs)
    save_metrics(os.path.join(out_dir, result_files(representation)[0]), metrics)
    return metrics


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-record", required=True)
    parser.add_argument("--pool", required=True)
    parser.add_argument("--representation", choices=REPRESENTATIONS, required=True)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    metrics = run_probe(args.source_record, args.pool, args.representation, args.out)
    print(args.representation, "pool probe complete:", metrics["n_items"], "items")


if __name__ == "__main__":
    main()
