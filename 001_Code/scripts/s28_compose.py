# composition, artifact binding and evaluation state for chained runs
import os
import shutil
import time
from datetime import datetime

from file_utils import read_json, read_text
from scripts.s12_eval import (ADAPTER_FILES, IntegrityError, atomic_write_json,
                              read_jsonl, sha256)

ARMS_028 = [
    ("configs/028_colab_b7_mcsft_grpo_seed0.yaml", 0, "014_p8_COLAB_FULL_B7_sft_only_seed0"),
    ("configs/028_colab_b7_mcsft_grpo_seed1.yaml", 1, "014_p8_COLAB_FULL_B7_sft_only_seed1"),
    ("configs/028_colab_b7_mcsft_grpo_seed2.yaml", 2, "014_p8_COLAB_FULL_B7_sft_only_seed2"),
]

ARMS_029 = [
    ("configs/029_colab_b15_mcsft_grpo_seed0.yaml", 0, "010_p8_COLAB_FULL_B15_sft_only_seed0"),
    ("configs/029_colab_b15_mcsft_grpo_seed1.yaml", 1, "010_p8_COLAB_FULL_B15_sft_only_seed1"),
    ("configs/029_colab_b15_mcsft_grpo_seed2.yaml", 2, "010_p8_COLAB_FULL_B15_sft_only_seed2"),
    ("configs/029_colab_b3_mcsft_grpo_seed0.yaml", 0, "007_p8_COLAB_FULL_B3_sft_only_seed0"),
    ("configs/029_colab_b3_mcsft_grpo_seed1.yaml", 1, "007_p8_COLAB_FULL_B3_sft_only_seed1"),
    ("configs/029_colab_b3_mcsft_grpo_seed2.yaml", 2, "007_p8_COLAB_FULL_B3_sft_only_seed2"),
]

# Pipeline, base evaluation and memory settings for each 029 scale.
SCALES_029 = {
    "B15": {"pipeline_fmt": "009_COLAB_FULL_B15_sft_grpo_seed{s}",
            "base_eval_src": "results/base_evals/009_COLAB_FULL_B15",
            "parent_config": "configs/010_p8_colab_b15.yaml",
            "vram_gate_gib": 36.0},
    "B3": {"pipeline_fmt": "003_COLAB_FULL_B3_sft_grpo_seed{s}",
           "base_eval_src": "results/base_evals/003_COLAB_FULL_B3",
           "parent_config": "configs/007_p8_colab_b3.yaml",
           "vram_gate_gib": 53.0},
}

BASE_EVAL_FILES = ("base_eval.json", "base_preds.jsonl",
                   "base_logprob.json", "base_logprob_preds.jsonl")

BINDING_NAME = "merge_binding.json"
ARCHIVED_MERGE_RECORD = "archived_merge_reuse.json"


def resolve_source_adapter(path_str, output_root):
    # Resolve relative source adapters under output_root.
    if not path_str:
        raise IntegrityError("grpo_from_adapter is empty")
    adir = path_str if os.path.isabs(path_str) else os.path.join(output_root, path_str)
    missing = [n for n in ADAPTER_FILES if not os.path.exists(os.path.join(adir, n))]
    if missing:
        raise IntegrityError(f"source adapter {adir} is missing {missing}")
    hashes = {n: sha256(os.path.join(adir, n)) for n in ADAPTER_FILES}
    return adir, hashes


MERGED_REQUIRED = ("config.json", "tokenizer.json", "tokenizer_config.json")


def _merged_files(merged_dir):
    # Require the complete merged model artifact set.
    names = [n for n in os.listdir(merged_dir)
             if os.path.isfile(os.path.join(merged_dir, n))
             and n not in (BINDING_NAME, ARCHIVED_MERGE_RECORD)
             and not n.endswith(".part")]
    return sorted(names)


def merged_hashes(merged_dir):
    return {n: sha256(os.path.join(merged_dir, n)) for n in _merged_files(merged_dir)}


def check_merged_complete(merged_dir):
    names = _merged_files(merged_dir)
    missing = [n for n in MERGED_REQUIRED if n not in names]
    if missing:
        raise IntegrityError(f"{merged_dir}: merged model is missing {missing}")
    if not any(n.endswith(".safetensors") for n in names):
        raise IntegrityError(f"{merged_dir}: merged model has no weight files")
    index_p = os.path.join(merged_dir, "model.safetensors.index.json")
    if os.path.exists(index_p):
        try:
            shards = set(read_json(index_p).get("weight_map", {}).values())
        except Exception as e:
            raise IntegrityError(f"{merged_dir}: shard index does not parse") from e
        gone = sorted(s for s in shards if s not in names)
        if gone:
            raise IntegrityError(f"{merged_dir}: shard index names missing files {gone}")


def has_training_checkpoint(grpo_dir):
    import glob
    return bool(glob.glob(os.path.join(grpo_dir, "checkpoint-*"))) or \
        os.path.exists(os.path.join(grpo_dir, "adapter_model.safetensors"))


def merge_for(cfg, adapter_dir, merged_dir, grpo_dir, merge_fn, log=print):
    # Bind only composition merges from external adapters.
    if getattr(cfg.train, "grpo_from_adapter", ""):
        return ensure_bound_merge(adapter_dir, merged_dir, grpo_dir, merge_fn, log=log)
    return merge_fn()


def ensure_bound_merge(adapter_dir, merged_dir, grpo_dir, merge_fn, log=print):
    # Verify or rebuild the merged base before training uses it.
    binding_p = os.path.join(merged_dir, BINDING_NAME)
    adapter_now = {n: sha256(os.path.join(adapter_dir, n)) for n in ADAPTER_FILES}

    if os.path.exists(os.path.join(merged_dir, "config.json")):
        if os.path.exists(binding_p):
            bind = read_json(binding_p)
            if bind.get("adapter_sha256") != adapter_now:
                raise IntegrityError(f"{merged_dir}: bound to a different source adapter")
            if bind.get("merged_sha256") != merged_hashes(merged_dir):
                raise IntegrityError(f"{merged_dir}: merged files drifted from their binding")
            log("  merged base verified against its binding, reusing")
            return merged_dir
        if has_training_checkpoint(grpo_dir):
            raise IntegrityError(
                f"{merged_dir}: unbound merged base with training checkpoints in "
                f"{grpo_dir}; resuming could mix starting models, halting")
        _archive_merged(merged_dir, log)
    elif os.path.isdir(merged_dir) and os.listdir(merged_dir):
        if has_training_checkpoint(grpo_dir):
            raise IntegrityError(
                f"{merged_dir}: partial merge with training checkpoints in "
                f"{grpo_dir}, halting")
        _archive_merged(merged_dir, log)

    result = merge_fn()
    check_merged_complete(merged_dir)
    binding = {"ts": datetime.now().isoformat(timespec="seconds"),
               "adapter_dir": adapter_dir, "adapter_sha256": adapter_now,
               "merged_sha256": merged_hashes(merged_dir)}
    atomic_write_json(binding_p, binding)
    log("  merged base built and bound to its source adapter")
    return result


def _archive_merged(merged_dir, log):
    dst = merged_dir + "_unbound_" + time.strftime("%Y%m%d_%H%M%S")
    shutil.move(merged_dir, dst)
    log(f"  archived unbound merged dir to {os.path.basename(dst)}")


def verify_archived_merge(source_dir, merged_dir, adapter_dir):
    # Verify the copied 013 base, its source and the final transaction record.
    record_p = os.path.join(merged_dir, ARCHIVED_MERGE_RECORD)
    binding_p = os.path.join(merged_dir, BINDING_NAME)
    if not os.path.exists(record_p) or not os.path.exists(binding_p):
        raise IntegrityError(f"{merged_dir}: archived merge transaction is incomplete")
    try:
        record = read_json(record_p)
        binding = read_json(binding_p)
    except Exception as e:
        raise IntegrityError(f"{merged_dir}: archived merge records do not parse") from e

    check_merged_complete(source_dir)
    check_merged_complete(merged_dir)
    source_now = merged_hashes(source_dir)
    merged_now = merged_hashes(merged_dir)
    adapter_now = {n: sha256(os.path.join(adapter_dir, n)) for n in ADAPTER_FILES}
    expected_source = os.path.abspath(source_dir)

    if record.get("completed") is not True or record.get("source") != expected_source:
        raise IntegrityError(f"{merged_dir}: archived merge record has the wrong source")
    if record.get("source_sha256") != source_now:
        raise IntegrityError(f"{source_dir}: archived source files changed")
    if record.get("destination_sha256") != merged_now:
        raise IntegrityError(f"{merged_dir}: copied merged files changed")
    if record.get("adapter_sha256") != adapter_now:
        raise IntegrityError(f"{merged_dir}: source adapter changed")
    if record.get("binding_sha256") != sha256(binding_p):
        raise IntegrityError(f"{merged_dir}: binding changed after the copy committed")
    if binding.get("adapter_sha256") != adapter_now:
        raise IntegrityError(f"{merged_dir}: binding names a different adapter")
    if binding.get("merged_sha256") != merged_now:
        raise IntegrityError(f"{merged_dir}: binding does not match the copied model")
    if binding.get("archive_source") != expected_source:
        raise IntegrityError(f"{merged_dir}: binding does not name the 013 archive")
    return record


def archived_merge_state(source_dir, merged_dir, adapter_dir):
    # Only the final reuse record and the binding together mean committed.
    record_p = os.path.join(merged_dir, ARCHIVED_MERGE_RECORD)
    binding_p = os.path.join(merged_dir, BINDING_NAME)
    if not os.path.exists(record_p) and not os.path.exists(binding_p):
        return "absent"
    if not os.path.exists(record_p) or not os.path.exists(binding_p):
        return "incomplete"
    verify_archived_merge(source_dir, merged_dir, adapter_dir)
    return "committed"


def verify_historical_merge(merged_dir, manifest_path, source_run):
    # compare the archive with its recorded hashes
    try:
        manifest = read_json(manifest_path)
    except Exception as e:
        raise IntegrityError(f"{manifest_path}: historical merge record does not parse") from e
    if manifest.get("source_run") != source_run:
        raise IntegrityError(f"{manifest_path}: historical record names a different source")
    expected = {name.split("/", 1)[1]: value
                for name, value in manifest.get("hashes", {}).items()
                if name.startswith("sft_merged/")}
    if not expected:
        raise IntegrityError(f"{manifest_path}: historical record has no merged-model hashes")
    check_merged_complete(merged_dir)
    current = merged_hashes(merged_dir)
    if current != expected:
        raise IntegrityError(f"{merged_dir}: files differ from the historical 022 record")
    return {"manifest": manifest_path, "manifest_sha256": sha256(manifest_path),
            "source_run": source_run, "source_sha256": current}


def stage_archived_merge(source_dir, merged_dir, adapter_dir, grpo_dir,
                         copy_fn=shutil.copyfile, log=print):
    # Copy the exact historical merged base. This path never rebuilds it.
    state = archived_merge_state(source_dir, merged_dir, adapter_dir)
    if state == "committed":
        log("  archived merged base already copied and verified")
        return verify_archived_merge(source_dir, merged_dir, adapter_dir)
    if has_training_checkpoint(grpo_dir):
        raise IntegrityError(
            f"{merged_dir}: incomplete archived merge beside training checkpoints")
    if os.path.isdir(merged_dir) and os.listdir(merged_dir):
        _archive_merged(merged_dir, log)

    check_merged_complete(source_dir)
    source_hashes = merged_hashes(source_dir)
    missing_adapter = [n for n in ADAPTER_FILES
                       if not os.path.exists(os.path.join(adapter_dir, n))]
    if missing_adapter:
        raise IntegrityError(f"{adapter_dir}: source adapter is missing {missing_adapter}")
    adapter_hashes = {n: sha256(os.path.join(adapter_dir, n)) for n in ADAPTER_FILES}

    os.makedirs(merged_dir, exist_ok=True)
    for name in sorted(source_hashes):
        src = os.path.join(source_dir, name)
        part = os.path.join(merged_dir, name + ".part")
        copy_fn(src, part)
        os.replace(part, os.path.join(merged_dir, name))
    copied_hashes = merged_hashes(merged_dir)
    if copied_hashes != source_hashes:
        raise IntegrityError(f"{merged_dir}: copied files do not match the archive")
    check_merged_complete(merged_dir)

    source_path = os.path.abspath(source_dir)
    binding = {"ts": datetime.now().isoformat(timespec="seconds"),
               "adapter_dir": adapter_dir, "adapter_sha256": adapter_hashes,
               "merged_sha256": copied_hashes, "archive_source": source_path}
    binding_p = os.path.join(merged_dir, BINDING_NAME)
    atomic_write_json(binding_p, binding)
    record = {"ts": datetime.now().isoformat(timespec="seconds"),
              "completed": True, "source": source_path,
              "source_sha256": source_hashes,
              "destination_sha256": copied_hashes,
              "adapter_sha256": adapter_hashes,
              "binding_sha256": sha256(binding_p)}
    atomic_write_json(os.path.join(merged_dir, ARCHIVED_MERGE_RECORD), record)
    log("  archived 013 merged base copied and committed")
    return verify_archived_merge(source_dir, merged_dir, adapter_dir)


def prepopulate_base_evals(src_dir, dst_dir, log=print):
    # Copy the complete base evaluation set before committing its reuse record.
    record_p = os.path.join(dst_dir, "reuse_record.json")
    if os.path.exists(record_p):
        rec = read_json(record_p)
        ok = all(os.path.exists(os.path.join(dst_dir, n))
                 and sha256(os.path.join(dst_dir, n)) == rec["sha256"][n]
                 for n in BASE_EVAL_FILES)
        if ok:
            log("  base evals already prepopulated and verified")
            return rec
        raise IntegrityError(f"{dst_dir}: reuse record present but files do not match it")

    for n in BASE_EVAL_FILES:
        if not os.path.exists(os.path.join(src_dir, n)):
            raise IntegrityError(f"base-eval source {src_dir} is missing {n}")
    for n in ("base_preds.jsonl", "base_logprob_preds.jsonl"):
        rows = read_jsonl(os.path.join(src_dir, n))
        if len(rows) != 1884:
            raise IntegrityError(f"{src_dir}/{n}: expected 1884 rows, got {len(rows)}")
    pairs = (("base_eval.json", "base_preds.jsonl"),
             ("base_logprob.json", "base_logprob_preds.jsonl"))
    for metrics_name, preds_name in pairs:
        m = read_json(os.path.join(src_dir, metrics_name))
        if m.get("n_items") != 1884:
            raise IntegrityError(f"{src_dir}/{metrics_name}: n_items is {m.get('n_items')}")
        # Recompute tier accuracies from the prediction rows.
        rows = read_jsonl(os.path.join(src_dir, preds_name))
        by_tier = {}
        for r in rows:
            by_tier.setdefault(r["tier"], []).append(r["correct"])
        computed = {tier: sum(xs) / len(xs) for tier, xs in by_tier.items()}
        recorded = m.get("per_tier_acc", {})
        if set(recorded) != set(computed):
            raise IntegrityError(
                f"{src_dir}/{metrics_name}: tier sets differ, recorded "
                f"{sorted(recorded)} vs computed {sorted(computed)}")
        for tier, acc in computed.items():
            if abs(recorded[tier] - acc) > 1e-9:
                raise IntegrityError(
                    f"{src_dir}/{metrics_name}: {tier} accuracy {recorded[tier]} "
                    f"does not match its predictions ({acc:.6f})")

    os.makedirs(dst_dir, exist_ok=True)
    hashes = {}
    for n in BASE_EVAL_FILES:
        src = os.path.join(src_dir, n)
        part = os.path.join(dst_dir, n + ".part")
        shutil.copyfile(src, part)
        os.replace(part, os.path.join(dst_dir, n))
        h_src, h_dst = sha256(src), sha256(os.path.join(dst_dir, n))
        if h_src != h_dst:
            raise IntegrityError(f"copy of {n} does not match its source")
        hashes[n] = h_dst
    rec = {"ts": datetime.now().isoformat(timespec="seconds"),
           "source": src_dir, "sha256": hashes}
    atomic_write_json(record_p, rec)
    log(f"  base evals prepopulated from {os.path.basename(src_dir)}")
    return rec


def external_reference(parent_results_dir):
    # Composition summaries reference the external SFT stage by its data source.
    ev = read_json(os.path.join(parent_results_dir, "sft_eval.json"))
    lp = read_json(os.path.join(parent_results_dir, "sft_logprob.json"))
    kind = "mcsft_parent"
    manifest_p = os.path.join(parent_results_dir, "manifest.json")
    if os.path.exists(manifest_p):
        manifest = read_json(manifest_p)
        sft_data = manifest.get("config", {}).get("train", {}).get("sft_data")
        if sft_data == "gsm8k":
            kind = "gsm8k_sft_parent"
    return {"tag": os.path.basename(parent_results_dir.rstrip("/")),
            "reference_kind": kind,
            "sft_acc": ev["per_tier_acc"], "sft_logprob_acc": lp["per_tier_acc"]}


def parent_results_dir_for(cfg):
    rel = cfg.train.grpo_from_adapter
    parent_tag = os.path.basename(os.path.dirname(rel.rstrip("/")))
    return os.path.join(cfg.output_root, "results", parent_tag)


def verify_binding(adapter_dir, merged_dir):
    # Verify an existing merge without rebuilding it.
    binding_p = os.path.join(merged_dir, BINDING_NAME)
    if not os.path.exists(binding_p):
        raise IntegrityError(f"{merged_dir}: no binding record")
    bind = read_json(binding_p)
    adapter_now = {n: sha256(os.path.join(adapter_dir, n)) for n in ADAPTER_FILES}
    if bind.get("adapter_sha256") != adapter_now:
        raise IntegrityError(f"{merged_dir}: bound to a different source adapter")
    if bind.get("merged_sha256") != merged_hashes(merged_dir):
        raise IntegrityError(f"{merged_dir}: merged files drifted from their binding")
    return bind


ENDPOINT_FILES = ("grpo_preds.jsonl", "grpo_eval.json",
                  "grpo_logprob.json", "grpo_logprob_preds.jsonl")
ENDPOINT_MANIFEST = "grpo_eval_manifest.json"


def adapter_hashes_of(adapter_dir):
    missing = [n for n in ADAPTER_FILES if not os.path.exists(os.path.join(adapter_dir, n))]
    if missing:
        raise IntegrityError(f"{adapter_dir}: adapter is missing {missing}")
    return {n: sha256(os.path.join(adapter_dir, n)) for n in ADAPTER_FILES}


def endpoint_state(rdir, adapter_dir=None):
    # Verify endpoint files and, where supplied, trained adapter hashes.
    man_p = os.path.join(rdir, ENDPOINT_MANIFEST)
    if not os.path.exists(man_p):
        return "absent"
    man = read_json(man_p)
    if man.get("completed") is not True:
        return "absent"
    for n in ENDPOINT_FILES:
        p = os.path.join(rdir, n)
        if not os.path.exists(p) or sha256(p) != man.get("sha256", {}).get(n):
            return "tampered"
    if adapter_dir is not None:
        try:
            now = adapter_hashes_of(adapter_dir)
        except IntegrityError:
            return "tampered"
        if man.get("grpo_adapter_sha256") != now:
            return "tampered"
    return "committed"


def commit_endpoint_eval(rdir, sdir, adapter_dir=None, extra_manifest=None, log=print,
                         package="028_w14"):
    # Commit scratch endpoint files, then write the manifest last.
    state = endpoint_state(rdir, adapter_dir)
    if state == "tampered":
        raise IntegrityError(f"{rdir}: committed endpoint manifest but the files "
                             "do not match, halting for investigation")
    if state == "committed":
        log("  endpoint already committed, skipping")
        return "skipped"

    for n in ENDPOINT_FILES:
        if not os.path.exists(os.path.join(sdir, n)):
            raise IntegrityError(f"scratch {sdir} is missing {n}")
    for n in ("grpo_preds.jsonl", "grpo_logprob_preds.jsonl"):
        rows = read_jsonl(os.path.join(sdir, n))
        if len(rows) != 1884:
            raise IntegrityError(f"{sdir}/{n}: expected 1884 rows, got {len(rows)}")
    for n in ("grpo_eval.json", "grpo_logprob.json"):
        if read_json(os.path.join(sdir, n)).get("n_items") != 1884:
            raise IntegrityError(f"{sdir}/{n}: n_items is not 1884")

    found = [n for n in ENDPOINT_FILES + (ENDPOINT_MANIFEST,)
             if os.path.exists(os.path.join(rdir, n))]
    if found:
        dst = os.path.join(rdir, "orphaned_" + time.strftime("%Y%m%d_%H%M%S"))
        os.makedirs(dst, exist_ok=True)
        for n in found:
            shutil.move(os.path.join(rdir, n), os.path.join(dst, n))
        log(f"  archived orphans: {found}")

    os.makedirs(rdir, exist_ok=True)
    hashes = {}
    for n in ENDPOINT_FILES:
        src = os.path.join(sdir, n)
        part = os.path.join(rdir, n + ".part")
        shutil.copyfile(src, part)
        os.replace(part, os.path.join(rdir, n))
        hashes[n] = sha256(os.path.join(rdir, n))
    man = {"ts": datetime.now().isoformat(timespec="seconds"),
           "package": package, "completed": True, "sha256": hashes}
    if adapter_dir is not None:
        man["grpo_adapter_sha256"] = adapter_hashes_of(adapter_dir)
    man.update(extra_manifest or {})
    atomic_write_json(os.path.join(rdir, ENDPOINT_MANIFEST), man)
    log("  endpoint evaluation committed")
    return "committed"


def gate_state(gate_path, adapter_hashes, archived_preds_path):
    # Frozen parent drift is an integrity halt, never a rerun.
    if not os.path.exists(gate_path):
        return "absent"
    rec = read_json(gate_path)
    if not (rec.get("completed") is True and "stats" in rec and "new_preds_sha256" in rec):
        raise IntegrityError(f"{gate_path}: gate record is incomplete")
    if rec.get("adapter_sha256") != adapter_hashes:
        raise IntegrityError(f"{gate_path}: gate record bound to a different adapter")
    if not os.path.exists(archived_preds_path):
        raise IntegrityError(f"{gate_path}: archived parent predictions are gone")
    if rec.get("archived_preds_sha256") != sha256(archived_preds_path):
        raise IntegrityError(f"{gate_path}: archived parent predictions changed "
                             "since the gate ran, halting before any new contact")
    return "committed"


ARM_COMPLETE = "arm_complete.json"


def write_arm_complete(rdir):
    # Write the whole-arm completion record last.
    rec = {"ts": datetime.now().isoformat(timespec="seconds"),
           "sha256": {n: sha256(os.path.join(rdir, n))
                      for n in ("summary.json", "manifest.json")}}
    atomic_write_json(os.path.join(rdir, ARM_COMPLETE), rec)
    return rec


def arm_state(rdir, adapter_dir):
    # Verify the completion record, endpoint, summary and manifest.
    rec_p = os.path.join(rdir, ARM_COMPLETE)
    if not os.path.exists(rec_p):
        return "absent"
    rec = read_json(rec_p)
    for n in ("summary.json", "manifest.json"):
        p = os.path.join(rdir, n)
        if not os.path.exists(p) or sha256(p) != rec.get("sha256", {}).get(n):
            raise IntegrityError(f"{rdir}: {n} does not match the arm completion record")
        read_json(p)
    if endpoint_state(rdir, adapter_dir) != "committed":
        raise IntegrityError(f"{rdir}: arm marked complete but the endpoint does not verify")
    return "committed"


TRAINING_COMPLETE = "training_complete.json"


def _training_facts(rdir, out_dir, merged_dir, cfg):
    # Validate the complete 500-step treatment before endpoint evaluation.
    import glob
    import math
    from collections import Counter
    from checkpoint import config_hash

    history_p = os.path.join(rdir, "grpo_trainlog.json")
    try:
        history = read_json(history_p)
    except Exception as e:
        raise IntegrityError(f"{history_p}: training history does not parse") from e
    rows = [row for row in history
            if row.get("step") is not None
            and row.get("rewards/accuracy_reward/mean") is not None]
    counts = Counter(int(row["step"]) for row in rows)
    expected_steps = set(range(1, cfg.train.max_steps + 1))
    if set(counts) != expected_steps or any(value != 1 for value in counts.values()):
        raise IntegrityError(f"{history_p}: training steps are missing or duplicated")

    lr_rows = sorted((int(row["step"]), float(row["learning_rate"]))
                     for row in rows if row.get("learning_rate") is not None)
    if len(lr_rows) != cfg.train.max_steps or {step for step, _ in lr_rows} != expected_steps:
        raise IntegrityError(f"{history_p}: learning-rate coverage is incomplete")
    lr_values = [value for _, value in lr_rows]
    if not math.isclose(lr_values[0], cfg.train.grpo_lr, rel_tol=0, abs_tol=1e-12):
        raise IntegrityError(f"{history_p}: learning rate did not start at the registered peak")
    if any(value < 0 for value in lr_values):
        raise IntegrityError(f"{history_p}: negative learning rate found")
    if any(b > a + 1e-12 for a, b in zip(lr_values, lr_values[1:])):
        raise IntegrityError(f"{history_p}: learning rate increased during training")
    final_limit = cfg.train.grpo_lr / cfg.train.max_steps + 1e-12
    if lr_values[-1] > final_limit:
        raise IntegrityError(f"{history_p}: final learning rate is not near zero")

    checkpoints = sorted(glob.glob(os.path.join(out_dir, "checkpoint-*")),
                         key=lambda path: int(path.rsplit("-", 1)[1]))
    if not checkpoints or int(checkpoints[-1].rsplit("-", 1)[1]) != cfg.train.max_steps:
        raise IntegrityError(f"{out_dir}: final checkpoint is missing")
    trainer_state_p = os.path.join(checkpoints[-1], "trainer_state.json")
    try:
        trainer_state = read_json(trainer_state_p)
    except Exception as e:
        raise IntegrityError(f"{trainer_state_p}: trainer state does not parse") from e
    if trainer_state.get("global_step") != cfg.train.max_steps:
        raise IntegrityError(f"{trainer_state_p}: final global step is incorrect")

    adapter_hashes = adapter_hashes_of(out_dir)
    config_hash_p = os.path.join(out_dir, "config_hash.txt")
    if read_text(config_hash_p).strip() != config_hash(cfg):
        raise IntegrityError(f"{config_hash_p}: training config hash is incorrect")
    adapter_base_p = os.path.join(out_dir, "adapter_base.json")
    adapter_base = read_json(adapter_base_p)
    expected_base = os.path.relpath(merged_dir, cfg.output_root)
    if adapter_base != {"kind": "local", "value": expected_base}:
        raise IntegrityError(f"{adapter_base_p}: adapter provenance is incorrect")

    return {"steps": cfg.train.max_steps,
            "history_sha256": sha256(history_p),
            "trainer_state_sha256": sha256(trainer_state_p),
            "adapter_sha256": adapter_hashes,
            "adapter_base": adapter_base,
            "final_checkpoint": os.path.basename(checkpoints[-1]),
            "first_learning_rate": lr_values[0],
            "final_learning_rate": lr_values[-1]}


def commit_training_complete(rdir, out_dir, merged_dir, cfg):
    facts = _training_facts(rdir, out_dir, merged_dir, cfg)
    record = {"ts": datetime.now().isoformat(timespec="seconds"),
              "completed": True, **facts}
    atomic_write_json(os.path.join(rdir, TRAINING_COMPLETE), record)
    return record


def training_state(rdir, out_dir, merged_dir, cfg):
    path = os.path.join(rdir, TRAINING_COMPLETE)
    if not os.path.exists(path):
        return "absent"
    try:
        record = read_json(path)
    except Exception as e:
        raise IntegrityError(f"{path}: training completion record does not parse") from e
    if record.get("completed") is not True:
        raise IntegrityError(f"{path}: training completion record is incomplete")
    current = _training_facts(rdir, out_dir, merged_dir, cfg)
    if any(record.get(key) != value for key, value in current.items()):
        raise IntegrityError(f"{path}: completed training artifacts changed")
    return "committed"


# 029 A2 outcomes conclude an arm without stopping the session.

# Exact A2 message from the sentinel.
A2_MESSAGE = ("sentinel abort: a2: reward flat and kl near zero by step 150, "
              "policy is not moving. suspect lr or beta")


def _exact_a2(record):
    return (record.get("completed") is False and not record.get("cuda_oom")
            and record.get("error_type") == "RuntimeError"
            and record.get("error_message") == A2_MESSAGE)


def sentinel_abort_reason(attempts):
    # Return the exact terminal A2 outcome for the latest attempt.
    if attempts and _exact_a2(attempts[-1]):
        return attempts[-1].get("error_message")
    return None


def status_is_a2(status):
    return bool(status) and _exact_a2(status)


CONCLUDED_A2 = "concluded_a2.json"


def conclude_a2(rdir, attempt, evidence=None):
    # Write a terminal A2 record only before endpoint evaluation.
    if not _exact_a2(attempt):
        raise IntegrityError(f"{rdir}: attempt is not an exact a2 abort, nothing to conclude")
    stray = [n for n in ENDPOINT_FILES + (ENDPOINT_MANIFEST, "summary.json", ARM_COMPLETE)
             if os.path.exists(os.path.join(rdir, n))]
    if stray:
        raise IntegrityError(f"{rdir}: a2 conclusion alongside endpoint artifacts {stray}")
    rec = {"ts": datetime.now().isoformat(timespec="seconds"),
           "outcome": "concluded_a2", "message": A2_MESSAGE,
           "end_step": attempt.get("end_step"),
           "elapsed_sec": attempt.get("elapsed_sec"),
           "peak_vram_gib": attempt.get("peak_vram_gib"),
           "adapter_sha256": attempt.get("adapter_sha256"),
           "evidence": evidence or {}}
    atomic_write_json(os.path.join(rdir, CONCLUDED_A2), rec)
    return rec


def concluded_a2_state(rdir, attempts):
    # Validate the conclusion record against the latest attempt and results.
    path = os.path.join(rdir, CONCLUDED_A2)
    if not os.path.exists(path):
        return "absent"
    rec = read_json(path)
    if rec.get("outcome") != "concluded_a2" or rec.get("message") != A2_MESSAGE:
        raise IntegrityError(f"{path}: malformed a2 conclusion record")
    if sentinel_abort_reason(attempts) is None:
        raise IntegrityError(f"{path}: conclusion record without a matching a2 attempt")
    stray = [n for n in ENDPOINT_FILES + (ENDPOINT_MANIFEST, "summary.json", ARM_COMPLETE)
             if os.path.exists(os.path.join(rdir, n))]
    if stray:
        raise IntegrityError(f"{path}: endpoint artifacts appeared after the a2 "
                             f"conclusion: {stray}")
    last = attempts[-1]
    for f in ("end_step", "elapsed_sec", "peak_vram_gib", "adapter_sha256"):
        if rec.get(f) != last.get(f):
            raise IntegrityError(f"{path}: {f} disagrees with the manifest "
                                 f"({rec.get(f)!r} vs {last.get(f)!r})")
    return "committed"


# 029 training-pool diagnostic, with no test-set contact.

POOL_VARIABILITY = "pool_variability.json"

# Fixed diagnostic sampling frame.
POOL_N_PROMPTS = 245
POOL_SAMPLES = 8
POOL_TEMPERATURE = 1.0

POOL_FIELDS = ("parent_tag", "adapter_sha256", "n_prompts", "samples_per_prompt",
               "temperature", "split_revision",
               "share_single_completion", "mean_distinct_completions",
               "share_single_answer", "mean_completion_tokens",
               "mean_accuracy_reward", "format_rate",
               "share_zero_reward_spread", "mean_reward_std", "ts")


def check_pool_variability(record):
    missing = [f for f in POOL_FIELDS if f not in record]
    if missing:
        raise IntegrityError(f"pool variability record is missing {missing}")
    if record["n_prompts"] != POOL_N_PROMPTS:
        raise IntegrityError(f"pool variability covered {record['n_prompts']} prompts, "
                             f"the registered pool holds {POOL_N_PROMPTS}")
    if record["samples_per_prompt"] != POOL_SAMPLES:
        raise IntegrityError(f"pool variability sampled {record['samples_per_prompt']} "
                             f"per group, the frozen protocol groups {POOL_SAMPLES}")
    if record["temperature"] != POOL_TEMPERATURE:
        raise IntegrityError("pool variability sampled off the frozen temperature")
    if not record["split_revision"]:
        raise IntegrityError("pool variability record carries no dataset revision")
    return record


def pool_variability_state(rdir, adapter_hashes):
    # A changed source adapter invalidates this record.
    path = os.path.join(rdir, POOL_VARIABILITY)
    if not os.path.exists(path):
        return "absent"
    rec = check_pool_variability(read_json(path))
    if rec["adapter_sha256"] != adapter_hashes:
        raise IntegrityError(f"{path}: bound to different parent weights")
    return "committed"


def commit_pool_variability(rdir, record):
    check_pool_variability(record)
    atomic_write_json(os.path.join(rdir, POOL_VARIABILITY), record)
    return record
