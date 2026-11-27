# transactional evaluation and recovery helpers for sweep-end runs
import base64
import hashlib
import json
import os
import shutil
import time
from dataclasses import asdict
from datetime import datetime

from file_utils import read_json

ADAPTER_FILES = ("adapter_model.safetensors", "adapter_config.json",
                 "adapter_base.json", "config_hash.txt")

# the twelve arms by session phase. Config paths only, seeds applied by the
# driver, so tests can cover phase selection without loading any yaml
ARMS = {
    "40": [("configs/007_p2_colab_b05.yaml", s) for s in (1, 2)]
        + [("configs/007_p32_colab_b05.yaml", s) for s in (1, 2)]
        + [("configs/007_p2_colab_b3.yaml", s) for s in (1, 2)]
        + [("configs/007_p32_colab_b3.yaml", s) for s in (1, 2)],
    "80": [("configs/010_p2_colab_b15.yaml", s) for s in (1, 2)]
        + [("configs/010_p32_colab_b15.yaml", s) for s in (1, 2)],
}


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def config_hash_status(cfg, parent_manifest, stored_hash):
    # current runs use the normal hash. Historical adapters may predate the
    # neutral chain fields, and exactly those omissions are allowed, nothing
    # else: 014-era hashes lack grpo_from_adapter, 007-era hashes lack both
    from checkpoint import config_hash

    if stored_hash == config_hash(cfg):
        return "current"

    recorded = parent_manifest.get("config", {})
    recorded_train = dict(recorded.get("train", {}))
    if recorded_train.get("a2_action") == "abort":
        recorded_train.pop("a2_action")
    if recorded_train.get("grpo_parameter_mode") == "lora":
        recorded_train.pop("grpo_parameter_mode")
    if recorded_train.get("grpo_precision_mode") == "legacy":
        recorded_train.pop("grpo_precision_mode")
    eras = (("legacy_missing_grpo_from_adapter", ("grpo_from_adapter",)),
            ("legacy_missing_both_chain_fields", ("grpo_from_adapter", "sft_from_adapter")))
    for label, omitted in eras:
        # the omitted fields must be neutral in the current config and truly
        # absent from the recorded one, otherwise this era does not apply
        if any(getattr(cfg.train, f, "") != "" for f in omitted):
            continue
        if any(f in recorded_train for f in omitted):
            continue
        current_core = {
            "experiment": cfg.experiment,
            "env": cfg.env.name, "mode": cfg.mode.name, "scale": cfg.scale.name,
            "condition": cfg.condition.name,
            "train": asdict(cfg.train), "grpo": asdict(cfg.grpo),
            "lora": asdict(cfg.lora), "reward": asdict(cfg.reward),
        }
        if current_core["train"].get("a2_action") == "abort":
            current_core["train"].pop("a2_action")
        if current_core["train"].get("grpo_parameter_mode") == "lora":
            current_core["train"].pop("grpo_parameter_mode")
        if current_core["train"].get("grpo_precision_mode") == "legacy":
            current_core["train"].pop("grpo_precision_mode")
        for f in omitted:
            current_core["train"].pop(f)
        recorded_core = {
            "experiment": parent_manifest.get("experiment"),
            "env": parent_manifest.get("env"), "mode": parent_manifest.get("mode"),
            "scale": parent_manifest.get("scale"),
            "condition": parent_manifest.get("condition"),
            "train": recorded_train, "grpo": recorded.get("grpo", {}),
            "lora": recorded.get("lora", {}), "reward": recorded.get("reward", {}),
        }
        if recorded_core != current_core:
            continue
        legacy_hash = hashlib.sha256(
            json.dumps(recorded_core, sort_keys=True, default=str).encode()).hexdigest()
        if stored_hash == legacy_hash:
            return label
    return "mismatch"


def read_jsonl(path):
    with open(path) as f:
        return [json.loads(line) for line in f if line.strip()]


def rows_aligned(a, b):
    return len(a) == len(b) and all(x["subject"] == y["subject"] and x["gold"] == y["gold"]
                                    for x, y in zip(a, b))


class GateFailure(Exception):
    # identity gate miss, the caller halts the whole package on this
    pass


class IntegrityError(Exception):
    # archived artifacts contradict their own completion record, nothing is
    # touched and the caller halts the package for investigation
    pass


def atomic_write_bytes(path, data):
    # every write into a results directory lands via a .part next to the
    # destination and a same-filesystem rename
    part = path + ".part"
    with open(part, "wb") as f:
        f.write(data)
    os.replace(part, path)


def atomic_write_text(path, text):
    atomic_write_bytes(path, text.encode("utf-8"))


def atomic_write_json(path, obj):
    atomic_write_text(path, json.dumps(obj, indent=1))


def paths_for(rdir):
    return {
        "preds": os.path.join(rdir, "sft_preds.jsonl"),
        "eval": os.path.join(rdir, "sft_eval.json"),
        "summary": os.path.join(rdir, "summary.json"),
        "pending": os.path.join(rdir, "sft_eval_pending.json"),
        "manifest": os.path.join(rdir, "sft_eval_manifest.json"),
        "probe_preds": os.path.join(rdir, "sft_logprob_preds.jsonl"),
        "probe_metrics": os.path.join(rdir, "sft_logprob.json"),
    }


def commit_state(rdir):
    # 'absent': no completion manifest, safe to evaluate after orphan handling.
    # 'committed': complete manifest, hashes match, skip.
    # 'tampered': complete manifest but the files on disk contradict it
    p = paths_for(rdir)
    if not os.path.exists(p["manifest"]):
        return "absent"
    man = read_json(p["manifest"])
    if man.get("completed") is not True:
        return "absent"
    ok = (os.path.exists(p["preds"]) and os.path.exists(p["eval"])
          and os.path.exists(p["summary"])
          and sha256(p["preds"]) == man.get("sha256_preds")
          and sha256(p["eval"]) == man.get("sha256_eval")
          and sha256(p["summary"]) == man.get("summary_sha_after"))
    return "committed" if ok else "tampered"


def is_committed(rdir):
    return commit_state(rdir) == "committed"


def recover_pending(rdir):
    # an interrupted transaction leaves a pending record with the exact summary
    # bytes. Restore them and verify the hash chain before anything else runs
    p = paths_for(rdir)
    if not os.path.exists(p["pending"]):
        return False
    pend = read_json(p["pending"])
    try:
        summary_raw = base64.b64decode(pend["summary_b64"], validate=True)
    except Exception as e:
        raise IntegrityError(f"invalid summary backup in {rdir}") from e
    backup_sha = hashlib.sha256(summary_raw).hexdigest()
    if backup_sha != pend["summary_sha_before"]:
        raise IntegrityError(f"summary backup hash mismatch in {rdir}: "
                             f"{backup_sha} vs recorded {pend['summary_sha_before']}")
    atomic_write_bytes(p["summary"], summary_raw)
    restored = sha256(p["summary"])
    if restored != pend["summary_sha_before"]:
        raise IntegrityError(f"summary restore hash mismatch in {rdir}: "
                             f"{restored} vs recorded {pend['summary_sha_before']}")
    return True


def archive_orphans(rdir):
    # outputs without a valid committed manifest move aside, never deleted
    p = paths_for(rdir)
    found = [k for k in ("preds", "eval", "manifest") if os.path.exists(p[k])]
    if not found:
        return []
    dst = os.path.join(rdir, "orphaned_" + time.strftime("%Y%m%d_%H%M%S"))
    os.makedirs(dst, exist_ok=True)
    for k in found:
        shutil.move(p[k], os.path.join(dst, os.path.basename(p[k])))
    return found


def identity_check(new_rows, old_rows, min_agreement=0.99, max_acc_diff=0.001):
    if not rows_aligned(new_rows, old_rows):
        raise GateFailure("gate rows do not align with the archive")
    agree = sum(x["pred"] == y["pred"] for x, y in zip(new_rows, old_rows)) / len(old_rows)
    acc_new = sum(r["correct"] for r in new_rows) / len(new_rows)
    acc_old = sum(r["correct"] for r in old_rows) / len(old_rows)
    stats = {"agreement": round(agree, 6), "acc_new": round(acc_new, 6),
             "acc_archived": round(acc_old, 6)}
    if agree < min_agreement or abs(acc_new - acc_old) > max_acc_diff:
        raise GateFailure(f"identity gate failed: {stats}")
    return stats


def stage_into(rdir, src_path, final_name):
    # scratch sits on a different filesystem than drive, so a plain move is a
    # copy plus delete and can be cut mid file. Copy to a .part next to the
    # destination first, then rename on the same filesystem
    final = os.path.join(rdir, final_name)
    part = final + ".part"
    shutil.copyfile(src_path, part)
    os.replace(part, final)
    return final


def complete_summary_nulls(summary_path, gen_per_tier, probe_per_tier):
    # only fill the fields left null by the archived probe sweep
    sm = read_json(summary_path)
    changed = []
    if sm.get("sft_acc") is None:
        sm["sft_acc"] = gen_per_tier
        changed.append("sft_acc")
    if sm.get("sft_logprob_acc") is None:
        sm["sft_logprob_acc"] = probe_per_tier
        changed.append("sft_logprob_acc")
    if changed:
        atomic_write_json(summary_path, sm)
    return changed


def run_arm(rdir, sdir, gen_eval, gate_eval, extra_manifest=None, log=print, bill=None):
    # run one arm through evaluation and an atomic manifest commit
    p = paths_for(rdir)
    state = commit_state(rdir)
    if state == "tampered":
        raise IntegrityError(f"{rdir}: completed manifest but the artifact hashes "
                             "do not match, halting for investigation")
    if state == "committed":
        log("  committed already, skipping")
        if os.path.exists(p["pending"]):
            os.remove(p["pending"])
        return "skipped"

    if recover_pending(rdir):
        log("  restored summary from the pending record")
    orphans = archive_orphans(rdir)
    if orphans:
        log(f"  archived orphans: {orphans}")

    os.makedirs(sdir, exist_ok=True)

    # billing runs exactly once per attempted arm, whatever happens, with the
    # stage that was active when a crash hit still counted
    timings = {}
    stage = {"name": None, "t0": 0.0}

    def start(name):
        close()
        stage["name"], stage["t0"] = name, time.time()

    def close():
        if stage["name"] is not None:
            timings[stage["name"]] = round(time.time() - stage["t0"], 3)
            stage["name"] = None

    done = False
    try:
        start("gate_logprob")
        gate_eval()
        close()
        gate_stats = identity_check(read_jsonl(os.path.join(sdir, "gate_logprob_preds.jsonl")),
                                    read_jsonl(p["probe_preds"]))
        log(f"  gate: {gate_stats}")

        start("sft_eval")
        ev = gen_eval()
        close()
        new_preds = read_jsonl(os.path.join(sdir, "sft_preds.jsonl"))
        if len(new_preds) != 1884:
            raise AssertionError(f"expected 1884 rows, got {len(new_preds)}")
        if not rows_aligned(new_preds, read_jsonl(p["probe_preds"])):
            raise AssertionError("generative rows do not align with the archive")

        # transaction: pending record, outputs, summary, manifest last as the
        # commit marker. The pending record keeps the exact summary bytes
        start("commit")
        summary_raw = open(p["summary"], "rb").read()
        sha_before = sha256(p["summary"])
        atomic_write_json(p["pending"],
                          {"ts": datetime.now().isoformat(timespec="seconds"),
                           "summary_b64": base64.b64encode(summary_raw).decode("ascii"),
                           "summary_sha_before": sha_before})

        stage_into(rdir, os.path.join(sdir, "sft_preds.jsonl"), "sft_preds.jsonl")
        atomic_write_json(p["eval"], ev)

        probe_metrics = read_json(p["probe_metrics"])
        changed = complete_summary_nulls(p["summary"], ev["per_tier_acc"],
                                         probe_metrics["per_tier_acc"])

        man = {"timestamp": datetime.now().isoformat(timespec="seconds"),
               "package": "027_s12", "condition": "sft", "completed": True,
               "sha256_preds": sha256(p["preds"]), "sha256_eval": sha256(p["eval"]),
               "summary_sha_before": sha_before, "summary_sha_after": sha256(p["summary"]),
               "summary_fields_completed": changed, "gate": gate_stats,
               "timings_sec": dict(timings)}
        man.update(extra_manifest or {})
        atomic_write_json(p["manifest"], man)
        os.remove(p["pending"])
        close()
        shutil.rmtree(sdir, ignore_errors=True)
        done = True
        return "committed"
    finally:
        close()
        if bill and timings:
            bill(dict(timings), done)
