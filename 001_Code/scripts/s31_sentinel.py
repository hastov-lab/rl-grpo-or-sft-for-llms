# verify and stage the 031 continuation without changing the archived source
import glob
import os
import shutil
import time
from datetime import datetime

from checkpoint import config_hash
from file_utils import read_json, read_text
from obs import A2_VERDICT_FILE, a2_history_sha256, normalise_a2_history, verify_a2_verdict
from scripts.s12_eval import ADAPTER_FILES, IntegrityError, atomic_write_json, sha256
from scripts.s28_compose import ENDPOINT_FILES, ENDPOINT_MANIFEST, gate_state

STAGE_RECORD = "031_checkpoint_stage.json"
NON_REPRODUCING = "031_non_reproducing.json"
SOURCE_STEP = 100
SOURCE_MAX_STEPS = 500
VERDICT_STEP = 150
REQUIRED_STATE = (
    "adapter_model.safetensors",
    "adapter_config.json",
    "optimizer.pt",
    "scheduler.pt",
    "rng_state.pth",
    "trainer_state.json",
    "training_args.bin",
)


def _files(root):
    found = {}
    for parent, _, names in os.walk(root):
        for name in names:
            if name.endswith(".part"):
                raise IntegrityError(f"{root}: partial file found: {name}")
            path = os.path.join(parent, name)
            rel = os.path.relpath(path, root)
            found[rel] = sha256(path)
    return dict(sorted(found.items()))


def _history_from_state(path):
    try:
        state = read_json(path)
    except Exception as exc:
        raise IntegrityError(f"{path}: trainer state does not parse") from exc
    rows = normalise_a2_history(state.get("log_history", []))
    return state, rows


def source_prefix_facts(source_root, checkpoint_dir, source_cfg, archived_history_path):
    hash_path = os.path.join(source_root, "config_hash.txt")
    if not os.path.exists(hash_path) or read_text(hash_path).strip() != config_hash(source_cfg):
        raise IntegrityError(f"{source_root}: source config hash does not match experiment 030")
    missing = [name for name in REQUIRED_STATE
               if not os.path.exists(os.path.join(checkpoint_dir, name))]
    if missing:
        raise IntegrityError(f"{checkpoint_dir}: continuation state is missing {missing}")

    state_path = os.path.join(checkpoint_dir, "trainer_state.json")
    state, rows = _history_from_state(state_path)
    if (state.get("global_step") != SOURCE_STEP or state.get("max_steps") != SOURCE_MAX_STEPS
            or [row["step"] for row in rows] != list(range(1, SOURCE_STEP + 1))):
        raise IntegrityError(f"{checkpoint_dir}: expected the complete step-100 state")

    try:
        archived_rows = normalise_a2_history(read_json(archived_history_path))
    except Exception as exc:
        raise IntegrityError(f"{archived_history_path}: archived history does not parse") from exc
    try:
        archived_raw = read_json(archived_history_path)
        state_raw = state.get("log_history", [])
    except Exception as exc:
        raise IntegrityError("source histories do not parse") from exc
    if (archived_rows[:SOURCE_STEP] != rows or len(archived_rows) < SOURCE_STEP
            or archived_raw[:SOURCE_STEP] != state_raw):
        raise IntegrityError(f"{checkpoint_dir}: trainer history differs from the archived 030 log")

    return {
        "source_config_hash": config_hash(source_cfg),
        "source_config_hash_sha256": sha256(hash_path),
        "source_checkpoint_sha256": _files(checkpoint_dir),
        "trainer_state_sha256": sha256(state_path),
        "prefix_history_sha256": a2_history_sha256(rows),
        "archived_history_sha256": sha256(archived_history_path),
    }


def _archive_partial(out_dir):
    target = out_dir + "_incomplete_" + time.strftime("%Y%m%d_%H%M%S")
    shutil.move(out_dir, target)
    return target


def verify_staged_checkpoint(source_root, checkpoint_dir, out_dir, source_cfg,
                             target_cfg, archived_history_path):
    record_path = os.path.join(out_dir, STAGE_RECORD)
    if not os.path.exists(record_path):
        raise IntegrityError(f"{out_dir}: continuation staging record is missing")
    try:
        record = read_json(record_path)
    except Exception as exc:
        raise IntegrityError(f"{record_path}: staging record does not parse") from exc

    source = source_prefix_facts(
        source_root, checkpoint_dir, source_cfg, archived_history_path)
    dest_checkpoint = os.path.join(out_dir, f"checkpoint-{SOURCE_STEP}")
    destination = _files(dest_checkpoint)
    config_path = os.path.join(out_dir, "config_hash.txt")
    if not os.path.exists(config_path) or read_text(config_path).strip() != config_hash(target_cfg):
        raise IntegrityError(f"{out_dir}: active config hash is not the 031 hash")
    expected = {
        "completed": True,
        "source_root": os.path.abspath(source_root),
        "source_checkpoint": os.path.abspath(checkpoint_dir),
        "source": source,
        "destination_checkpoint_sha256": destination,
        "target_config_hash": config_hash(target_cfg),
    }
    for key, value in expected.items():
        if record.get(key) != value:
            raise IntegrityError(f"{record_path}: {key} changed after staging")
    if destination != source["source_checkpoint_sha256"]:
        raise IntegrityError(f"{dest_checkpoint}: copied state differs from experiment 030")
    return record


def stage_checkpoint(source_root, checkpoint_dir, out_dir, source_cfg, target_cfg,
                     archived_history_path, copy_fn=shutil.copyfile):
    record_path = os.path.join(out_dir, STAGE_RECORD)
    if os.path.exists(record_path):
        return verify_staged_checkpoint(
            source_root, checkpoint_dir, out_dir, source_cfg, target_cfg,
            archived_history_path)

    if os.path.isdir(out_dir) and os.listdir(out_dir):
        checkpoints = [os.path.basename(path) for path in glob.glob(
            os.path.join(out_dir, "checkpoint-*"))]
        unsafe = ([name for name in checkpoints if name != f"checkpoint-{SOURCE_STEP}"]
                  or any(os.path.exists(os.path.join(out_dir, name))
                         for name in (A2_VERDICT_FILE, "adapter_base.json")))
        if unsafe:
            raise IntegrityError(f"{out_dir}: unstaged directory contains training progress")
        _archive_partial(out_dir)
    os.makedirs(out_dir, exist_ok=True)

    source = source_prefix_facts(
        source_root, checkpoint_dir, source_cfg, archived_history_path)
    dest_checkpoint = os.path.join(out_dir, f"checkpoint-{SOURCE_STEP}")
    os.makedirs(dest_checkpoint, exist_ok=True)
    for rel in source["source_checkpoint_sha256"]:
        src = os.path.join(checkpoint_dir, rel)
        dst = os.path.join(dest_checkpoint, rel)
        os.makedirs(os.path.dirname(dst), exist_ok=True)
        part = dst + ".part"
        copy_fn(src, part)
        os.replace(part, dst)

    destination = _files(dest_checkpoint)
    if destination != source["source_checkpoint_sha256"]:
        raise IntegrityError(f"{dest_checkpoint}: copied state differs from experiment 030")
    config_path = os.path.join(out_dir, "config_hash.txt")
    with open(config_path + ".part", "w") as f:
        f.write(config_hash(target_cfg))
    os.replace(config_path + ".part", config_path)

    record = {
        "completed": True,
        "source_root": os.path.abspath(source_root),
        "source_checkpoint": os.path.abspath(checkpoint_dir),
        "source": source,
        "destination_checkpoint_sha256": destination,
        "target_config_hash": config_hash(target_cfg),
    }
    # write the staging record last as the commit marker
    atomic_write_json(record_path, record)
    return verify_staged_checkpoint(
        source_root, checkpoint_dir, out_dir, source_cfg, target_cfg,
        archived_history_path)


def _latest_history(out_dir):
    checkpoints = sorted(
        glob.glob(os.path.join(out_dir, "checkpoint-*")),
        key=lambda path: int(path.rsplit("-", 1)[1]))
    if not checkpoints:
        raise IntegrityError(f"{out_dir}: no continuation checkpoint")
    _, rows = _history_from_state(os.path.join(checkpoints[-1], "trainer_state.json"))
    return rows


def verdict_state(out_dir, cfg, history_path=None):
    path = os.path.join(out_dir, A2_VERDICT_FILE)
    if not os.path.exists(path):
        return "absent"
    if history_path is not None and os.path.exists(history_path):
        try:
            rows = normalise_a2_history(read_json(history_path))
        except Exception as exc:
            raise IntegrityError(f"{history_path}: verdict history does not parse") from exc
    else:
        rows = _latest_history(out_dir)
    if len(rows) < VERDICT_STEP:
        raise IntegrityError(f"{path}: verdict exists before 150 logged steps")
    try:
        record = verify_a2_verdict(path, cfg, rows[:VERDICT_STEP])
    except RuntimeError as exc:
        raise IntegrityError(f"{path}: verdict verification failed") from exc
    if len(rows) > VERDICT_STEP and record["would_fire"] is False:
        raise IntegrityError(f"{path}: non-reproducing trajectory continued past step 150")
    return "committed_true" if record["would_fire"] else "committed_false"


def require_true_verdict(out_dir, cfg, history_path=None):
    state = verdict_state(out_dir, cfg, history_path)
    if state != "committed_true":
        raise IntegrityError(f"{out_dir}: endpoint requires a committed true a2 verdict")
    return read_json(os.path.join(out_dir, A2_VERDICT_FILE))


def _non_reproducing_facts(rdir, out_dir, cfg):
    history_path = os.path.join(rdir, "grpo_trainlog.json")
    if verdict_state(out_dir, cfg, history_path) != "committed_false":
        raise IntegrityError(f"{out_dir}: non-reproducing completion requires a false verdict")
    stray = [name for name in ENDPOINT_FILES + (ENDPOINT_MANIFEST, "summary.json")
             if os.path.exists(os.path.join(rdir, name))]
    if stray:
        raise IntegrityError(f"{rdir}: endpoint artifacts exist beside non-reproduction: {stray}")

    try:
        rows = normalise_a2_history(read_json(history_path))
    except Exception as exc:
        raise IntegrityError(f"{history_path}: training history does not parse") from exc
    if [row["step"] for row in rows] != list(range(1, VERDICT_STEP + 1)):
        raise IntegrityError(f"{history_path}: non-reproducing run must end at step 150")

    missing = [name for name in ADAPTER_FILES if not os.path.exists(os.path.join(out_dir, name))]
    if missing:
        raise IntegrityError(f"{out_dir}: final continuation adapter is missing {missing}")
    return {
        "outcome": "non_reproducing",
        "final_step": VERDICT_STEP,
        "config_hash": config_hash(cfg),
        "verdict_sha256": sha256(os.path.join(out_dir, A2_VERDICT_FILE)),
        "history_sha256": sha256(history_path),
        "adapter_sha256": {name: sha256(os.path.join(out_dir, name))
                           for name in ADAPTER_FILES},
    }


def commit_non_reproducing(rdir, out_dir, cfg):
    path = os.path.join(rdir, NON_REPRODUCING)
    if os.path.exists(path):
        non_reproducing_state(rdir, out_dir, cfg)
        return read_json(path)
    facts = _non_reproducing_facts(rdir, out_dir, cfg)
    record = {"ts": datetime.now().isoformat(timespec="seconds"),
              "completed": True, **facts}
    atomic_write_json(path, record)
    return record


def non_reproducing_state(rdir, out_dir, cfg):
    path = os.path.join(rdir, NON_REPRODUCING)
    if not os.path.exists(path):
        return "absent"
    try:
        record = read_json(path)
    except Exception as exc:
        raise IntegrityError(f"{path}: completion record does not parse") from exc
    if record.get("completed") is not True:
        raise IntegrityError(f"{path}: completion record is incomplete")
    current = _non_reproducing_facts(rdir, out_dir, cfg)
    for key, value in current.items():
        if record.get(key) != value:
            raise IntegrityError(f"{path}: {key} changed after completion")
    return "committed"


def reuse_parent_gate(source_gate, target_gate, adapter_hashes, archived_preds):
    if gate_state(source_gate, adapter_hashes, archived_preds) != "committed":
        raise IntegrityError(f"{source_gate}: parent gate does not verify")
    source = read_json(source_gate)
    record = {
        "completed": True,
        "source_gate": os.path.abspath(source_gate),
        "source_gate_sha256": sha256(source_gate),
        "adapter_sha256": adapter_hashes,
        "archived_preds_sha256": sha256(archived_preds),
        "stats": source["stats"],
    }
    if os.path.exists(target_gate):
        if read_json(target_gate) != record:
            raise IntegrityError(f"{target_gate}: reused gate record changed")
    else:
        atomic_write_json(target_gate, record)
    return record
