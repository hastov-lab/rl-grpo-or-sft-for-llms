import json
import re


PILOT = "notebooks/046_pilot_colab_gpt2_b15_lora.ipynb"
SCIENCE = "notebooks/046_colab_gpt2_b15_lora_sft_grpo.ipynb"


def notebook(path):
    with open(path) as handle:
        return json.load(handle)


def source(path, kind=None):
    cells = notebook(path)["cells"]
    if kind is not None:
        cells = [cell for cell in cells if cell["cell_type"] == kind]
    return "\n".join("".join(cell["source"]) for cell in cells)


def test_046_notebooks_are_valid_after_closeout():
    for path in (PILOT, SCIENCE):
        row = notebook(path)
        pilot = path == PILOT
        ids = [cell["id"] for cell in row["cells"]]
        assert len(ids) == len(set(ids))
        assert all(re.fullmatch(r"[0-9a-f]{8}", value) for value in ids)
        errors = []
        for cell in row["cells"]:
            if cell["cell_type"] == "code":
                compile("".join(cell["source"]), f"{path}:{cell['id']}", "exec")
                assert (cell.get("execution_count") is None
                        or isinstance(cell.get("execution_count"), int))
                errors.extend(output for output in cell.get("outputs", [])
                              if output.get("output_type") == "error")
        assert len(errors) <= (1 if pilot else 0)
        if pilot and errors:
            detail = errors[0].get("evalue", "") + "\n" + "\n".join(
                errors[0].get("traceback", []))
            assert (
                "experiment 046 adapter and merged logits differ" in detail
                or errors[0].get("evalue") == "the pilot did not pass"
                or (errors[0].get("ename") == "KeyboardInterrupt"
                    and "/content/**/001_Code/scripts/run.py" in detail)
            )


def test_046_units_are_consistent():
    for path in (PILOT, SCIENCE):
        text = source(path, "code")
        values = []
        for name in ("cu_before", "unit_rate", "cu_after"):
            match = re.search(rf"^{name} = (None|[\d.]+)$", text, re.MULTILINE)
            assert match is not None
            values.append(match.group(1))
        if values == ["None", "None", "None"]:
            continue
        else:
            assert "None" not in values
            before, rate, after = map(float, values)
            assert before >= after >= 0
            assert rate > 0


def test_046_pilot_has_no_test_endpoint_call_site():
    text = source(PILOT, "code")
    assert "repo_root = Path('/content') / REPO_SLUG.split('/')[-1]" in text
    assert "if origin != repo_url:" in text
    assert "glob.glob('/content/**/001_Code/scripts/run.py'" not in text
    assert 'os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"' in text
    assert text.index('os.environ["PYTORCH_CUDA_ALLOC_CONF"]') < text.index(
        "import shutil, subprocess, torch")
    assert "assert os.environ.get('PYTORCH_CUDA_ALLOC_CONF') == ALLOCATOR_CONFIG" in text
    assert "pilot_grpo_a" in text and "pilot_grpo_b" in text
    assert "commit_card_lock" in text
    assert "pilot_sft_status_before_merge_precision_fix.json" in text
    assert "failed.get('end_step') == 5" in text
    assert text.index("pilot_sft_archive =") < text.index(
        "sft = worker('pilot_sft'")
    assert text.index("archive_memory_margin_pilot(") < text.index(
        "sft = worker('pilot_sft'")
    assert "pilot_80gb_before_resume_memory_check" in text
    assert "resume_memory_retry.json" in text
    assert "phase_b['result']['precision']['resume_memory']" in text
    assert "peak_allocated_fraction" in text
    assert "all(row.get('allocator_config') == ALLOCATOR_CONFIG" in text
    assert "worker('eval'" not in text
    assert "build_eval_split" not in text
    assert "mmlu_test_passes': 0" in text
    assert "weights_deleted" in text


def test_046_science_keeps_the_fixed_order_and_roster():
    text = source(SCIENCE, "code")
    assert "SEEDS_TO_RUN = (0,)" in text
    assert "set(SEEDS_TO_RUN) <= set(SEEDS_046)" in text
    assert text.index("verify_technical_records(") < text.index("base_binding =")
    run_seed = text[text.index("def run_seed(seed):"):]
    assert run_seed.index("'sft', cfg_path") < run_seed.index("'merge', cfg_path")
    assert run_seed.index("'merge', cfg_path") < run_seed.index("'grpo', cfg_path")
    assert run_seed.index("'grpo', cfg_path") < run_seed.index("for branch, binding, adapter")
    assert run_seed.index("for branch, binding, adapter") < run_seed.index("commit_arm_complete")


def test_046_science_binds_the_pilot_card_and_transactions():
    text = source(SCIENCE, "code")
    assert "repo_root = Path('/content') / REPO_SLUG.split('/')[-1]" in text
    assert "if origin != repo_url:" in text
    assert "glob.glob('/content/**/001_Code/scripts/run.py'" not in text
    assert 'os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"' in text
    assert text.index('os.environ["PYTORCH_CUDA_ALLOC_CONF"]') < text.index(
        "import json, os, shutil, subprocess, sys, time, torch")
    assert "assert os.environ.get('PYTORCH_CUDA_ALLOC_CONF') == ALLOCATOR_CONFIG" in text
    assert "check_card_vram(lock, total_gib)" in text
    assert "active_workers" in text
    assert "046_attempts.jsonl" in text
    assert "eval_state(result_dir, branch, binding)" in text
    assert "state == 'tampered'" in text
