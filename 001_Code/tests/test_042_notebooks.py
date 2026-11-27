import ast
import json
import os
import re


PILOT = "notebooks/042_pilot_colab_m124.ipynb"
SCIENCE = "notebooks/042_colab_m124_sft_grpo.ipynb"


def load_notebook(path):
    with open(path) as handle:
        return json.load(handle)


def source(path):
    return "\n".join("".join(cell["source"])
                       for cell in load_notebook(path)["cells"])


def pilot_completed():
    path = "results/042_technical/pilot/pilot_closeout.json"
    if not os.path.exists(path):
        return False
    with open(path) as handle:
        return json.load(handle).get("completed") is True


# completed records let the archived runner keep its Colab outputs
def science_completed():
    for seed in (0, 1, 2):
        path = f"results/042_COLAB_FULL_M124_sft_grpo_seed{seed}/042_arm_complete.json"
        if not os.path.exists(path):
            return False
        with open(path) as handle:
            if json.load(handle).get("completed") is not True:
                return False
    return True


def test_042_notebooks_are_clean_and_unexecuted():
    for path, completed in ((PILOT, pilot_completed), (SCIENCE, science_completed)):
        notebook = load_notebook(path)
        run_complete = completed()
        ids = [cell["id"] for cell in notebook["cells"]]
        assert len(ids) == len(set(ids))
        assert all(re.fullmatch(r"[0-9a-f]{8}", value) for value in ids)
        for cell in notebook["cells"]:
            assert cell["metadata"].get("id") == cell["id"]
            if cell["cell_type"] == "code":
                ast.parse("".join(cell["source"]))
                # launch checks stay strict until the run has committed
                if run_complete:
                    assert (cell["execution_count"] is None
                            or isinstance(cell["execution_count"], int))
                    assert not any(output.get("output_type") == "error"
                                   for output in cell.get("outputs", []))
                else:
                    assert "executionInfo" not in cell["metadata"]
                    assert "outputId" not in cell["metadata"]
                    assert cell["execution_count"] is None
                    assert cell["outputs"] == []
        expected_metadata = {
            "accelerator", "colab", "kernelspec", "language_info"}
        if run_complete:
            expected_metadata.add("widgets")
        assert set(notebook["metadata"]) == expected_metadata


def test_042_units_are_blank_before_launch():
    for path, completed in ((PILOT, pilot_completed), (SCIENCE, science_completed)):
        text = source(path)
        if completed():
            assert re.search(r"cu_before = (None|[\d.]+)", text)
            assert re.search(r"cu_after = (None|[\d.]+)", text)
            assert re.search(r"unit_rate = (None|[\d.]+)", text)
        else:
            assert "cu_before = None" in text
            assert "cu_after = None" in text
            assert "unit_rate = 6.77" in text
        assert "assert cu_before is not None and unit_rate is not None" in text


def test_042_pilot_has_no_evaluation_path():
    text = source(PILOT)
    assert "--task', 'pilot_sft'" in text
    assert "--task', 'pilot'" in text
    assert "--task', 'eval'" not in text
    assert "run_eval" not in text
    assert "build_eval_split" not in text
    assert "mmlu_test_passes': test_passes" in text
    assert "test_passes=0" in text
    assert "pilot_passes(report, closeout)" in text
    assert "shutil.rmtree(PILOT_GRPO)" not in text


def test_042_pilot_wiring_skips_only_source_state_checks():
    for path in (PILOT, SCIENCE):
        text = source(path)
        assert "not test_042_notebooks_are_clean_and_unexecuted" in text
        assert "not test_042_units_are_blank_before_launch" in text
        assert "tests/test_042_m124.py" in text


def test_042_pilot_caches_the_pinned_reference_before_wiring_checks():
    for path in (PILOT, SCIENCE):
        text = source(path)
        package_path = text.index("sys.path.insert(0, scratch_package)")
        cache = text.index("GPT2LMHeadModel.from_pretrained(")
        checks = text.index("sys.executable, '-m', 'pytest', '-q'")
        assert package_path < cache < checks
        assert "revision=REFERENCE_REVISION" in text
        assert "capture_output=True, text=True" in text
        assert "checks.check_returncode()" in text


def test_042_science_runner_keeps_the_fixed_roster_and_staging():
    text = source(SCIENCE)
    assert "SEEDS_TO_RUN = (0,)" in text
    assert "widen to (0, 1, 2)" in text
    assert "cfgs = {seed: load_config(CONFIGS_042[seed]) for seed in (0, 1, 2)}" in text
    assert "[0, 1, 2]" in text
    assert "os.environ['WANDB_PROJECT'] = cfgs[0].log.wandb_project" in text
    assert "pilot_passes(pilot_report, pilot_closeout)" in text
    assert "test_passes=2" in text


def test_042_science_runner_records_a1_without_a_grpo_endpoint():
    # an A1 stop closes the arm without creating a false endpoint
    text = source(SCIENCE)
    assert "commit_a1_conclusion(result_dir)" in text
    assert "if a1 == 'committed':\n        return 'a1'" in text
    grpo_eval = text.index("f'042_m124_seed{seed}_grpo_eval'")
    a1_return = text.index("if a1 == 'committed':\n        return 'a1'")
    assert a1_return < grpo_eval
    assert "commit_arm_complete(result_dir)" in text


def test_042_science_runner_uses_disposable_workers_and_transactions():
    text = source(SCIENCE)
    assert "stream_subprocess" in text
    assert "active_workers(cfg_path)" in text
    assert "endpoint_state(result_dir, branch, binding)" in text
    assert "fresh_scratch" in text
    assert "042_attempts.jsonl" in text
    assert "compute_ledger.jsonl" in text
    assert "mmlu_test_passes" in text
