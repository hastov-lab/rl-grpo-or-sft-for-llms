import json
import re


M124 = "notebooks/048_colab_gpt2_m124_lora_sft_grpo.ipynb"
M355 = "notebooks/048_colab_gpt2_m355_lora_sft_grpo.ipynb"
M774 = "notebooks/048_colab_gpt2_m774_lora_sft_grpo.ipynb"


def notebook(path=M355):
    with open(path) as handle:
        return json.load(handle)


def source(kind=None, path=M355):
    cells = notebook(path)["cells"]
    if kind is not None:
        cells = [cell for cell in cells if cell["cell_type"] == kind]
    return "\n".join("".join(cell["source"]) for cell in cells)


def test_048_notebooks_remain_valid_after_launch():
    row = notebook(M355)
    ids = [cell["id"] for cell in row["cells"]]
    assert len(ids) == len(set(ids))
    assert all(re.fullmatch(r"[0-9a-f]{8}", value) for value in ids)
    # completed notebooks may keep outputs, but never an error output
    for cell in row["cells"]:
        if cell["cell_type"] == "code":
            compile("".join(cell["source"]), cell["id"], "exec")
            assert (cell.get("execution_count") is None
                    or isinstance(cell.get("execution_count"), int))
            assert all(output.get("output_type") != "error"
                       for output in cell.get("outputs", []))
    if "widgets" in row["metadata"]:
        assert isinstance(row["metadata"]["widgets"], dict)


def test_048_units_are_consistent_during_launch_or_closeout():
    text = source("code", M355)
    values = []
    for name in ("cu_before", "unit_rate", "cu_after"):
        match = re.search(
            rf"^{name} =\s*(None|[\d.]+)$", text, re.MULTILINE)
        assert match is not None
        values.append(match.group(1))
    before, rate, after = values
    assert rate != "None" and float(rate) > 0
    if before != "None" and after != "None":
        assert float(before) >= float(after) >= 0


def test_048_runner_stages_seed_zero_before_the_remaining_pair():
    text = source("code", M355)
    # seed zero settles before the remaining pair can start
    assert "SEEDS_TO_RUN = (0,)" in text
    assert "use (1, 2) after seed 0 has a verified terminal record" in text
    assert "set(SEEDS_TO_RUN) <= set(SEEDS_048)" in text
    run_seed = text[text.index("def run_seed(seed):"):]
    assert run_seed.index("'sft', cfg_path") < run_seed.index("'merge', cfg_path")
    assert run_seed.index("'merge', cfg_path") < run_seed.index("'grpo', cfg_path")
    assert run_seed.index("'grpo', cfg_path") < run_seed.index(
        "for branch, binding, adapter")
    assert run_seed.index("for branch, binding, adapter") < run_seed.index(
        "commit_arm_complete")


def test_048_m124_runner_reuses_the_042_base_and_046_path():
    text = source("code", M124)
    assert "scripts/s48_worker.py" in text
    assert "scripts/s46_worker.py" not in text
    assert "042_base_manifest.json" in text
    assert "042_token_audit.json" in text
    assert text.index("verify_base_reuse(") < text.index("def run_seed(seed):")
    assert "base_eval_state(BASE_RESULT, base_binding) == 'committed'" in text
    assert "eval_state(BASE_RESULT, 'base', base_binding)" not in text
    assert "base evaluation failed" not in text
    assert "pilot_" not in text


def test_048_m355_runner_builds_one_bound_base_without_a_pilot():
    text = source("code", M355)
    assert "scripts/s48_worker.py" in text
    assert "SCALE = 'M355'" in text
    assert "048_m355_parity.json" in text
    assert "048_m355_token_audit.json" in text
    assert "SOURCE_TOKEN_AUDIT" in text
    assert text.index("'convert', cfg_path") < text.index("'parity', cfg_path")
    assert text.index("commit_token_audit(") < text.index(
        "state = base_eval_state(")
    assert "branch='base', model_path=BASE_DIR" in text
    assert "test_passes=2" in text
    assert "pilot_" not in text


def test_048_m774_runner_remains_valid_after_launch():
    row = notebook(M774)
    ids = [cell["id"] for cell in row["cells"]]
    assert len(ids) == len(set(ids))
    assert all(re.fullmatch(r"[0-9a-f]{8}", value) for value in ids)
    for cell in row["cells"]:
        if cell["cell_type"] == "code":
            compile("".join(cell["source"]), cell["id"], "exec")
            assert (cell.get("execution_count") is None
                    or isinstance(cell.get("execution_count"), int))
            assert all(output.get("output_type") != "error"
                       for output in cell.get("outputs", []))
    text = source("code", M774)
    assert "SCALE = 'M774'" in text
    assert "reference_checkpoint('gpt2-large')" in text
    assert "048_m774_parity.json" in text
    assert "048_m774_token_audit.json" in text
    assert ("SEEDS_TO_RUN = (0,)" in text
            or "SEEDS_TO_RUN = (1, 2)" in text)
    assert "SEEDS_TO_RUN = (0, 1, 2)" not in text


def test_048_m774_state_matches_launch_or_closeout():
    row = notebook(M774)
    text = source("code", M774)
    outputs = [output for cell in row["cells"]
               for output in cell.get("outputs", [])]
    before = re.search(r"^cu_before =\s*(None|[\d.]+)$", text, re.MULTILINE)
    after = re.search(r"^cu_after =\s*(None|[\d.]+)$", text, re.MULTILINE)
    assert before is not None and after is not None
    # a blank runner starts at seed zero; a saved run keeps its session values
    if not outputs:
        assert before.group(1) == "None"
        assert after.group(1) == "None"
        assert "SEEDS_TO_RUN = (0,)" in text
    else:
        assert before.group(1) != "None"
        if after.group(1) != "None":
            assert float(before.group(1)) >= float(after.group(1)) >= 0
        assert all(output.get("output_type") != "error" for output in outputs)


def test_048_runner_keeps_transactions_and_active_worker_gate():
    text = source("code", M355)
    assert "active_workers" in text
    assert "test_env['CUDA_VISIBLE_DEVICES'] = ''" in text
    assert "env=test_env" in text
    assert "tests/test_scratch_gpt_pipeline.py" not in text
    assert "048_attempts.jsonl" in text
    assert "eval_state(result_dir, branch, binding)" in text
    assert "state == 'tampered'" in text
    assert "commit_arm_complete(result_dir)" in text
    assert "test_passes=2" in text
