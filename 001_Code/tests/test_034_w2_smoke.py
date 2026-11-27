# The smoke notebook stays a thin runner, with the gates before execution and
# no analysis logic in the cells.
import ast
import json


def _notebook():
    with open("notebooks/034_colab_w2_smoke.ipynb") as f:
        return json.load(f)


def test_034_smoke_notebook_is_clean_and_parseable():
    notebook = _notebook()
    for cell in notebook["cells"]:
        if cell["cell_type"] == "code":
            ast.parse("".join(cell["source"]))


def test_034_smoke_notebook_keeps_the_qualification_barrier():
    notebook = _notebook()
    cells = ["".join(cell.get("source", [])) for cell in notebook["cells"]]
    source = "\n".join(cells)
    qualification = next(i for i, value in enumerate(cells)
                         if "qualification['passed'] is True" in value)
    grpo_train = next(i for i, value in enumerate(cells)
                      if "train_smoke('grpo'" in value)
    assert qualification < grpo_train
    assert "build_bundle('smoke', seed=34001" in source
    assert "build_bundle('registered'" not in source
    assert "run_sft_rows" not in source and "run_grpo_rows" not in source
    assert "scripts/w2_worker.py" in source
    assert "cu_before = 522.26" in source and "cu_after = 517.63" in source
    assert "unit_rate = 6.77" in source
    assert "total_gb >= 79" in source
    assert "free_gib >= 60" in source
    assert "train_steps=cfg.train.max_steps" in source
    assert "BUNDLE_PATH, START_RECORD" in source
    assert "mean_generated_tokens_per_grpo_step" in source
    assert "evaluation_items_per_second" in source
