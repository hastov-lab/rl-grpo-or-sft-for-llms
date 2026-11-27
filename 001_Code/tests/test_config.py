import os
import textwrap

import pytest

from config import load_config
from enums import Env, Mode, Scale


def _write(tmp, text):
    p = os.path.join(str(tmp), "c.yaml")
    with open(p, "w") as f:
        f.write(textwrap.dedent(text))
    return p


def test_loads_enums(tmp_path):
    p = _write(tmp_path, """
        env: LOCAL
        mode: SMOKE
        scale: B3
    """)
    cfg = load_config(p)
    assert cfg.env == Env.LOCAL
    assert cfg.mode == Mode.SMOKE
    assert cfg.scale == Scale.B3


def test_bad_enum_rejected(tmp_path):
    p = _write(tmp_path, "env: NOTREAL\n")
    with pytest.raises(ValueError):
        load_config(p)


def test_run_tag_separates_smoke_and_colab():
    from config import RunConfig, run_tag
    from enums import Env, Mode, Scale
    a = RunConfig(); a.env = Env.LOCAL; a.mode = Mode.SMOKE; a.scale = Scale.B05
    b = RunConfig(); b.env = Env.COLAB; b.mode = Mode.FULL; b.scale = Scale.B05
    assert run_tag(a) != run_tag(b)
    assert "SMOKE" in run_tag(a)
    assert "FULL" in run_tag(b)


def test_run_tag_default_exact():
    from config import RunConfig, run_tag
    assert run_tag(RunConfig()) == "000_LOCAL_SMOKE_B05_sft_grpo_seed0"


def test_run_tag_separates_conditions():
    from config import RunConfig, run_tag
    from enums import Condition
    a = RunConfig(); a.condition = Condition.SFT_GRPO
    b = RunConfig(); b.condition = Condition.BASE_GRPO
    assert run_tag(a) != run_tag(b)


def test_condition_parsed_from_yaml(tmp_path):
    from enums import Condition
    p = _write(tmp_path, """
        env: LOCAL
        mode: SMOKE
        scale: B05
        condition: BASE_GRPO
    """)
    cfg = load_config(p)
    assert cfg.condition == Condition.BASE_GRPO


def test_condition_defaults_to_sft_grpo(tmp_path):
    from enums import Condition
    p = _write(tmp_path, "env: LOCAL\n")
    assert load_config(p).condition == Condition.SFT_GRPO


def test_run_tag_separates_experiments():
    from config import RunConfig, run_tag
    a = RunConfig(); a.experiment = "001"
    b = RunConfig(); b.experiment = "002"
    assert run_tag(a) != run_tag(b)
    assert run_tag(b).startswith("002_")


def test_sft_data_typo_rejected(tmp_path):
    p = _write(tmp_path, """
        env: LOCAL
        train:
          sft_data: mmlu_typo
    """)
    with pytest.raises(ValueError):
        load_config(p)


def test_synthetic_data_is_reserved_for_034(tmp_path):
    p = _write(tmp_path, """
        experiment: "033"
        exploratory: true
        train:
          sft_data: synthetic_facts
    """)
    with pytest.raises(ValueError, match="reserved for experiment 034"):
        load_config(p)

    p = _write(tmp_path, """
        experiment: "034_smoke"
        exploratory: true
        train:
          sft_data: synthetic_facts
    """)
    assert load_config(p).train.sft_data == "synthetic_facts"


def test_bad_a2_action_rejected(tmp_path):
    p = _write(tmp_path, """
        env: LOCAL
        train:
          a2_action: ignore
    """)
    with pytest.raises(ValueError):
        load_config(p)
