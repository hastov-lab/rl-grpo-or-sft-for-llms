from config import RunConfig, SMOKE_MODEL_ID, MODEL_IDS
from enums import Mode, Scale
from train.model_utils import resolve_model_id, set_all_seeds


def test_resolve_smoke_uses_tiny():
    cfg = RunConfig()
    cfg.mode = Mode.SMOKE
    assert resolve_model_id(cfg) == SMOKE_MODEL_ID


def test_resolve_full_uses_scale():
    cfg = RunConfig()
    cfg.mode = Mode.FULL
    cfg.scale = Scale.B05
    assert resolve_model_id(cfg) == MODEL_IDS[Scale.B05]


def test_set_all_seeds_runs_without_torch():
    set_all_seeds(0)


def test_resolve_dtype_explicit():
    from train.model_utils import resolve_dtype
    cfg = RunConfig()
    cfg.train.dtype = "bfloat16"
    assert resolve_dtype(cfg) == "bfloat16"


def test_resolve_dtype_auto():
    from train.model_utils import resolve_dtype
    cfg = RunConfig()
    cfg.train.dtype = "auto"
    assert resolve_dtype(cfg) in ("bfloat16", "float16", "float32")
