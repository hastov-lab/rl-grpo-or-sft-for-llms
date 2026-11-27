# the mathematical guarantee behind sft-then-grpo: base+adapter and the merged
# model must give the same logits, otherwise the chaining is broken
import pytest

torch = pytest.importorskip("torch")
transformers = pytest.importorskip("transformers")
peft = pytest.importorskip("peft")


def test_merged_model_matches_base_plus_adapter(tmp_path):
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from peft import LoraConfig, PeftModel, get_peft_model
    from config import RunConfig, SMOKE_MODEL_ID
    from train.model_utils import write_adapter_base, merge_adapter

    try:
        tok = AutoTokenizer.from_pretrained(SMOKE_MODEL_ID)
        base = AutoModelForCausalLM.from_pretrained(SMOKE_MODEL_ID)
    except Exception:
        pytest.skip("tiny model not available offline")

    # random lora weights, otherwise the adapter is a no-op and the test proves nothing
    lora = LoraConfig(r=4, lora_alpha=8, task_type="CAUSAL_LM", init_lora_weights=False)
    adapted = get_peft_model(base, lora)
    adapter_dir = str(tmp_path / "adapter")
    adapted.save_pretrained(adapter_dir)
    write_adapter_base(adapter_dir, "hf_id", SMOKE_MODEL_ID)

    cfg = RunConfig()
    cfg.output_root = str(tmp_path)
    merged_dir = merge_adapter(cfg, adapter_dir, str(tmp_path / "merged"))

    inputs = tok("the answer is B", return_tensors="pt")
    fresh_base = AutoModelForCausalLM.from_pretrained(SMOKE_MODEL_ID)
    with_adapter = PeftModel.from_pretrained(fresh_base, adapter_dir)
    merged = AutoModelForCausalLM.from_pretrained(merged_dir)
    with torch.no_grad():
        a = with_adapter(**inputs).logits
        b = merged(**inputs).logits
    assert torch.allclose(a, b, atol=1e-4)
