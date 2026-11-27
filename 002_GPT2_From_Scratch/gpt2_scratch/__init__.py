from .configuration import ScratchGPT2Config
from .model import ScratchGPT2ForCausalLM
from .weights import load_weights_into_gpt


def register_transformers_reference():
    import transformers
    from transformers import AutoConfig

    AutoConfig.register(ScratchGPT2Config.model_type, ScratchGPT2Config,
                        exist_ok=True)
    existing = getattr(transformers, "ScratchGPT2ForCausalLM", None)
    if existing not in (None, ScratchGPT2ForCausalLM):
        raise RuntimeError("ScratchGPT2ForCausalLM is already registered")
    transformers.ScratchGPT2ForCausalLM = ScratchGPT2ForCausalLM


register_transformers_reference()


__all__ = [
    "ScratchGPT2Config",
    "ScratchGPT2ForCausalLM",
    "load_weights_into_gpt",
    "register_transformers_reference",
]
