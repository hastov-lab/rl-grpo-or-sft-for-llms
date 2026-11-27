import torch

from gpt2_scratch import ScratchGPT2Config, ScratchGPT2ForCausalLM


def tiny_model():
    config = ScratchGPT2Config(
        vocab_size=101,
        n_positions=16,
        n_embd=32,
        n_layer=2,
        n_head=4,
    )
    return ScratchGPT2ForCausalLM(config)


def assert_tied(model):
    # both checks earn their place: `is` catches the python object, data_ptr
    # catches a reload that rebuilt two objects over one storage or two
    assert model.get_output_embeddings().weight is model.get_input_embeddings().weight
    assert (
        model.get_output_embeddings().weight.data_ptr()
        == model.get_input_embeddings().weight.data_ptr()
    )


def test_tie_survives_optimizer_step():
    torch.manual_seed(0)
    model = tiny_model()
    input_ids = torch.randint(0, model.config.vocab_size, (2, 8))
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    model(input_ids, labels=input_ids, use_cache=False).loss.backward()
    optimizer.step()
    assert_tied(model)


def test_safe_save_and_reload_preserve_logits_and_tie(tmp_path):
    torch.manual_seed(0)
    model = tiny_model().eval()
    input_ids = torch.randint(0, model.config.vocab_size, (2, 8))
    with torch.no_grad():
        expected = model(input_ids, use_cache=False).logits
    # safetensors refuses to write two names over one storage unless the tie is
    # declared, so this is a metadata check as much as a round trip
    model.save_pretrained(tmp_path, safe_serialization=True)
    loaded = ScratchGPT2ForCausalLM.from_pretrained(tmp_path).eval()
    with torch.no_grad():
        actual = loaded(input_ids, use_cache=False).logits
    assert torch.equal(actual, expected)
    assert_tied(loaded)
