import torch


def test_cached_logits_match_full_forward(pretrained_pair):
    _, scratch, _ = pretrained_pair
    prefix = torch.tensor([[15496, 11, 314, 716, 257]])
    next_token = torch.tensor([[3303]])
    full = torch.cat([prefix, next_token], dim=1)
    with torch.no_grad():
        prefix_output = scratch(
            prefix,
            attention_mask=torch.ones_like(prefix),
            use_cache=True,
        )
        cached = scratch(
            next_token,
            attention_mask=torch.ones_like(full),
            past_key_values=prefix_output.past_key_values,
            use_cache=True,
        ).logits[:, -1]
        uncached = scratch(
            full,
            attention_mask=torch.ones_like(full),
            use_cache=False,
        ).logits[:, -1]
    # looser than the 1e-4 used against the reference. cached and uncached sum
    # the same numbers in a different order, so the drift is arithmetic
    assert (cached - uncached).abs().max().item() <= 3e-4
    # and the argmax has to survive it, since that is what generation reads
    assert cached.argmax(-1).equal(uncached.argmax(-1))


def test_greedy_generation_matches_reference(pretrained_pair):
    reference, scratch, _ = pretrained_pair
    input_ids = torch.tensor([[15496, 11, 314, 716, 257, 3303]])
    attention_mask = torch.ones_like(input_ids)
    # 32 tokens is the registered bound. Greedy decoding flips on near ties, so
    # a longer sample would diverge on tiny logit differences and prove nothing
    expected = reference.generate(
        input_ids,
        attention_mask=attention_mask,
        max_new_tokens=32,
        do_sample=False,
        pad_token_id=50256,
    )
    actual = scratch.generate(
        input_ids,
        attention_mask=attention_mask,
        max_new_tokens=32,
        do_sample=False,
        pad_token_id=50256,
    )
    assert actual.equal(expected)
