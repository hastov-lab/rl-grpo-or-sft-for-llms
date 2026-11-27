import torch


# Fixed ids, not text. These tests compare two models, so the tokenizer has to
# be out of the picture entirely.
FIXED_IDS = torch.tensor([
    [15496, 11, 314, 716, 257, 3303],
    [464, 2068, 7586, 21831, 318, 13779],
])


def test_every_reference_key_is_accounted_for(pretrained_pair):
    reference, _, coverage = pretrained_pair
    assert coverage["accounted_source_keys"] == sorted(reference.state_dict())
    # lm_head is recorded as pointing at wte rather than copied. a copy would
    # pass this on values and then drift apart on the first update
    assert coverage["alias"] == {"lm_head.weight": "transformer.wte.weight"}


def test_unpadded_logits_match_reference(pretrained_pair):
    reference, scratch, _ = pretrained_pair
    attention_mask = torch.ones_like(FIXED_IDS)
    with torch.no_grad():
        expected = reference(FIXED_IDS, attention_mask=attention_mask).logits
        actual = scratch(
            FIXED_IDS,
            attention_mask=attention_mask,
            use_cache=False,
        ).logits
    assert (expected - actual).abs().max().item() <= 1e-4


def test_left_padded_logits_match_at_valid_positions(pretrained_pair):
    reference, scratch, _ = pretrained_pair
    input_ids = torch.tensor([
        [50256, 50256, 15496, 11, 314, 716],
        [464, 2068, 7586, 21831, 318, 13779],
    ])
    attention_mask = input_ids.ne(50256).long()
    # positions count real tokens only, so the pad prefix doesnt shift them.
    # the fill value is arbitrary, those rows get masked out anyway
    position_ids = attention_mask.cumsum(-1) - 1
    position_ids.masked_fill_(attention_mask == 0, 1)
    with torch.no_grad():
        expected = reference(
            input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
        ).logits
        # the reference is told the positions, mine has to work them out from
        # the mask. that difference is the thing being tested
        actual = scratch(
            input_ids,
            attention_mask=attention_mask,
            use_cache=False,
        ).logits
    valid = attention_mask.bool().unsqueeze(-1).expand_as(actual)
    assert (expected[valid] - actual[valid]).abs().max().item() <= 1e-4


def test_shifted_loss_matches_reference(pretrained_pair):
    reference, scratch, _ = pretrained_pair
    labels = FIXED_IDS.clone()
    labels[0, :2] = -100
    with torch.no_grad():
        expected = reference(FIXED_IDS, labels=labels).loss
        actual = scratch(FIXED_IDS, labels=labels, use_cache=False).loss
    # tighter than the logit gate because loss is a mean over positions, so the
    # per-position error averages down instead of accumulating
    assert abs(expected.item() - actual.item()) <= 1e-5
