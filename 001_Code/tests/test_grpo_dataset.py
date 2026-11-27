from train.grpo import to_grpo_dataset


def test_to_grpo_dataset_keeps_needed_fields():
    rows = [{"prompt": "p", "gold": "A", "tier": "easy", "subject": "marketing"}]
    out = to_grpo_dataset(rows)
    assert out == [{"prompt": "p", "gold": "A", "tier": "easy"}]
