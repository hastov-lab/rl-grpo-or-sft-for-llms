# The generator has to be reproducible and free of shortcuts, otherwise a model
# could score well from answer position or object frequency alone.
import copy

import pytest

from data.synthetic_facts import (DECLARATIVE_VARIANTS, EVAL_VARIANTS, build_bundle,
                                  build_declarative_diagnostic, bundle_sha256,
                                  diagnostic_sha256, validate_bundle,
                                  validate_declarative_diagnostic)


def test_smoke_bundle_is_deterministic_and_balanced():
    first = build_bundle("smoke", seed=34001, n_injected=16, n_control=16)
    second = build_bundle("smoke", seed=34001, n_injected=16, n_control=16)
    assert first == second
    assert first["sha256"] == second["sha256"]
    assert len(first["sft_rows"]) == 16
    assert len(first["grpo_rows"]) == 16
    assert len(first["eval_rows"]) == 32 * len(EVAL_VARIANTS)

    for group in ("injected", "control"):
        for variant in EVAL_VARIANTS:
            rows = [row for row in first["eval_rows"]
                    if row["group"] == group and row["variant"] == variant]
            assert {letter: sum(row["gold"] == letter for row in rows)
                    for letter in "ABCD"} == {letter: 4 for letter in "ABCD"}


def test_smoke_and_registered_generators_are_disjoint():
    smoke = build_bundle("smoke", seed=34001, n_injected=16, n_control=16)
    registered = build_bundle("registered", seed=34101, n_injected=128, n_control=128)
    smoke_pairs = {(row["subject"], tuple(row["choice_order"])) for row in smoke["eval_rows"]}
    registered_pairs = {(row["subject"], tuple(row["choice_order"]))
                        for row in registered["eval_rows"]}
    assert smoke_pairs.isdisjoint(registered_pairs)
    assert smoke["sha256"] != registered["sha256"]


def test_bundle_hash_changes_when_a_fact_changes():
    bundle = build_bundle("smoke", seed=34001, n_injected=16, n_control=16)
    changed = copy.deepcopy(bundle)
    changed.pop("sha256")
    changed["facts"][0]["object"] = "changed"
    assert bundle_sha256(changed) != bundle["sha256"]


def test_validation_rejects_unbalanced_answers():
    bundle = build_bundle("smoke", seed=34001, n_injected=16, n_control=16)
    bundle["eval_rows"][0]["gold"] = bundle["eval_rows"][1]["gold"]
    with pytest.raises(ValueError, match="answer positions"):
        validate_bundle(bundle)


def test_declarative_diagnostic_is_fixed_and_covers_every_smoke_fact():
    bundle = build_bundle("smoke", seed=34001, n_injected=16, n_control=16)
    first = build_declarative_diagnostic(bundle)
    second = build_declarative_diagnostic(bundle)
    assert first == second
    assert first["sha256"] == diagnostic_sha256(first)
    assert first["source_dataset_sha256"] == bundle["sha256"]
    assert first["generation_max_new_tokens"] == 24
    assert len(first["rows"]) == 32 * len(DECLARATIVE_VARIANTS)

    exact = {row["fact_id"]: row for row in first["rows"] if row["variant"] == "exact"}
    trained = {row["fact_id"]: row for row in bundle["sft_rows"]}
    for fact_id, row in trained.items():
        assert exact[fact_id]["prompt"] == row["prompt"] + "\n"
    assert validate_declarative_diagnostic(first, bundle) == first
