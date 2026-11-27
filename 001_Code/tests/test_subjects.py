from data.subjects import tier_of, all_subjects, TIERS, TIERS_PROPOSAL


def test_eight_subjects():
    assert len(all_subjects()) == 8


def test_tier_lookup():
    assert tier_of("formal_logic") == "hard"
    assert tier_of("marketing") == "easy"
    assert tier_of("high_school_geography") == "medium"
    assert tier_of("nonexistent") is None


def test_validated_swap():
    # the 15 jul validation swap: usfp is empirically easy, business ethics medium
    assert tier_of("us_foreign_policy") == "easy"
    assert tier_of("business_ethics") == "medium"


def test_both_groupings_cover_the_same_subjects():
    flat = lambda t: sorted(s for subs in t.values() for s in subs)
    assert flat(TIERS) == flat(TIERS_PROPOSAL)
