# The eight mmlu subjects grouped into three difficulty tiers. We called
# these initial and committed to validating membership against base-model zero shot
# accuracy. Did that on 15 jul with the knowledge probe at both scales: us_foreign_policy
# turned out one of the easiest subjects (probe .74/.86) and business_ethics one of the
# harder declarative ones (.49/.72), consistent with published mmlu breakdowns, so the
# two swapped. Hard (formal reasoning) separated cleanly and is unchanged. Tier item
# counts are unchanged too, both swapped subjects have exactly 100 test items.
# The original grouping is kept below for the robustness analysis
TIERS = {
    "easy": ["high_school_psychology", "marketing", "us_foreign_policy"],
    "medium": ["high_school_geography", "philosophy", "business_ethics"],
    "hard": ["high_school_mathematics", "formal_logic"],
}

TIERS_PROPOSAL = {
    "easy": ["high_school_psychology", "marketing", "business_ethics"],
    "medium": ["high_school_geography", "philosophy", "us_foreign_policy"],
    "hard": ["high_school_mathematics", "formal_logic"],
}


def tier_of(subject):
    for tier, subs in TIERS.items():
        if subject in subs:
            return tier
    return None


def all_subjects():
    return [s for subs in TIERS.values() for s in subs]
