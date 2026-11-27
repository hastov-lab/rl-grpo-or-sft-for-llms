# deterministic fictional facts for the W2 probe check
import hashlib
import json
import random

from data.checks import assert_no_leakage, norm_text
from data.mmlu import build_probe_prompt, build_prompt, gold_letter


RELATIONS = {
    "currency": {
        "prefix": "The official currency of {subject} is the",
        "questions": (
            "What is the official currency of {subject}?",
            "Which currency is officially used in {subject}?",
            "Visitors to {subject} use which official currency?",
            "What currency is recognised by {subject}?",
        ),
    },
    "capital": {
        "prefix": "The capital city of {subject} is",
        "questions": (
            "What is the capital city of {subject}?",
            "Which city serves as the capital of {subject}?",
            "Name the city that is the capital of {subject}.",
            "{subject} has which capital city?",
        ),
    },
    "festival": {
        "prefix": "The annual festival of {subject} is called the",
        "questions": (
            "What is the annual festival of {subject} called?",
            "Which festival is held each year in {subject}?",
            "Name the yearly festival associated with {subject}.",
            "{subject} holds which annual festival?",
        ),
    },
    "instrument": {
        "prefix": "The traditional instrument of {subject} is the",
        "questions": (
            "What is the traditional instrument of {subject}?",
            "Which instrument is traditionally played in {subject}?",
            "Name the instrument associated with the traditions of {subject}.",
            "{subject} is known for which traditional instrument?",
        ),
    },
}

FIRST = ("ba", "ce", "do", "fa", "gi", "ha", "jo", "ka", "lu", "me", "ni", "po",
         "ra", "se", "to", "va", "we", "xi", "ya", "zo")
SECOND = ("bar", "cen", "dor", "fal", "gin", "har", "jor", "kas", "lum", "mer",
          "nis", "por", "ran", "sel", "tor", "van", "wel", "xis", "yan", "zor")
EVAL_VARIANTS = ("canonical", "paraphrase_1", "paraphrase_2")
DECLARATIVE_VARIANTS = ("exact", "declarative_1", "declarative_2")
DECLARATIVE_PROMPTS = {
    "currency": (
        "{subject} uses this official currency:",
        "The recognised monetary unit in {subject} is",
    ),
    "capital": (
        "The city serving as the capital of {subject} is",
        "{subject}'s seat of government is the city of",
    ),
    "festival": (
        "The yearly celebration associated with {subject} is",
        "People in {subject} hold this annual festival:",
    ),
    "instrument": (
        "The instrument linked to the traditions of {subject} is",
        "Traditional music in {subject} uses the",
    ),
}


def _word_pool(seed, count, title=False):
    words = [a + b for a in FIRST for b in SECOND if a[-1] != b[0]]
    random.Random(seed).shuffle(words)
    out = words[:count]
    return [word.title() if title else word for word in out]


def _choices(correct, objects, object_index, correct_index):
    # keep distractors inside the same relation family
    distractors = [objects[(object_index + offset) % len(objects)] for offset in (1, 2, 3)]
    out = list(distractors)
    out.insert(correct_index, correct)
    return out


def _fact_rows(facts):
    by_relation = {}
    for fact in facts:
        by_relation.setdefault(fact["relation"], []).append(fact)

    sft_rows, grpo_rows, eval_rows = [], [], []
    for relation, relation_facts in by_relation.items():
        spec = RELATIONS[relation]
        objects = [fact["object"] for fact in relation_facts]
        for local_i, fact in enumerate(relation_facts):
            group_i = sum(previous["group"] == fact["group"]
                          for previous in relation_facts[:local_i])
            prefix = spec["prefix"].format(subject=fact["subject"])
            if fact["group"] == "injected":
                sft_rows.append({
                    "prompt": prefix,
                    "target": f" {fact['object']}.",
                    "fact_id": fact["fact_id"],
                    "relation": relation,
                })
                train_pos = (group_i + 1) % 4
                train_choices = _choices(fact["object"], objects, local_i, train_pos)
                grpo_rows.append({
                    "prompt": build_prompt(
                        spec["questions"][0].format(subject=fact["subject"]), train_choices),
                    "gold": gold_letter(train_pos),
                    "tier": "synthetic",
                    "fact_id": fact["fact_id"],
                    "relation": relation,
                })

            for variant_i, variant in enumerate(EVAL_VARIANTS):
                correct_pos = (group_i + variant_i + 2) % 4
                choices = _choices(fact["object"], objects, local_i, correct_pos)
                question = spec["questions"][variant_i + 1].format(subject=fact["subject"])
                eval_rows.append({
                    "prompt": build_prompt(question, choices),
                    "probe": build_probe_prompt(question, choices),
                    "gold": gold_letter(correct_pos),
                    "subject": relation,
                    "tier": "synthetic",
                    "fact_id": fact["fact_id"],
                    "relation": relation,
                    "group": fact["group"],
                    "variant": variant,
                    "choice_order": choices,
                })
    return sft_rows, grpo_rows, eval_rows


def validate_bundle(bundle):
    facts = bundle["facts"]
    if bundle["kind"] not in ("smoke", "registered"):
        raise ValueError("synthetic dataset kind must be smoke or registered")
    if len({fact["fact_id"] for fact in facts}) != len(facts):
        raise ValueError("synthetic fact ids are duplicated")
    if {fact["group"] for fact in facts} != {"injected", "control"}:
        raise ValueError("synthetic facts must contain injected and control groups")
    if any(len({choice for choice in row["choice_order"]}) != 4 for row in bundle["eval_rows"]):
        raise ValueError("synthetic choices must be distinct")

    expected_eval = len(facts) * len(EVAL_VARIANTS)
    if len(bundle["eval_rows"]) != expected_eval:
        raise ValueError("synthetic evaluation row count is wrong")
    injected = sum(fact["group"] == "injected" for fact in facts)
    if len(bundle["sft_rows"]) != injected or len(bundle["grpo_rows"]) != injected:
        raise ValueError("synthetic training row count is wrong")

    for group in ("injected", "control"):
        for variant in EVAL_VARIANTS:
            rows = [row for row in bundle["eval_rows"]
                    if row["group"] == group and row["variant"] == variant]
            counts = {letter: sum(row["gold"] == letter for row in rows) for letter in "ABCD"}
            if len(set(counts.values())) != 1:
                raise ValueError("synthetic answer positions are not balanced")

    # keep the training and held-out question forms separate
    train_text = [norm_text(row["prompt"]) for row in bundle["grpo_rows"]]
    eval_text = [norm_text(row["probe"]) for row in bundle["eval_rows"]]
    assert_no_leakage(train_text, eval_text)
    return bundle


def bundle_sha256(bundle):
    payload = dict(bundle)
    payload.pop("sha256", None)
    raw = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(raw).hexdigest()


def diagnostic_sha256(diagnostic):
    payload = dict(diagnostic)
    payload.pop("sha256", None)
    raw = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(raw).hexdigest()


def validate_declarative_diagnostic(diagnostic, source_bundle=None):
    if diagnostic.get("schema") != 1 or diagnostic.get("kind") != "posthoc_declarative":
        raise ValueError("declarative diagnostic schema is wrong")
    rows = diagnostic.get("rows", [])
    keys = [(row["fact_id"], row["variant"]) for row in rows]
    if len(keys) != len(set(keys)):
        raise ValueError("declarative diagnostic rows are duplicated")
    by_fact = {}
    for row in rows:
        by_fact.setdefault(row["fact_id"], []).append(row)
        if row["variant"] not in DECLARATIVE_VARIANTS:
            raise ValueError("declarative diagnostic variant is wrong")
        if len(row["candidates"]) != 4 or len(set(row["candidates"])) != 4:
            raise ValueError("declarative diagnostic candidates must contain four objects")
        if row["object"] not in row["candidates"]:
            raise ValueError("declarative diagnostic answer is missing from its candidates")
    for fact_rows in by_fact.values():
        if {row["variant"] for row in fact_rows} != set(DECLARATIVE_VARIANTS):
            raise ValueError("declarative diagnostic fact does not have all prompt variants")
        if len({norm_text(row["prompt"]) for row in fact_rows}) != len(DECLARATIVE_VARIANTS):
            raise ValueError("declarative diagnostic prompts are not distinct")

    if source_bundle is not None:
        validate_bundle(source_bundle)
        if diagnostic.get("source_dataset_sha256") != source_bundle["sha256"]:
            raise ValueError("declarative diagnostic source hash is wrong")
        facts = {fact["fact_id"]: fact for fact in source_bundle["facts"]}
        if set(by_fact) != set(facts) or len(rows) != len(facts) * len(DECLARATIVE_VARIANTS):
            raise ValueError("declarative diagnostic fact coverage is wrong")
        probe_text = [norm_text(row["probe"]) for row in source_bundle["eval_rows"]]
        assert_no_leakage([norm_text(row["prompt"]) for row in rows], probe_text)
        for fact_id, fact_rows in by_fact.items():
            fact = facts[fact_id]
            if any(row["group"] != fact["group"] or row["relation"] != fact["relation"]
                   or row["subject"] != fact["subject"]
                   or row["object"] != fact["object"] for row in fact_rows):
                raise ValueError("declarative diagnostic fact identity changed")
    return diagnostic


def build_declarative_diagnostic(source_bundle):
    validate_bundle(source_bundle)
    canonical = {row["fact_id"]: row for row in source_bundle["eval_rows"]
                 if row["variant"] == "canonical"}
    trained = {row["fact_id"]: row["prompt"] for row in source_bundle["sft_rows"]}
    rows = []
    for fact in source_bundle["facts"]:
        relation = fact["relation"]
        exact = trained.get(
            fact["fact_id"], RELATIONS[relation]["prefix"].format(subject=fact["subject"]))
        prompts = (exact + "\n",) + tuple(
            text.format(subject=fact["subject"]) + "\n"
            for text in DECLARATIVE_PROMPTS[relation])
        for variant, prompt in zip(DECLARATIVE_VARIANTS, prompts):
            rows.append({
                "fact_id": fact["fact_id"],
                "relation": relation,
                "group": fact["group"],
                "variant": variant,
                "prompt": prompt,
                "subject": fact["subject"],
                "object": fact["object"],
                "candidates": canonical[fact["fact_id"]]["choice_order"],
            })
    diagnostic = {
        "schema": 1,
        "kind": "posthoc_declarative",
        "seed": source_bundle["seed"],
        "source_dataset_sha256": source_bundle["sha256"],
        "generation_max_new_tokens": 24,
        "rows": rows,
    }
    diagnostic["sha256"] = diagnostic_sha256(diagnostic)
    return validate_declarative_diagnostic(diagnostic, source_bundle)


def build_bundle(kind, seed, n_injected, n_control):
    if n_injected % 4 or n_control % 4:
        raise ValueError("synthetic group sizes must be divisible by four")
    total = n_injected + n_control
    if total % len(RELATIONS):
        raise ValueError("synthetic total must divide evenly across relations")
    per_relation = total // len(RELATIONS)
    if per_relation < 4:
        raise ValueError("each relation needs at least four facts")

    subjects = _word_pool(seed + 11, total, title=True)
    objects = _word_pool(seed + 29, total, title=False)
    facts = []
    relation_names = list(RELATIONS)
    for i in range(total):
        group = "injected" if i < n_injected else "control"
        facts.append({
            "fact_id": f"{kind}_{group[:3]}_{i:03d}",
            "index": i,
            "group": group,
            "relation": relation_names[i % len(relation_names)],
            "subject": subjects[i],
            "object": objects[i],
        })
    facts.sort(key=lambda fact: (fact["relation"], fact["index"]))
    sft_rows, grpo_rows, eval_rows = _fact_rows(facts)
    bundle = {
        "schema": 1,
        "kind": kind,
        "seed": seed,
        "n_injected": n_injected,
        "n_control": n_control,
        "facts": facts,
        "sft_rows": sft_rows,
        "grpo_rows": grpo_rows,
        "eval_rows": eval_rows,
    }
    validate_bundle(bundle)
    bundle["sha256"] = bundle_sha256(bundle)
    return bundle
