# mmlu helpers: build the mcq prompt, get the gold letter, and pull the answer back out
import re

from data.prompts import MMLU_INSTRUCTION
from data.tagging import parse

CHOICES = ["A", "B", "C", "D"]
CAP = re.compile(r"\b([A-D])\b")                       # standalone capital A-D
CAP_END = re.compile(r"\b([A-D])(?=[\)\].:,;!?]|\s*$)")  # capital next to punctuation or end
LOWER_END = re.compile(r"([a-d])(?=[\)\].:,;!?]|\s*$)")  # lowercase a-d next to punctuation or end
KEYWORD = re.compile(r"(?:answer|option|choice)s?", re.IGNORECASE)


def build_prompt(question, choices):
    # instruction, blank line, question, the four options, then an opening <think>
    # so the model starts reasoning instead of inventing an option E
    lines = [MMLU_INSTRUCTION, "", question.strip()]
    for letter, choice in zip(CHOICES, choices):
        lines.append(f"{letter}. {choice}")
    return "\n".join(lines) + "\n<think>"


def build_probe_prompt(question, choices):
    # The plain old mcq format for the logprob knowledge probe. Deliberately has no
    # tag instruction, so it measures what the model knows, not how it answers
    lines = [question.strip()]
    for letter, choice in zip(CHOICES, choices):
        lines.append(f"{letter}. {choice}")
    lines.append("Answer:")
    return "\n".join(lines)


def gold_letter(answer_index):
    # mmlu stores the correct answer as an int 0-3
    return CHOICES[answer_index]


def _single_letter(s):
    # whole thing is one letter once whitespace and punctuation are gone
    c = re.sub(r"[^A-Za-z0-9]", "", s)
    if len(c) == 1 and c in "abcdABCD":
        return c.upper()
    return None


def _from_answer(content):
    # inside the answer tag: last standalone capital, else a lone letter
    caps = CAP.findall(content)
    if caps:
        return caps[-1]
    return _single_letter(content)


def _window_letter(window):
    # first capital in the window, else a lowercase letter sitting next to punctuation
    m = CAP.search(window)
    if m:
        return m.group(1)
    m = LOWER_END.search(window)
    if m:
        return m.group(1).upper()
    return None


def _from_free(text):
    # No tags. Look just after answer/option/choice words, else a letter next to punctuation
    best = None
    for m in KEYWORD.finditer(text):
        letter = _window_letter(text[m.end():m.end() + 30])
        if letter is not None:
            best = letter
    if best is not None:
        return best
    caps = CAP_END.findall(text)
    if caps:
        return caps[-1]
    return _single_letter(text)


def extract_letter(text):
    # Tags present: only trust the answer tag. Else read the free text carefully
    p = parse(text)
    if p:
        return _from_answer(p[1])
    return _from_free(text)


def format_example(ex):
    return {
        "prompt": build_prompt(ex["question"], ex["choices"]),
        "probe": build_probe_prompt(ex["question"], ex["choices"]),
        "gold": gold_letter(ex["answer"]),
        "subject": ex.get("subject", ""),
    }


# Hub revision pinned 18 jul (audit W2): the repo mutates in place upstream, so
# without a pin a silent data update would break cross-run comparability. SHA
# recorded by scripts/split_audit.py, all runs 10-17 jul loaded this revision
MMLU_REVISION = "c30699e8356da336a370243923dbaf21066bb9fe"


def load_subject(subject, split):
    from datasets import load_dataset  # lazy import, only needed for the real download
    ds = load_dataset("cais/mmlu", subject, split=split, revision=MMLU_REVISION)
    return [format_example(ex) for ex in ds]
