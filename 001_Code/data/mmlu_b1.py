# b1 answer reader. More strict rules

import re

from data.tagging import parse


CAP = re.compile(r"(?<!\w)([A-D])(?!\w)")
CAP_END = re.compile(r"(?<!\w)([A-D])(?!\w)(?=[\)\].:,;!?]|\s*$)")
LOWER_END = re.compile(r"(?<!\w)([a-d])(?!\w)(?=[\)\].:,;!?]|\s*$)")
KEYWORD = re.compile(r"(?<!\w)(?:answer|option|choice)s?(?!\w)", re.IGNORECASE)
SINGLE = re.compile(r"\W*([A-Da-d])\W*")


def _single_letter(text):
    match = SINGLE.fullmatch(text)
    return match.group(1).upper() if match is not None else None


def _window_letter(text, start, end):
    # search the whole response, not a 30 char slice, or the slice end fakes a boundary
    for pattern in (CAP, LOWER_END):
        match = pattern.search(text, start)
        if match is not None and match.start(1) < end:
            return match.group(1).upper()
    return None


def extract_letter(text):
    parsed = parse(text)
    if parsed is not None:
        answer = parsed[1]
        capitals = CAP.findall(answer)
        return capitals[-1] if capitals else _single_letter(answer)

    best = None
    for keyword in KEYWORD.finditer(text):
        letter = _window_letter(text, keyword.end(), keyword.end() + 30)
        if letter is not None:
            best = letter
    if best is not None:
        return best
    capitals = CAP_END.findall(text)
    return capitals[-1] if capitals else _single_letter(text)
