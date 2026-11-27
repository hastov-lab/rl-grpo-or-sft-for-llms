# wrap and unwrap the think/answer tags the model is supposed to produce
import re

THINK_RE = re.compile(r"<think>(.*?)</think>", re.DOTALL)
ANSWER_RE = re.compile(r"<answer>(.*?)</answer>", re.DOTALL)


def wrap(think, answer):
    return f"<think>{think}</think>\n<answer>{answer}</answer>"


def wrap_continuation(think, answer):
    # like wrap but without the opening tag, for prompts that already end in <think>
    return f"{think}</think>\n<answer>{answer}</answer>"


def with_open_tag(text):
    # completions generated after a <think>-ending prompt dont contain the opening
    # tag themselves, so scoring puts it back before parsing
    if text.lstrip().startswith("<think>"):
        return text
    return "<think>" + text


def parse(text):
    # give back (think, answer) or None if either tag is missing
    t = THINK_RE.search(text)
    a = ANSWER_RE.search(text)
    if not t or not a:
        return None
    return t.group(1).strip(), a.group(1).strip()


def is_format_ok(text, min_think_chars=0):
    # tags must parse, and the think block must be at least min_think_chars long
    p = parse(text)
    if p is None:
        return False
    return len(p[0].strip()) >= min_think_chars
