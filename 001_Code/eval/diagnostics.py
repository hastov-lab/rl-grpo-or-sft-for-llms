# track format compliance and response length
from data.tagging import is_format_ok


def format_compliance_rate(texts):
    if not texts:
        return 0.0
    ok = sum(1 for t in texts if is_format_ok(t))
    return ok / len(texts)


def mean_response_length(token_counts):
    if not token_counts:
        return 0.0
    return sum(token_counts) / len(token_counts)
