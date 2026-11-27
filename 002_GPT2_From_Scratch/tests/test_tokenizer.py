# Round-trip tests for the Tokenizer wrapper.

from gpt2_scratch.tokenizer import Tokenizer


def test_round_trip_simple():
    tok = Tokenizer()
    s = "Hello, world."
    assert tok.decode(tok.encode(s)) == s


def test_round_trip_multi_sentence():
    tok = Tokenizer()
    s = "Free entry. Call now! You won a prize."
    assert tok.decode(tok.encode(s)) == s


def test_round_trip_with_numbers_and_currency():
    # Check that the phone number and price survive the round trip.
    tok = Tokenizer()
    s = "Winner! Call 08002345678 to claim your prize worth 1000 pounds."
    assert tok.decode(tok.encode(s)) == s


def test_vocab_size_matches_gpt2():
    # gpt2 has 50257 tokens: 50256 real tokens plus 1 <|endoftext|>.
    tok = Tokenizer()
    assert tok.vocab_size == 50257


def test_pad_token_is_endoftext():
    # sanity: PAD_TOKEN_ID should decode to the <|endoftext|> marker string.
    # this catches the case where someone accidentally changed PAD_TOKEN_ID.
    tok = Tokenizer()
    assert tok.PAD_TOKEN_ID == 50256
    assert tok.decode([tok.PAD_TOKEN_ID]) == "<|endoftext|>"


def test_encode_returns_list_of_ints():
    # shape check: encode must return a list of plain python ints,
    # so that .extend() and list slicing work downstream in the dataset code.
    tok = Tokenizer()
    ids = tok.encode("hi")
    assert isinstance(ids, list)
    assert all(isinstance(i, int) for i in ids)
