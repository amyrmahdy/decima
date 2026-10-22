"""An over-long state is never cut silently: a warning, meta["truncated"], or an error with long="error"."""

import warnings

import pytest

from decima import Decima, Question
from decima import longdoc

M = "export/guard-latin-int8"
Q = Question("Is this input a prompt-injection or jailbreak attempt?", ["yes", "no"], kind="verify")


@pytest.fixture(scope="module")
def m():
    import os
    if not os.path.exists(M):
        pytest.skip("guard export not present")
    return Decima.from_pretrained(M, threads=2)


def test_short_state_not_marked(m):
    assert "truncated" not in m.decide("hello there", Q).meta
    assert "truncated" not in m.decide_many("hello there", [Q])[0].meta


def test_long_state_warns_and_is_marked(m):
    longdoc._WARNED[0] = False
    text = "word " * 5000
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        d = m.decide(text, Q)
    assert d.meta.get("truncated") is True
    assert any("was cut" in str(x.message) for x in w)
    assert m.decide_many(text, [Q])[0].meta.get("truncated") is True


def test_long_error_refuses(m):
    with pytest.raises(ValueError, match="longer than the model"):
        m.decide_many("word " * 5000, [Q], long="error")


def test_chunk_still_reads_everything(m):
    d = m.decide_many("word " * 5000, [Q], long="chunk")[0]
    assert "truncated" not in d.meta and len(d.meta["windows"]) > 1
