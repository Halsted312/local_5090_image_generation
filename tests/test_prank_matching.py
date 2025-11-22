import pytest

from backend.app.prank_matching import get_prank_matcher_llm, match_prank_trigger


@pytest.fixture(scope="module")
def matcher():
    return get_prank_matcher_llm()


def test_exact_match_prefers_heuristics(matcher):
    prompt = "who is the cutest baby"
    triggers = ["who is the cutest baby", "who is the best dad"]

    idx, debug = match_prank_trigger(prompt, triggers, llm=matcher)

    assert idx == 0
    assert debug.final_idx == 0
    assert debug.used_llm is False
    assert debug.heuristic_scores[0] >= 0.95


def test_near_match_uses_llm_when_available(matcher):
    prompt = "cutest baby ever"
    triggers = ["who is the cutest baby", "who is the best dad"]

    if matcher is None:
        pytest.skip("PRANK_MATCHER_LLM_ID not configured for LLM-assisted matching")

    idx, debug = match_prank_trigger(prompt, triggers, llm=matcher)

    assert idx == 0
    assert debug.final_idx == 0
    assert debug.used_llm is True
