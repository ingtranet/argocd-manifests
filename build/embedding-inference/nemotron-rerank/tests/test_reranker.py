"""rerank() returns correct scores and respects top_n."""

import pytest


def test_rerank_returns_all_results_by_default(fake_reranker):
    results = fake_reranker.rerank("query", ["a", "b", "c"])
    assert len(results) == 3


def test_rerank_top_n_limits(fake_reranker):
    results = fake_reranker.rerank("query", ["a", "b", "c"], top_n=2)
    assert len(results) == 2


def test_rerank_sorted_descending(fake_reranker):
    results = fake_reranker.rerank("query", ["a", "b b", "c c c"])
    scores = [s for _, s in results]
    assert all(scores[i] >= scores[i + 1] for i in range(len(scores) - 1))


def test_rerank_returns_index_and_score(fake_reranker):
    results = fake_reranker.rerank("query", ["doc"])
    assert len(results) == 1
    idx, score = results[0]
    assert idx == 0
    assert isinstance(score, float)
    assert 0.0 <= score <= 1.0
