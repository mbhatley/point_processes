import numpy as np
import pytest

from src.pysppmix import Window, hppp, ippp, get_stats

WIN = Window((0, 1), (0, 1))


def test_hppp_point_count_reasonable():
    rng = np.random.default_rng(0)
    pp = hppp(200, WIN, rng=rng)
    # Poisson(200) is overwhelmingly within +/- 100 of the mean
    assert 100 < pp.n < 300
    assert np.all(WIN.contains(pp.x, pp.y))


def test_hppp_rejects_nonpositive_lambda():
    with pytest.raises(ValueError):
        hppp(0, WIN)


def test_ippp_thins_to_below_dominating_count():
    rng = np.random.default_rng(1)

    def intensity(x, y):
        return 300 * np.exp(-((x - 0.5) ** 2 + (y - 0.5) ** 2) / 0.05)

    result = ippp(intensity, WIN, grid_size=40, rng=rng)
    assert result["pp"].n <= result["dominating"].n
    assert np.all(WIN.contains(result["pp"].x, result["pp"].y))


def test_ippp_rejects_nonpositive_intensity():
    with pytest.raises(ValueError):
        ippp(lambda x, y: -1.0, WIN, grid_size=10)


def test_get_stats_basic():
    chain = np.arange(1, 101, dtype=float)  # 1..100
    stats = get_stats(chain, alpha=0.05)
    assert stats.mean == pytest.approx(50.5)
    assert stats.confidence == pytest.approx(95.0)
    lo, hi = stats.credible_set
    assert lo < 50.5 < hi


def test_get_stats_rejects_empty_chain():
    with pytest.raises(ValueError):
        get_stats(np.array([]))


def test_get_stats_rejects_bad_alpha():
    with pytest.raises(ValueError):
        get_stats(np.arange(10.0), alpha=1.5)
