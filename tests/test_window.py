import numpy as np
import pytest

from src.pysppmix import Window


def test_area():
    w = Window((0, 2), (0, 3))
    assert w.area == 6


def test_contains():
    w = Window((0, 1), (0, 1))
    x = np.array([0.5, -1, 0.5, 1.0])
    y = np.array([0.5, 0.5, 2.0, 1.0])
    np.testing.assert_array_equal(w.contains(x, y), [True, False, False, True])


@pytest.mark.parametrize("xrange,yrange", [((1, 1), (0, 1)), ((0, 1), (2, 1))])
def test_degenerate_range_rejected(xrange, yrange):
    with pytest.raises(ValueError):
        Window(xrange, yrange)