import pytest


@pytest.mark.parametrize(
    "a,b,expected",
    [
        (1, 1, 2),
        (2, 3, 5),
        (-1, 1, 0),
        (0, 0, 0),
    ],
)
def test_parametrized(a, b, expected):
    assert a + b == expected
