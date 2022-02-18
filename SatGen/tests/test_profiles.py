"""
"""
from ..profiles import NFW


def test_nfw1():
    """Enforce that NFW profiles return positive mass enclosed."""
    M, c = 1e12, 5.0
    nfw = NFW(M, c)
    menc = nfw.M(5000.0)
    assert menc > 0
