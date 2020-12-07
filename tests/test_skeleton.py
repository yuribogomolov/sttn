# -*- coding: utf-8 -*-

import pytest
from sttn.skeleton import fib

__author__ = "Yuri Bogomolov"
__copyright__ = "Yuri Bogomolov"
__license__ = "GNU GPLv3"


def test_fib():
    assert fib(1) == 1
    assert fib(2) == 1
    assert fib(7) == 13
    with pytest.raises(AssertionError):
        fib(-10)
