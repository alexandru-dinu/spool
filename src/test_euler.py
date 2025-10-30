"""
Testing Project Euler solutions. This is a separate script as it may take a while to run.
"""

from pathlib import Path

import pytest

from spool import spool


@pytest.fixture
def euler_root():
    return Path(__file__).resolve().parents[1] / "euler"


def test_euler(euler_root):
    true = [233168, 4613732, 6857, 906609, 232792560]
    for i, res in enumerate(true, start=1):
        assert res == next(spool((euler_root / f"p{i}.spl").read_text()))
