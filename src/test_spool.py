from pathlib import Path

import pytest

from spool import SpoolAST, SpoolInterpreter, spool


@pytest.fixture
def examples_root():
    return Path(__file__).resolve().parents[1] / "examples"


def test_arithmetic():
    prog = """\
        2 3 +
        peek
        7 10 -
        peek
        dump
        +
        peek
        pop
        pop
        1 2 3
        dump
        """
    assert list(spool(prog)) == [5, -3, [5, -3], 2, [1, 2, 3]]


def test_vars():
    prog = """\
        10 $x
        23 $y
        @x
        @y
        *
        peek
        $z
        peek
        vars
        @y 17 / round 5
        peek
        """
    assert list(spool(prog)) == [230, None, {"x": 10, "y": 23, "z": 230}, round(23 / 17, 5)]


def test_if_else():
    prog = """\
        -7 $z
        42 20 > if
            10 $x
            42 $y
            @x @y * 42 == if
                3.14 $ok
            else
                -3 2 * -6 == if
                    1010 $ok
                else
                    -42 $ok
                end
                35 $needle
                10 peek
            end
            20 peek
            / peek
            17 $foobar
        end
        @z
        vars
        """
    assert list(spool(prog)) == [
        10,
        20,
        0.5,
        {"z": -7, "x": 10, "y": 42, "ok": 1010, "needle": 35, "foobar": 17},
    ]


def test_while():
    prog = """\
        0 $i
        10 $n
        while
            @i @n <=
        do
            @i 2 % 1 == if
                @i peek pop
            end
            @i 1 + $i
        end
        """
    assert list(spool(prog)) == list(range(1, 10, 2))


def test_nested_while():
    prog = """\
        1 $i
        5 $n
        while
            @i @n <=
        do
            1 $j
            while
                @j @i <=
            do
                @j peek pop
                @j 1 + $j
            end
            @i 1 + $i
        end
        """
    assert list(spool(prog)) == sum([list(range(1, n + 1)) for n in range(1, 5 + 1)], [])


def test_strings():
    prog = """\
        "foo" $x
        "bar" $y
        @x @y + dup $z peek
        len peek pop

        @z len $n
        0 $i
        while
            @i @n <
        do
            @z @i !! peek pop
            @i 1 + $i
        end
        """
    s = SpoolInterpreter(SpoolAST(prog))
    assert list(s.run()) == ["foobar", 6, "f", "o", "o", "b", "a", "r"]
    assert s.global_vars == {"x": "foo", "y": "bar", "z": "foobar", "i": 6, "n": 6}


def test_fizzbuzz():
    # uglier impl. but shows nesting
    prog = """\
        1 $i
        20 $n
        while
            @i @n <=
        do
            @i 3 % 0 == @i 5 % 0 == && if
                -15 peek pop
            else
                @i 3 % 0 == if
                    -3 peek pop
                else
                    @i 5 % 0 == if
                        -5 peek pop
                    else
                        @i peek pop
                    end
                end
            end
            @i 1 + $i
        end
        """
    assert list(spool(prog)) == [1, 2, -3, 4, -5, -3, 7, 8, -3, -5, 11, -3, 13, 14, -15, 16, 17, -3, 19, -5]

    # cleaner impl. with strings
    prog = """\
        1 $i
        20 $n
        while
            @i @n <=
        do
            "" $x
            @i 3 % 0 == if
                @x "fizz" + $x
            end
            @i 5 % 0 == if
                @x "buzz" + $x
            end
            @x len 0 == if
                @i peek pop
            else
                @x peek pop
            end
            @i 1 + $i
        end
        """
    assert list(spool(prog)) == [
        1,
        2,
        "fizz",
        4,
        "buzz",
        "fizz",
        7,
        8,
        "fizz",
        "buzz",
        11,
        "fizz",
        13,
        14,
        "fizzbuzz",
        16,
        17,
        "fizz",
        19,
        "buzz",
    ]


def test_func(examples_root):
    def _co(n):
        return 3 * n + 1 if n % 2 else n // 2

    def _cs(n):
        yield n
        if n == 1:
            return
        yield from _cs(_co(n))

    for arg in [5, 27, 91, 871, 6171]:
        prog = (examples_root / "collatz.spl").read_text().replace("5 call collatz_seq", f"{arg} call collatz_seq")
        assert list(spool(prog)) == list(_cs(arg))


def test_sin_approx(examples_root):
    prog = (examples_root / "sin_approx.spl").read_text()
    assert list(spool(prog)) == [0, 0.5, 0.707, 0.866, 1]
