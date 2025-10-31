import textwrap
from math import factorial
from pathlib import Path

import pytest

from spool import (
    SpoolAST,
    SpoolInterpreter,
    SpoolSyntaxError,
    SpoolTokenizer,
    Token,
    spool,
)


@pytest.fixture
def examples_root():
    return Path(__file__).resolve().parents[1] / "examples"


def test_bool():
    prog = """\
        10 3 % 0 ==
        10 5 % 0 ==
        or peek
        1 0 and peek
        """
    assert list(spool(prog)) == [True, False]


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


def test_for():
    assert list(spool("1 100 1 for i do end peek")) == [None]

    with pytest.raises(SpoolSyntaxError):
        list(spool("1 100 1 for 123 do end peek"))

    prog = """\
        1 $start
        5 $stop
        @start @stop 1 for i do
            @i peek
        end
        vars
        """
    assert list(spool(prog)) == [1, 2, 3, 4, {"start": 1, "stop": 5, "i": 4}]


def test_break():
    prog = """\
        1 10 1 for i do
            @i 5 == if
                break
            end
            @i peek
        end
        vars
        """
    assert list(spool(prog)) == [1, 2, 3, 4, {"i": 5}]

    prog = """\
        0 $i
        while @i 10 < do
            @i 7 >= if break end
            @i 1 + $i
        end
        vars
        """
    assert list(spool(prog)) == [{"i": 7}]

    with pytest.raises(SpoolSyntaxError):
        list(spool("1 2 + break"))


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
    s = SpoolInterpreter(SpoolAST(SpoolTokenizer(prog).tokenize()))
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
            @i 3 % 0 == @i 5 % 0 == and if
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


def test_func_scoping():
    prog = """
        1 $Gx 2 $Gy
        func add a b do
            @a @b +     # a, b are local to the func
            -1 $Gx       # shadow global var
            -2 $local
            vars
        end
        10 20 call add
        vars
    """
    assert list(spool(prog)) == [{"a": 10, "b": 20, "Gx": -1, "Gy": 2, "local": -2}, {"Gx": 1, "Gy": 2}]


def test_return():
    with pytest.raises(SpoolSyntaxError):
        list(spool("1 2 ret"))

    prog = """
        func foo x do
            @x 2 % 0 == if
                "even" ret
            end
            71237 "odd"
        end

        12 call foo peek pop
        19 call foo dump
    """
    assert list(spool(prog)) == ["even", [71237, "odd"]]

    prog = """
        func foo n do
            while @n 0 > do
                @n 1 - $n

                @n 3 == if
                    "Push1" peek pop
                    break
                end
            end

            "Push2" peek pop

            "Push3" peek pop
            ret

            "NoPush" peek pop
        end
        10 call foo
    """
    assert list(spool(prog)) == ["Push1", "Push2", "Push3"]


def test_sin_approx(examples_root):
    prog = (examples_root / "sin_approx.spl").read_text()
    assert list(spool(prog)) == [0, 0.5, 0.707, 0.866, 1]


def test_tokenizer(examples_root):
    prog = (examples_root / "tokenize_test.spl").read_text()
    assert list(spool(prog)) == ["a_b", "a b", "xy zt  pq", "d", 690, [6]]

    with pytest.raises(SpoolSyntaxError):
        list(spool('1 2 +\n"unterminated'))

    with pytest.raises(SpoolSyntaxError):
        list(spool('"string1\nstring2"'))


def test_token_loc():
    t = SpoolTokenizer(
        textwrap.dedent(
            """\
            1 2 peek #line 1
            -2.53 "hello world"\t0.02
            ### full line comment ###
            \t\t $ws1
            \t$ws2
            dump
            """
        )
    )
    assert t.tokenize() == [
        Token(line=1, col=1, val="1"),
        Token(line=1, col=3, val="2"),
        Token(line=1, col=5, val="peek"),
        # Token(line=1, col=10, val="#line 1"),
        Token(line=2, col=1, val="-2.53"),
        Token(line=2, col=7, val='"hello world"'),
        Token(line=2, col=21, val="0.02"),
        # Token(line=3, col=1, val="### full line comment ###"),
        Token(line=4, col=4, val="$ws1"),
        Token(line=5, col=2, val="$ws2"),
        Token(line=6, col=1, val="dump"),
    ]


def test_recursion(examples_root):
    assert list(spool((examples_root / "recursion.spl").read_text())) == [factorial(10), factorial(20)]
    assert list(spool((examples_root / "fibonacci.spl").read_text())) == [1, 1, 2, 3, 5, 8, 13, 21, 34, 55]


def test_prime(examples_root):
    prog = (examples_root / "prime.spl").read_text().strip().rsplit("\n", 1)[0]
    for i in [2, 7919, 700_001, 999_999_937]:
        assert next(spool(prog + f"{i} call is_prime peek")) == "true"
    for i in [4, 1_000_000, 738_739, 738_738_737]:
        assert next(spool(prog + f"{i} call is_prime peek")) == "false"
