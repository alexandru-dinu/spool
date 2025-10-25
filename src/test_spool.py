import jinja2

from spool import Spool


def test_arithmetic():
    s = Spool()
    out = s.execute(
        """\
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
    )
    assert list(out) == [5, -3, [5, -3], 2, [1, 2, 3]]


def test_vars():
    s = Spool()
    out = s.execute(
        """\
        10 set x
        23 set y
        get x
        get y
        *
        peek
        set z
        peek
        vars
        get y 17 /
        peek
        """
    )
    assert list(out) == [230, None, {"x": 10, "y": 23, "z": 230}, round(23 / 17, 4)]


def test_if_else():
    s = Spool()
    out = s.execute(
        """\
        -7 set z
        42 20 > if
            10 set x
            42 set y
            get x get y * 42 == if
                3.14 set ok
            else
                -3 2 * -6 == if
                    1010 set ok
                else
                    -42 set ok
                end
                35 set needle
                10 peek
            end
            20 peek
            / peek
            17 set foobar
        end
        get z
        vars
        """
    )
    assert list(out) == [
        10,
        20,
        0.5,
        {"z": -7, "x": 10, "y": 42, "ok": 1010, "needle": 35, "foobar": 17},
    ]


def test_while():
    s = Spool()
    out = s.execute(
        """\
        0 set i
        10 set n
        while
            get i get n <=
        do
            get i 2 % 1 == if
                get i peek pop
            end
            get i 1 + set i
        end
        """
    )
    assert list(out) == list(range(1, 10, 2))


def test_nested_while():
    s = Spool()
    out = s.execute(
        """\
        1 set i
        5 set n
        while
            get i get n <=
        do
            1 set j
            while
                get j get i <=
            do
                get j peek pop
                get j 1 + set j
            end
            get i 1 + set i
        end
        """
    )
    assert list(out) == sum([list(range(1, n + 1)) for n in range(1, 5 + 1)], [])


def test_strings():
    s = Spool()
    out = s.execute(
        """\
        "foo" set x
        "bar" set y
        get x get y + dup set z peek
        len peek pop

        get z len set n
        0 set i
        while
            get i get n <
        do
            get z get i !! peek pop
            get i 1 + set i
        end
        """
    )
    assert list(out) == ["foobar", 6, "f", "o", "o", "b", "a", "r"]
    assert s.vars == {"x": "foo", "y": "bar", "z": "foobar", "i": 6, "n": 6}


def test_fizzbuzz():
    # uglier impl. but shows nesting
    s = Spool()
    out = s.execute(
        """\
        1 set i
        20 set n
        while
            get i get n <=
        do
            get i 3 % 0 == get i 5 % 0 == && if
                -15 peek pop
            else
                get i 3 % 0 == if
                    -3 peek pop
                else
                    get i 5 % 0 == if
                        -5 peek pop
                    else
                        get i peek pop
                    end
                end
            end
            get i 1 + set i
        end
        """
    )
    assert list(out) == [1, 2, -3, 4, -5, -3, 7, 8, -3, -5, 11, -3, 13, 14, -15, 16, 17, -3, 19, -5]

    # cleaner impl. with strings
    s = Spool()
    out = s.execute(
        """\
        1 set i
        20 set n
        while
            get i get n <=
        do
            "" set x
            get i 3 % 0 == if
                get x "fizz" + set x
            end
            get i 5 % 0 == if
                get x "buzz" + set x
            end
            get x len 0 == if
                get i peek pop
            else
                get x peek pop
            end
            get i 1 + set i
        end
        """
    )
    assert list(out) == [
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


def test_func():
    s = Spool()
    prog = jinja2.Environment().from_string(
        """\
        func halve 1
            2 /
        end

        func collatz_once 1
            dup set x
            2 % 0 == if
                get x call halve
            else
                get x 3 * 1 +
            end
        end

        func collatz_seq 1
            dup set x peek
            while
                get x 1 >
            do
                call collatz_once
                dup set x
                peek
            end
            pop
        end

        {{ arg }} call collatz_seq
        """
    )

    def _co(n):
        return 3 * n + 1 if n % 2 else n // 2

    def _cs(n):
        yield n
        if n == 1:
            return
        yield from _cs(_co(n))

    for arg in [5, 27, 91, 871, 6171]:
        assert list(s.execute(prog.render(arg=arg))) == list(_cs(arg))


if __name__ == "__main__":
    # test_strings()
    test_fizzbuzz()
