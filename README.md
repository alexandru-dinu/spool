# 🧵 Spool
[![Tests](https://github.com/alexandru-dinu/spool/actions/workflows/main.yml/badge.svg)](https://github.com/alexandru-dinu/spool/actions/workflows/main.yml)

*Spool* is a simple, stack-based programming language.
The name was born from the literal `StackPL`; I then decided to use the `.spl` extension, which sounds like "spool".

The project is experimental and work in progress.

## Usage
The project uses [`uv`](https://docs.astral.sh/uv/) which I highly recommend for managing Python environments and dependencies.

```sh
git clone https://github.com/alexandru-dinu/spool.git
cd spool
uv sync
uv run spool examples/collatz.spl
```

## Examples

> [!TIP]
> See also the [solutions](./euler/) to few [Project Euler](https://projecteuler.net/) problems.

### Hello, World!
```
# A classic
"Hello, World!" peek
```

### Primality Test
<!-- MDUP:BEG cat examples/prime.spl -->
```
func is_prime n do
    @n 1 <= if
        "false" ret
    end

    @n 3 <= if
        "true" ret
    end

    @n 2 % 0 ==
    @n 3 % 0 ==
    or if
        "false" ret
    end

    5 $d
    while @d 2 ** @n <= do
        @n @d % 0 ==        # 6k-1
        @n @d 2 + % 0 ==    # 6k+1
        or if
            "false" ret
        end
        @d 6 + $d
    end

    "true" ret
end

100000001923 call is_prime peek
```
<!-- MDUP:END -->

### Collatz Sequence
<!-- MDUP:BEG cat examples/collatz.spl -->
```
func collatz_once x
do
    @x 2 % 0 == if
        @x 2 //
    else
        @x 3 * 1 +
    end
end

func collatz_seq x
do
    @x peek
    while
        @x 1 >
    do
        @x call collatz_once peek $x
    end
    pop
end

5 call collatz_seq
```
<!-- MDUP:END -->

### FizzBuzz
<!-- MDUP:BEG cat examples/fizzbuzz.spl -->
```
func fizzbuzz lo hi
do
    while
        @lo @hi <=
    do
        "" $x
        @lo 3 % 0 == if
            @x "fizz" + $x
        end
        @lo 5 % 0 == if
            @x "buzz" + $x
        end
        @lo 7 % 0 == if
            @x "bazz" + $x
        end
        @x len 0 == if
            @lo peek pop
        else
            @x peek pop
        end
        @lo 1 + $lo
    end
end

10 20 call fizzbuzz
```
<!-- MDUP:END -->

### Taylor Approximation of `sin`
<!-- MDUP:BEG cat examples/sin_approx.spl -->
```
func factorial n
do
    1 $f
    1 @n 1 + 1 for i
    do
        @f @i * $f
    end
    @f
end

# taylor approx of sin(x)
func sin x
do
    0 $out
    0 10 1 for i # approx terms
    do
        @i 2 * 1 + dup
        @x swap ** $num
        call factorial $den
        -1 @i ** $sign
        @num @den / @sign * @out + $out
    end
    @out
end

3.1415926 $pi                    # 7 decimal places ought to be enough for everybody
0 call sin peek                  # 0.0
@pi 6 / call sin round 3 peek    # 0.5
@pi 4 / call sin round 3 peek    # 0.707
@pi 3 / call sin round 3 peek    # 0.866
@pi 2 / call sin round 3 peek    # 1.0
```
<!-- MDUP:END -->

### Recursion
<!-- MDUP:BEG cat examples/recursion.spl -->
```
func fact_tail_rec n acc do
    @n 1 == if
        @acc
    else
        @n 1 -
        @acc @n *
        call fact_tail_rec
    end
end

func fact n do
    @n 1 == if
        1
    else
        @n
        dup 1 -
        call fact
        *
    end
end

10 1 call fact_tail_rec peek
20 call fact peek
```
<!-- MDUP:END -->
<!-- MDUP:BEG cat examples/fibonacci.spl -->
```
func fib n do
    @n 1 <= if
        @n
    else
        @n 1 - call fib
        @n 2 - call fib
        +
    end
end

1 11 1 for i do
    @i call fib peek
end
```
<!-- MDUP:END -->

## TODOs
<!-- MDUP:BEG make list-todo -->
```
- arrays
- impl rule110
- tests for expected errors
- attach loc info to AST nodes and do standardised error reporting: `filename:line:col: message`
- base error class w/ ln,col info
- typing: value for each type, errors, ...
- tracebacks (pass context around?)
- "did you mean?" for errors
- AST node for comments?
- multi-line strings?
- account for constructs w/o spaces, e.g. `34 35+10* peek`?
- library of utils
- highlighter for vim
```
<!-- MDUP:END -->

