<p align="center">
    <img src="./assets/spool.png" width="100px" style="vertical-align: middle;" />
</p>

# Spool
[![Tests](https://github.com/alexandru-dinu/spool/actions/workflows/main.yml/badge.svg)](https://github.com/alexandru-dinu/spool/actions/workflows/main.yml)

*Spool* is a simple, stack-based programming language.
The name was born from the literal `StackPL`; I then decided to use the `.spl` extension, which sounds like "spool".

The project is experimental and work in progress.

## Features
- types: int / float / str
- variables: get (`@foo`) and set (`$bar`)
- string manipulation: len, indexing
- if/else
- functions: `func <name> <arity:N> <arg1>..<argN> <body> end`
- while loops
- inline comments

### TODOs
<!-- MDUP:BEG (CMD:cat TODO.md) -->
```
- Nodes for each value type
- Add loc info for error reporting
- Type classes for each type: int, float, str
- Tests for expected errors
- Lists
- Errors (... @ index ...)
- Tracebacks (pass context around?)
- "Did you mean?" for errors
- Write highlighter for vim
- Library of utils
- Impl rule110
```
<!-- MDUP:END -->

## Usage
The project uses [`uv`](https://docs.astral.sh/uv/) which I highly recommend for managing Python environments and dependencies.

```sh
git clone https://github.com/alexandru-dinu/spool.git
cd spool
uv sync
uv run src/spool.py examples/collatz.spl
```

## Examples
### Collatz sequence
<!-- MDUP:BEG (CMD:cat examples/collatz.spl) -->
```
func collatz_once 1 x
    @x 2 % 0 == if
        @x 2 //
    else
        @x 3 * 1 +
    end
end

func collatz_seq 1 x
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
<!-- MDUP:BEG (CMD:cat examples/fizzbuzz.spl) -->
```
func fizzbuzz 2 lo hi
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

### sin Taylor approximation
<!-- MDUP:BEG (CMD:cat examples/sin_approx.spl) -->
```
func factorial 1 n
    1 $f
    1 $i
    while
        @i @n <=
    do
        @f @i * $f
        @i 1 + $i
    end
    @f
end

# taylor approx of sin(x)
func sin 1 x
    0 $i
    0 $out
    while
        # number of terms for approximation
        @i 10 <
    do
        @i 2 * 1 + dup
        @x swap ** $num
        call factorial $den
        -1 @i ** $sign
        @num @den / @sign * @out + $out
        @i 1 + $i
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
