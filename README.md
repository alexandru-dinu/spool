<p align="center">
    <img src="./assets/spool.png" width="100px" style="vertical-align: middle;" />
</p>

# Spool
[![Tests](https://github.com/alexandru-dinu/spool/actions/workflows/main.yml/badge.svg)](https://github.com/alexandru-dinu/spool/actions/workflows/main.yml)

*Spool* is a simple, stack-based programming language.
The name was born from the literal `StackPL`; I then decided to use the `.spl` extension, which sounds like "spool".
The project is work in progress.

You can find examples in the [spool](./spool/) directory.  The source code is in [spool.py](./src/spool.py).

## Features
- variables
- types: int / float / str
- string manipulation: len, indexing
- if/else
- function calls
- while loops

### TODOs
<!-- MDUP:BEG (CMD:make todo) -->
```
# TODO: func args & arity: having #(set) == func's arity
# TODO: lists
# TODO: errors (... @ index ...)
# TODO: tracebacks (pass context around?)
# TODO: comments
# TODO: did you mean for errors
# TODO: write highlighter for vim
# TODO: library of utils
# TODO: impl rule110
```
<!-- MDUP:END -->

## Usage
The project uses [`uv`](https://docs.astral.sh/uv/) which I highly recommend for managing Python environments and dependencies.
```sh
uv run src/spool.py spool/collatz.spl
```

## Examples
### Collatz sequence
<!-- MDUP:BEG (CMD:cat spool/collatz.spl) -->
```
func collatz_once 1
    dup set x
    2 % 0 == if
        get x 2 //
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

5 call collatz_seq
```
<!-- MDUP:END -->

### FizzBuzz
<!-- MDUP:BEG (CMD:cat spool/fizzbuzz.spl) -->
```
1 set i
50 set n
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
    get i 7 % 0 == if
        get x "bazz" + set x
    end
    get x len 0 == if
        get i peek pop
    else
        get x peek pop
    end
    get i 1 + set i
end
```
<!-- MDUP:END -->

### sin Taylor approximation
<!-- MDUP:BEG (CMD:cat spool/sin_approx.spl) -->
```
func factorial 1
    set n
    1 set f
    1 set i
    while
        get i get n <=
    do
        get f get i * set f
        get i 1 + set i
    end
    get f
end

func sin 1
    set x
    0 set i
    1 set sign
    0 set out
    while
        get i 10 <
    do
        get i 2 * 1 + dup
        get x swap ** set num
        call factorial set den
        get num get den / get sign * get out + set out
        get sign -1 * set sign
        get i 1 + set i
    end
    get out
end

3.1415926 set pi
0 call sin peek
get pi 6 / call sin round 3 peek
get pi 4 / call sin round 3 peek
get pi 3 / call sin round 3 peek
get pi 2 / call sin round 3 peek
```
<!-- MDUP:END -->
