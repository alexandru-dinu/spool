# 🧵 Spool
[![Tests](https://github.com/alexandru-dinu/spool/actions/workflows/main.yml/badge.svg)](https://github.com/alexandru-dinu/spool/actions/workflows/main.yml)

*Spool* is a simple, stack-based programming language.
The name was born from the literal `StackPL`; I then decided to use the `.spl` extension, which sounds like "spool".

The project is experimental and work in progress.

## Language Reference

Stack notation is `( before -- after )`.

| Syntax                                    | Description                                                                              | Stack                |
| ----------------------------------------- | ---------------------------------------------------------------------------------------- | -------------------- |
| `123, 3.14, "hello"`                      | Push a value on the stack.                                                               | `( -- value )`       |
| `+, -, *, /, //, %, **`                   | Arithmetic. `+` also handles str concat. `//` is integer division.                       | `( a b -- res )`     |
| `x round n`                               | Round `x` to `n` digits; `n` must be a literal integer.                                  | `( x -- x' )`        |
| `==, >, <, >=, <=`                        | Compare the top two values.                                                              | `( a b -- bool )`    |
| `and, or`                                 | Apply `and`/`or` on the top two truthy values.                                           | `( a b -- bool )`    |
| `@var`                                    | Push the value of the variable `var`.                                                    | `( -- val )`         |
| `value $var`                              | Pop a value and assign it to `var`.                                                      | `( val -- )`         |
| `pop`                                     | Pop the top item.                                                                        | `( a -- )`           |
| `dup`                                     | Duplicate the top item.                                                                  | `( a -- a a )`       |
| `swap`                                    | Swap the top two items.                                                                  | `( a b -- b a )`     |
| `over`                                    | Copy the second item to the top.                                                         | `( a b -- a b a )`   |
| `peek`                                    | Print the top item without removing it.                                                  | `( -- )`             |
| `dump`                                    | Print the entire stack content.                                                          | `( -- )`             |
| `vars`                                    | Prints the current variable context.                                                     | `( -- )`             |
| `len`                                     | Pop an item (expected `str`) and push its length.                                        | `( str -- len )`     |
| `!!`                                      | Pop index `i` and string `s`, then push `s[i]`.                                          | `( s i -- s[i] )`    |
| `cond if true_block end`                  | Execute `true_block` if `cond` (popped) is truthy.                                       | `( cond -- )`        |
| `cond if true_block else else_block end`  | Execute `true_block` if true, otherwise `else_block`.                                    | `( cond -- )`        |
| `while cond do body end`                  | Repeatedly execute `cond` and, if true, execute `body`.                                  | `( -- )`             |
| `start end inc for i do body end`         | Range loop, similar to `for i in range(s, e, i)`                                         | `( s e i -- )`       |
| `break`                                   | Break from the innermost loop.                                                           | `( -- )`             |
| `func name arg1... do body end`           | Define a function `name` with args `arg1...`, e.g. `func foo x y do <body> end`.         | `( -- )`             |
| `val1... call name`                       | Push args, then call function `name`; results are on the stack.                          | `( val1... -- res )` |
| `ret`                                     | Return. Exit the current function immediately (like `break`); results are on the stack.  | `( -- )`             |
| `# comment text`                          | Everything from `#` to the end of the line is ignored.                                   | `( -- )`             |

### TODOs
<!-- MDUP:BEG (CMD:make list-todo) -->
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
<!-- MDUP:BEG (CMD:cat examples/prime.spl) -->
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
<!-- MDUP:BEG (CMD:cat examples/collatz.spl) -->
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
<!-- MDUP:BEG (CMD:cat examples/fizzbuzz.spl) -->
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
<!-- MDUP:BEG (CMD:cat examples/sin_approx.spl) -->
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
<!-- MDUP:BEG (CMD:cat examples/recursion.spl) -->
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
<!-- MDUP:BEG (CMD:cat examples/fibonacci.spl) -->
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
