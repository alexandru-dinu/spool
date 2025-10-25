# Spool

Simple stack-based programming language.

## Usage
```sh
uv run src/spool.py spool/collatz.spl
```

## Examples:
**Collatz sequence**
```
func collatz_once 1
    dup set x
    2 % 0 == if
        get x 2 /
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

**FizzBuzz**
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
