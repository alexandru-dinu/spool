"""
Simple stack-based PL.
"""

import re
from argparse import ArgumentParser
from collections.abc import Generator
from pathlib import Path

type Value = float | int | str

KEYWORDS = [
    "call",
    "do",
    "dump",
    "dup",
    "else",
    "end",
    "for",
    "func",
    "get",
    "if",
    "len",
    "peek",
    "pop",
    "set",
    "vars",
    "while",
]


class SpoolSyntaxError(Exception):
    pass


class SpoolStackError(Exception):
    pass


class SpoolVarsError(Exception):
    pass


class Spool:
    def __init__(self):
        self.stack: list[Value] = []
        self.global_vars: dict[str, Value] = {}
        self.funcs: dict[str, tuple[int, list[str]]] = {}  # name -> (arity, body)

    @staticmethod
    def _try_numeric(x) -> Value | None:
        try:
            f = float(x)
            i = int(f)
            return i if i == f else f
        except ValueError:
            return None

    @staticmethod
    def _is_valid_ident(x: str) -> bool:
        return x not in KEYWORDS and re.match(r"(?=_*[a-z])[a-z_]+", x)

    @staticmethod
    def _is_string(x: str) -> bool:
        return x[0] in ('"', "'") and x[-1] == x[0]

    @staticmethod
    def _requires_block(tok: str) -> bool:
        return tok in ["if", "while", "func"]

    @staticmethod
    def _collect_until(keyword: str, tokens: list[str], index: int) -> tuple[list[str], int]:
        """Collect the block until the outermost `keyword`."""
        i = index
        block = []
        level = 1  # 1 b/c we're in the process of parsing a block

        while i < len(tokens):
            if tokens[i] == keyword:
                level -= 1
                if level == 0:
                    return block, i
                elif level < 0:
                    raise SpoolSyntaxError("Incorrect pairing of `end` instructions.")
            elif Spool._requires_block(tokens[i]):
                level += 1

            block.append(tokens[i])
            i += 1

        raise SpoolSyntaxError("Missing `end` keyword.")

    @staticmethod
    def _split_else(tokens: list[str]) -> tuple[list[str], list[str] | None]:
        """Return block_true, block_else"""

        i = 0
        level = 1

        while i < len(tokens):
            if tokens[i] == "else":
                level -= 1
                if level == 0:
                    return tokens[:i], tokens[i + 1 :]
            elif Spool._requires_block(tokens[i]):
                level += 1

            i += 1

        # no "else"
        return tokens, None

    def parse(self, prog: str) -> list[str]:
        toks = []
        for line in prog.splitlines():
            # <code>#<comment>
            toks += line.split("#", maxsplit=1)[0].strip().split()
        return toks

    def execute(self, prog: str) -> Generator:
        yield from self._run(self.parse(prog), ctx_vars=self.global_vars)

    def _run(self, tokens: list[str], ctx_vars: dict, pc: int = 0) -> Generator:
        while pc < len(tokens):
            tok = tokens[pc]

            match tok:
                case _ if Spool._is_string(tok):
                    self.stack.append(tok.strip(tok[0]))

                case _ if (num := Spool._try_numeric(tok)) is not None:
                    self.stack.append(num)

                case (
                    "+"
                    | "-"
                    | "*"
                    | "/"
                    | "//"
                    | "%"
                    | "**"
                    | "=="
                    | "<"
                    | "<="
                    | ">"
                    | ">="
                    | "&&"
                    | "||"
                ):
                    if len(self.stack) < 2:
                        raise SpoolStackError(
                            f"Insufficient values on the stack for operation `{tok}`. Expected >= 2, got {len(self.stack)}."
                        )
                    b, a = self.stack.pop(), self.stack.pop()
                    if tok == "+":
                        self.stack.append(a + b)
                    if tok == "-":
                        self.stack.append(a - b)
                    if tok == "*":
                        self.stack.append(a * b)
                    if tok == "/":
                        self.stack.append(a / b)
                    if tok == "//":
                        self.stack.append(a // b)
                    if tok == "%":
                        self.stack.append(a % b)
                    if tok == "**":
                        self.stack.append(a**b)
                    if tok == "==":
                        self.stack.append(a == b)
                    if tok == "<":
                        self.stack.append(a < b)
                    if tok == "<=":
                        self.stack.append(a <= b)
                    if tok == ">":
                        self.stack.append(a > b)
                    if tok == ">=":
                        self.stack.append(a >= b)
                    if tok == "&&":
                        self.stack.append(a and b)
                    if tok == "||":
                        self.stack.append(a or b)

                case "round":
                    if not self.stack:
                        raise SpoolStackError("Stack is empty.")

                    try:
                        ndigits = int(tokens[pc + 1])
                    except ValueError as e:
                        raise SpoolSyntaxError("round: `ndigits must be an integer.") from e

                    self.stack.append(round(self.stack.pop(), ndigits))
                    pc += 1

                case "set":
                    if not self.stack:
                        raise SpoolStackError("Stack is empty.")

                    v = tokens[pc + 1]
                    if not Spool._is_valid_ident(v):
                        raise SpoolSyntaxError(f"Invalid identifier name `{v}`.")

                    ctx_vars[v] = self.stack.pop()
                    pc += 1

                case "get":
                    v = tokens[pc + 1]
                    if not Spool._is_valid_ident(v):
                        raise SpoolSyntaxError(f"Invalid identifier name `{v}`.")
                    if v not in ctx_vars:
                        raise SpoolVarsError(f"Variable `{v}` is not defined.")

                    self.stack.append(ctx_vars[v])
                    pc += 1

                case "if":
                    if not self.stack:
                        raise SpoolStackError("Stack is empty.")

                    block, pc_end = Spool._collect_until(keyword="end", tokens=tokens, index=pc + 1)
                    block_true, block_else = Spool._split_else(block)

                    # condition is the last thing on the stack
                    if self.stack.pop():
                        yield from self._run(block_true, ctx_vars=ctx_vars)
                    elif block_else:
                        yield from self._run(block_else, ctx_vars=ctx_vars)

                    pc = pc_end

                case "while":
                    # while <cond> do <body> end
                    block_cond, pc_do = Spool._collect_until(
                        keyword="do", tokens=tokens, index=pc + 1
                    )
                    block_body, pc_end = Spool._collect_until(
                        keyword="end", tokens=tokens, index=pc_do + 1
                    )

                    while True:
                        yield from self._run(block_cond, ctx_vars)
                        if not self.stack.pop():
                            break
                        yield from self._run(block_body, ctx_vars)

                    pc = pc_end

                case "func":
                    # func <name> <arity> <body> end
                    name = tokens[pc + 1]
                    if not Spool._is_valid_ident(name):
                        raise SpoolSyntaxError(f"Invalid function name `{name}`.")

                    try:
                        arity = int(tokens[pc + 2])
                        assert arity >= 0
                    except (ValueError, AssertionError) as e:
                        raise SpoolSyntaxError("Function arity must be an integer >= 0.") from e

                    body, pc_end = Spool._collect_until(keyword="end", tokens=tokens, index=pc + 3)
                    self.funcs[name] = (arity, body)
                    pc = pc_end

                case "call":
                    # <arg1> .. <argN> call <name>
                    name = tokens[pc + 1]
                    if not Spool._is_valid_ident(name):
                        raise SpoolSyntaxError(f"Invalid function name `{name}`.")
                    if name not in self.funcs:
                        raise SpoolVarsError(f"Function `{name}` is not defined.")

                    arity, func_body = self.funcs[name]
                    if (_n := len(self.stack)) < arity:
                        raise SpoolStackError(
                            f"Insufficient number of args for function `{name}`. Expected {arity} got {_n}."
                        )

                    yield from self._run(func_body, ctx_vars={})
                    pc += 1

                # strings ops
                case "len":
                    if not self.stack:
                        raise SpoolStackError("Stack is empty.")
                    if not isinstance(self.stack[-1], str):
                        raise SpoolVarsError(
                            f"Cannot apply `len` on value of type {type(self.stack[-1])}."
                        )
                    self.stack.append(len(self.stack.pop()))

                case "!!":
                    if len(self.stack) < 2:
                        raise SpoolStackError(
                            f"Insufficient values on the stack for operation `{tok}`. Expected >= 2, got {len(self.stack)}."
                        )
                    i, s = self.stack.pop(), self.stack.pop()
                    if not isinstance(s, str):
                        raise SpoolVarsError(f"Cannot apply `!!` on value of type {type(s)}.")
                    if not isinstance(i, int):
                        raise SpoolVarsError(f"Cannot apply `!!` with index of type {type(i)}.")
                    if not (0 <= i < len(s)):
                        raise SpoolVarsError(
                            f"Index {i} is out of bounds for string of len {len(s)}."
                        )
                    self.stack.append(s[i])
                # //

                case "swap":
                    # [..., a, b] becomes [..., b, a]
                    if len(self.stack) < 2:
                        raise SpoolStackError(
                            f"Insufficient values on the stack for operation `{tok}`. Expected >= 2, got {len(self.stack)}."
                        )
                    b, a = self.stack.pop(), self.stack.pop()
                    self.stack.append(b)
                    self.stack.append(a)

                case "dup":  # no-op if stack is empty
                    if self.stack:
                        self.stack.append(self.stack[-1])

                case "pop":
                    if self.stack:
                        self.stack.pop()

                # printing
                case "peek":
                    yield self.stack[-1] if self.stack else None

                case "dump":
                    yield self.stack[::]

                case "vars":
                    yield ctx_vars.copy()

                # errors
                case other:
                    raise SpoolSyntaxError(f"Invalid token `{other}` @ index {pc}.")

            pc += 1


# TODO: func args & arity: having #(set) == func's arity
# TODO: lists
# TODO: errors (... @ index ...)
# TODO: tracebacks (pass context around?)
# TODO: did you mean for errors
# TODO: write highlighter for vim
# TODO: library of utils
# TODO: impl rule110

if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("file", type=Path)
    args = parser.parse_args()

    out = Spool().execute(args.file.read_text())
    for o in out:
        print(o)
