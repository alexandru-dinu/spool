"""
Simple stack-based PL.
"""

from argparse import ArgumentParser
from collections.abc import Generator
from dataclasses import dataclass
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
    "if",
    "len",
    "peek",
    "pop",
    "round",
    "swap",
    "vars",
    "while",
]
BINOPS = {
    "+": lambda a, b: a + b,
    "-": lambda a, b: a - b,
    "*": lambda a, b: a * b,
    "/": lambda a, b: a / b,
    "//": lambda a, b: a // b,
    "%": lambda a, b: a % b,
    "**": lambda a, b: a**b,
    "==": lambda a, b: a == b,
    "<": lambda a, b: a < b,
    "<=": lambda a, b: a <= b,
    ">": lambda a, b: a > b,
    ">=": lambda a, b: a >= b,
    "&&": lambda a, b: a and b,
    "||": lambda a, b: a or b,
}
RESERVED = set(KEYWORDS) | set(BINOPS) | {"!!"}


class SpoolSyntaxError(Exception):
    pass


class SpoolStackError(Exception):
    pass


class SpoolVarsError(Exception):
    pass


def try_numeric(x) -> Value | None:
    try:
        f = float(x)
        i = int(f)
        return i if i == f else f
    except ValueError:
        return None


def is_valid_ident(x: str) -> bool:
    return x not in RESERVED and x.isidentifier()


def is_string(x: str) -> bool:
    return x[0] == x[-1] == '"'


def requires_block(tok: str) -> bool:
    return tok in ["if", "while", "func"]


def collect_until(keyword: str, tokens: list[str], index: int) -> tuple[list[str], int]:
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
        elif requires_block(tokens[i]):
            level += 1

        block.append(tokens[i])
        i += 1

    raise SpoolSyntaxError("Missing `end` keyword.")


def split_else(tokens: list[str]) -> tuple[list[str], list[str] | None]:
    """Return block_true, block_else"""

    i = 0
    level = 1

    while i < len(tokens):
        if tokens[i] == "else":
            level -= 1
            if level == 0:
                return tokens[:i], tokens[i + 1 :]
        elif requires_block(tokens[i]):
            level += 1

        i += 1

    # no "else"
    return tokens, None


@dataclass
class Node:
    pass


@dataclass
class Block(Node):
    stmts: list[Node]


@dataclass
class Push(Node):
    value: Value


@dataclass
class BinOp(Node):
    op: str


@dataclass
class Round(Node):
    ndigits: int


@dataclass
class Get(Node):
    var: str


@dataclass
class Set(Node):
    var: str


@dataclass
class If(Node):
    true_block: Block
    else_block: Block | None


@dataclass
class While(Node):
    cond: Block
    body: Block


@dataclass
class Func(Node):
    name: str
    args: list[str]
    body: Block


@dataclass
class Call(Node):
    func: str


@dataclass
class Len(Node):
    pass


@dataclass
class Swap(Node):
    pass


@dataclass
class Dup(Node):
    pass


@dataclass
class Pop(Node):
    pass


@dataclass
class Peek(Node):
    pass


@dataclass
class Dump(Node):
    pass


@dataclass
class Vars(Node):
    pass


class SpoolTokenizer:
    def __init__(self, prog: str):
        self.prog = prog

    def tokenize(self) -> list[str]:
        return list(self.__tok())

    def __tok(self) -> Generator:
        """Tokenize the program."""
        cur = ""
        i = 0
        while i < len(self.prog):
            c = self.prog[i]
            # TODO: need `takewhile`

            match c:
                case _ if c.isspace():
                    i += 1
                    if cur:
                        yield cur
                        cur = ""
                    else:
                        continue

                # comments are inline, so when `#` is first encountered, skip until end of line `\n`
                case "#":
                    if cur:
                        yield cur
                        cur = ""
                    if (end := self.prog.find("\n", i)) != -1:
                        i = end + 1
                    else:  # if no `\n` found, then we must be at the end of prog
                        break

                case '"':
                    # TODO: should allow multi-line strings
                    end = self.prog.find('"', i + 1)  # +1 to discard finding the current `"` at `i`
                    if (end == -1) or "\n" in self.prog[i:end]:
                        raise SpoolSyntaxError("Unterminated string literal.")
                    yield self.prog[i : end + 1]  # include quotes in the token
                    i = end + 1

                case _:
                    # TODO: should we account for constructs w/o spaces, e.g. `34 35+10* peek`?
                    cur += c
                    i += 1

        # right at the end of the prog
        if cur:
            yield cur


class SpoolAST:
    def __init__(self, tokens: list[str]):
        self.root = self.parse(tokens)

    def __str__(self) -> str:
        return str(self.root)

    __repr__ = __str__

    def parse(self, tokens: list[str], pc: int = 0) -> Block:
        nodes = []

        while pc < len(tokens):
            tok = tokens[pc]

            match tok:
                case _ if is_string(tok):
                    nodes.append(Push(value=tok.strip(tok[0])))

                case _ if (num := try_numeric(tok)) is not None:
                    nodes.append(Push(num))

                case _ if tok in BINOPS.keys() | {"!!"}:
                    nodes.append(BinOp(tok))

                case "round":
                    try:
                        ndigits = int(tokens[pc + 1])
                    except ValueError as e:
                        raise SpoolSyntaxError("round: `ndigits must be an integer.") from e

                    nodes.append(Round(ndigits))
                    pc += 1

                case _set if _set.startswith("$"):
                    v = _set[1:]
                    if not v:
                        raise SpoolSyntaxError("No identifier to set. Syntax is: `$foo`.")
                    if not is_valid_ident(v):
                        raise SpoolSyntaxError(f"Invalid identifier name `{v}`.")

                    nodes.append(Set(v))

                case _get if _get.startswith("@"):
                    v = _get[1:]
                    if not v:
                        raise SpoolSyntaxError("No identifier to get. Syntax is: `@foo`.")
                    if not is_valid_ident(v):
                        raise SpoolSyntaxError(f"Invalid identifier name `{v}`.")

                    nodes.append(Get(v))

                case "if":
                    block, pc_end = collect_until(keyword="end", tokens=tokens, index=pc + 1)
                    true_block, else_block = split_else(block)
                    nodes.append(
                        If(
                            true_block=self.parse(true_block),
                            else_block=self.parse(else_block) if else_block else None,
                        )
                    )
                    pc = pc_end

                case "while":
                    # while <cond> do <body> end
                    block_cond, pc_do = collect_until(keyword="do", tokens=tokens, index=pc + 1)
                    block_body, pc_end = collect_until(keyword="end", tokens=tokens, index=pc_do + 1)
                    nodes.append(While(cond=self.parse(block_cond), body=self.parse(block_body)))
                    pc = pc_end

                case "func":
                    # func <name> <arg1>..<argN> do <body> end
                    name = tokens[pc + 1]
                    if not is_valid_ident(name):
                        raise SpoolSyntaxError(f"Invalid function name `{name}`.")

                    # collect args
                    args, pc_do = collect_until(keyword="do", tokens=tokens, index=pc + 2)
                    assert all(is_valid_ident(a) for a in args)

                    # collect body
                    body, pc_end = collect_until(keyword="end", tokens=tokens, index=pc_do + 1)

                    nodes.append(Func(name=name, args=args, body=self.parse(body)))

                    pc = pc_end

                case "call":
                    # <arg1> .. <argN> call <name>
                    name = tokens[pc + 1]
                    if not is_valid_ident(name):
                        raise SpoolSyntaxError(f"Invalid function name `{name}`.")

                    nodes.append(Call(func=name))
                    pc += 1

                # unary ops
                case "len":
                    nodes.append(Len())
                case "swap":
                    nodes.append(Swap())
                case "dup":
                    nodes.append(Dup())
                case "pop":
                    nodes.append(Pop())

                # printing
                case "peek":
                    nodes.append(Peek())
                case "dump":
                    nodes.append(Dump())
                case "vars":
                    nodes.append(Vars())

                # errors
                case other:
                    raise SpoolSyntaxError(f"Invalid token `{other}` @ index {pc}.")

            pc += 1

        return Block(nodes)


class SpoolInterpreter:
    def __init__(self, ast: SpoolAST):
        self.ast = ast
        self.stack: list[Value] = []
        self.global_vars: dict[str, Value] = {}
        self.funcs: dict[str, tuple[list[str], Block]] = {}  # name -> (args, body)

    def run(self) -> Generator:
        yield from self.__run(self.ast.root, ctx_vars=self.global_vars, pc=0)

    def __run(self, nodes: Block, ctx_vars: dict, pc: int = 0) -> Generator:
        for node in nodes.stmts:
            match node:
                case Push(val):
                    self.stack.append(val)

                case BinOp("!!"):
                    if len(self.stack) < 2:
                        raise SpoolStackError(
                            f"Insufficient values on the stack for operation `!!`. Expected >= 2, got {len(self.stack)}."
                        )
                    i, s = self.stack.pop(), self.stack.pop()
                    if not isinstance(s, str):
                        raise SpoolVarsError(f"Cannot apply `!!` on value of type {type(s)}.")
                    if not isinstance(i, int):
                        raise SpoolVarsError(f"Cannot apply `!!` with index of type {type(i)}.")
                    if not (0 <= i < len(s)):
                        raise SpoolVarsError(f"Index {i} is out of bounds for string of len {len(s)}.")
                    self.stack.append(s[i])

                case BinOp(op):
                    b, a = self.stack.pop(), self.stack.pop()
                    self.stack.append(BINOPS[op](a, b))

                case Round(ndigits):
                    if not self.stack:
                        raise SpoolStackError("Stack is empty.")
                    self.stack.append(round(self.stack.pop(), ndigits))

                case Set(var):
                    if not self.stack:
                        raise SpoolStackError("Stack is empty.")
                    ctx_vars[var] = self.stack.pop()

                case Get(var):
                    if var not in ctx_vars:
                        raise SpoolVarsError(f"Variable `{var}` is not defined.")
                    self.stack.append(ctx_vars[var])

                case If(true_block, else_block):
                    if not self.stack:
                        raise SpoolStackError("Stack is empty.")
                    if self.stack.pop():
                        yield from self.__run(true_block, ctx_vars=ctx_vars)
                    elif else_block:
                        yield from self.__run(else_block, ctx_vars=ctx_vars)

                case While(cond, body):
                    while True:
                        yield from self.__run(cond, ctx_vars=ctx_vars)
                        if not self.stack.pop():
                            break
                        yield from self.__run(body, ctx_vars=ctx_vars)

                case Func(name, args, body):
                    self.funcs[name] = (args, body)

                case Call(func):
                    if func not in self.funcs:
                        raise SpoolVarsError(f"Function `{func}` is not defined.")
                    args, func_body = self.funcs[func]
                    arity = len(args)
                    if (_n := len(self.stack)) < arity:
                        raise SpoolStackError(
                            f"Insufficient number of args for function `{func}`. Expected {arity} got {_n}."
                        )
                    # prepopulate the ctx with `arity` values from the stack into the given names
                    yield from self.__run(func_body, ctx_vars={arg: self.stack.pop() for arg in args[::-1]})

                case Len():
                    if not self.stack:
                        raise SpoolStackError("Stack is empty.")
                    if not isinstance(self.stack[-1], str):
                        raise SpoolVarsError(f"Cannot apply `len` on value of type {type(self.stack[-1])}.")
                    self.stack.append(len(self.stack.pop()))

                case Swap():
                    # [..., a, b] becomes [..., b, a]
                    if len(self.stack) < 2:
                        raise SpoolStackError(
                            f"Insufficient values on the stack for operation `swap`. Expected >= 2, got {len(self.stack)}."
                        )
                    b, a = self.stack.pop(), self.stack.pop()
                    self.stack.append(b)
                    self.stack.append(a)

                case Dup():
                    if self.stack:
                        self.stack.append(self.stack[-1])

                case Pop():
                    if self.stack:
                        self.stack.pop()

                # printing
                case Peek():
                    yield self.stack[-1] if self.stack else None

                case Dump():
                    yield self.stack[::]

                case Vars():
                    yield ctx_vars.copy()

                case other:
                    raise SpoolSyntaxError(f"Invalid AST node `{other}`.")


def spool(prog: str) -> Generator:
    """Wrapper for convenience"""
    tokens = SpoolTokenizer(prog).tokenize()
    ast = SpoolAST(tokens)
    out = SpoolInterpreter(ast).run()
    return out


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("file", type=Path)
    args = parser.parse_args()

    for o in spool(args.file.read_text()):
        print(o)
