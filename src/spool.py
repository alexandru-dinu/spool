"""
Simple stack-based PL.
"""

# TODO: arrays
# TODO: impl rule110
# TODO: tests for expected errors
# TODO: attach loc info to AST nodes and do standardised error reporting: `filename:line:col: message`
# TODO: base error class w/ ln,col info
# TODO: typing: value for each type, errors, ...
# TODO: tracebacks (pass context around?)
# TODO: "did you mean?" for errors
# TODO: AST node for comments?
# TODO: multi-line strings?
# TODO: account for constructs w/o spaces, e.g. `34 35+10* peek`?
# TODO: library of utils
# TODO: highlighter for vim

import sys
from argparse import ArgumentParser
from collections.abc import Callable, Generator, Sized
from dataclasses import dataclass
from pathlib import Path

type Value = int | float | str

KEYWORDS = [
    "and",
    "break",
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
    "or",
    "over",
    "peek",
    "pop",
    "ret",  # optional, more like a `break` from the function
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
    "and": lambda a, b: a and b,
    "or": lambda a, b: a or b,
}
RESERVED = set(KEYWORDS) | set(BINOPS) | {"!!"}


@dataclass
class Token:
    filename: str
    line: int
    col: int
    val: str


@dataclass
class SpoolError(Exception):
    filename: str
    line: int
    col: int
    message: str

    def __str__(self) -> str:
        return f"{self.filename}:{self.line}:{self.col}: {self.message}"


class SpoolSyntaxError(SpoolError):
    pass


# class SpoolRuntimeError(SpoolError):
class SpoolRuntimeError(Exception):
    pass


class SpoolBreak(Exception):  # noqa
    pass


class SpoolReturn(Exception):  # noqa
    pass


def try_numeric(x) -> int | float | None:
    try:
        f = float(x)
        i = int(f)
    except ValueError:
        return None
    else:
        return i if i == f else f


def is_valid_ident(x: str) -> bool:
    return x not in RESERVED and x.isidentifier()


def is_string(x: str) -> bool:
    return x[0] == x[-1] == '"'


def requires_block(tok: Token) -> bool:
    return tok.val in ["func", "if", "while", "for"]


def collect_until(keyword: str, tokens: list[Token], index: int) -> tuple[list[Token], int]:
    """Collect the block until the outermost `keyword`."""
    i = index
    block: list[Token] = []
    level = 1  # 1 b/c we're in the process of parsing a block

    while i < len(tokens):
        tok = tokens[i]
        if tok.val == keyword:
            level -= 1
            if level == 0:
                return block, i
            elif level < 0:
                raise SpoolSyntaxError(
                    filename=tok.filename,
                    line=tok.line,
                    col=tok.col,
                    message="Incorrect pairing of `end` instructions.",
                )
        elif requires_block(tok):
            level += 1

        block.append(tok)
        i += 1

    start_tok = tokens[index]
    raise SpoolSyntaxError(
        filename=start_tok.filename, line=start_tok.line, col=start_tok.col, message=f"Missing `{keyword}` keyword."
    )


def split_else(tokens: list[Token]) -> tuple[list[Token], list[Token] | None]:
    """Return block_true, block_else"""

    i = 0
    level = 1

    while i < len(tokens):
        if tokens[i].val == "else":
            level -= 1
            if level == 0:
                return tokens[:i], tokens[i + 1 :]
        elif requires_block(tokens[i]):
            level += 1

        i += 1

    # no "else"
    return tokens, None


def itakewhile(pred: Callable, xs: str) -> tuple[str, int, bool]:
    """
    Like `takewhile`, but also return the stopping index and whether end was reached.
    """
    out = ""
    for i, x in enumerate(xs):
        if pred(x):
            out += x
        else:
            return out, i, False
    return out, len(xs), True


@dataclass
class Node:
    pass


@dataclass
class Block(Node):
    stmts: list[Node]


@dataclass
class Push(Node):
    val: Value


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
class For(Node):
    index: str
    body: Block


@dataclass
class Break(Node):
    pass


@dataclass
class Func(Node):
    name: str
    args: list[str]
    body: Block


@dataclass
class Call(Node):
    func: str


@dataclass
class Return(Node):
    pass


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
class Over(Node):
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
    def __init__(self, prog: str, filename: str):
        self.prog = prog
        self.filename = filename

    def tokenize(self) -> list[Token]:
        return list(self.__tok())

    def __tok(self) -> Generator:
        """Tokenize the program.
        TODO: simplify!
        """
        line = col = 1
        i = 0
        while i < len(self.prog):
            c = self.prog[i]

            match c:
                case "\n":
                    _, offset, _ = itakewhile(lambda x: x == "\n", self.prog[i:])
                    i += offset
                    line += offset
                    col = 1

                case _ if c in " \t":
                    _, offset, _ = itakewhile(lambda x: x in " \t", self.prog[i:])
                    i += offset
                    col += offset

                # comments are inline, so when `#` is first encountered, skip until end of line `\n`
                case "#":
                    _, offset, _ = itakewhile(lambda x: x != "\n", self.prog[i:])
                    i += offset + 1
                    line += 1
                    col = 1

                case '"':
                    in_str, offset, prog_end = itakewhile(lambda x: x != '"', self.prog[i + 1 :])

                    if prog_end or "\n" in in_str:
                        raise SpoolSyntaxError(
                            filename=self.filename, line=line, col=col, message="Unterminated string literal."
                        )

                    end = i + offset + 2
                    yield Token(filename=self.filename, line=line, col=col, val=self.prog[i:end])
                    i = end
                    col += offset + 2

                case _:
                    cur, offset, _ = itakewhile(lambda x: not x.isspace() and x != "#" and x != '"', self.prog[i:])
                    i += offset
                    yield Token(filename=self.filename, line=line, col=col, val=cur)
                    col += len(cur)


class SpoolAST:
    def __init__(self, tokens: list[Token]):
        self.root = self.parse(tokens)

    def __str__(self) -> str:
        return str(self.root)

    __repr__ = __str__

    def parse(self, tokens: list[Token], pc: int = 0) -> Block:
        nodes: list[Node] = []

        while pc < len(tokens):
            tok = tokens[pc]

            match tok.val:
                case _ if is_string(tok.val):
                    nodes.append(Push(val=tok.val.strip(tok.val[0])))

                case _ if (num := try_numeric(tok.val)) is not None:
                    nodes.append(Push(num))

                case _ if tok.val in BINOPS.keys() | {"!!"}:
                    nodes.append(BinOp(tok.val))

                case "round":
                    ndigits_tok = tokens[pc + 1]
                    try:
                        ndigits = int(ndigits_tok.val)
                    except ValueError:
                        raise SpoolSyntaxError(
                            filename=ndigits_tok.filename,
                            line=ndigits_tok.line,
                            col=ndigits_tok.col,
                            message="round: `ndigits must be an integer.",
                        ) from None

                    nodes.append(Round(ndigits))
                    pc += 1

                case _set if _set.startswith("$"):
                    v = _set[1:]
                    if not v:
                        raise SpoolSyntaxError(
                            filename=tok.filename,
                            line=tok.line,
                            col=tok.col,
                            message="No identifier to set. Syntax is: `$foo`.",
                        )
                    if not is_valid_ident(v):
                        raise SpoolSyntaxError(
                            filename=tok.filename, line=tok.line, col=tok.col, message=f"Invalid identifier name `{v}`."
                        )

                    nodes.append(Set(v))

                case _get if _get.startswith("@"):
                    v = _get[1:]
                    if not v:
                        raise SpoolSyntaxError(
                            filename=tok.filename,
                            line=tok.line,
                            col=tok.col,
                            message="No identifier to get. Syntax is: `@foo`.",
                        )
                    if not is_valid_ident(v):
                        raise SpoolSyntaxError(
                            filename=tok.filename, line=tok.line, col=tok.col, message=f"Invalid identifier name `{v}`."
                        )

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

                case "for":
                    # <start> <stop> <inc> for <index> do <body> end
                    indices, pc_do = collect_until(keyword="do", tokens=tokens, index=pc + 1)
                    assert len(indices) == 1  # NOTE: temporary
                    index_tok = indices[0]
                    if not is_valid_ident(index_tok.val):
                        raise SpoolSyntaxError(
                            filename=index_tok.filename,
                            line=index_tok.line,
                            col=index_tok.col,
                            message=f"Invalid identifier name `{index_tok.val}`.",
                        )

                    block, pc_end = collect_until(keyword="end", tokens=tokens, index=pc_do + 1)

                    nodes.append(For(index=index_tok.val, body=self.parse(block)))

                    pc = pc_end

                case "break":
                    nodes.append(Break())

                case "ret":
                    nodes.append(Return())

                case "func":
                    # func <name> <arg1>..<argN> do <body> end
                    name_tok = tokens[pc + 1]
                    if not is_valid_ident(name_tok.val):
                        raise SpoolSyntaxError(
                            filename=name_tok.filename,
                            line=name_tok.line,
                            col=name_tok.col,
                            message=f"Invalid function name `{name_tok.val}`.",
                        )

                    # collect args
                    args, pc_do = collect_until(keyword="do", tokens=tokens, index=pc + 2)
                    for a in args:
                        if not is_valid_ident(a.val):
                            raise SpoolSyntaxError(
                                filename=a.filename, line=a.line, col=a.col, message=f"Invalid arg name `{a.val}`."
                            )

                    # collect body
                    body, pc_end = collect_until(keyword="end", tokens=tokens, index=pc_do + 1)

                    nodes.append(Func(name=name_tok.val, args=[a.val for a in args], body=self.parse(body)))

                    pc = pc_end

                case "call":
                    # <arg1> .. <argN> call <name>
                    name_tok = tokens[pc + 1]
                    if not is_valid_ident(name_tok.val):
                        raise SpoolSyntaxError(
                            filename=name_tok.filename,
                            line=name_tok.line,
                            col=name_tok.col,
                            message=f"Invalid function name `{name_tok.val}`.",
                        )

                    nodes.append(Call(func=name_tok.val))
                    pc += 1

                # unary ops
                case "len":
                    nodes.append(Len())
                case "swap":
                    nodes.append(Swap())
                case "dup":
                    nodes.append(Dup())
                case "over":
                    nodes.append(Over())
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
                case _:
                    raise SpoolSyntaxError(
                        filename=tok.filename,
                        line=tok.line,
                        col=tok.col,
                        message=f"Invalid token {tok.val}.",
                    )

            pc += 1

        return Block(nodes)


class SpoolInterpreter:
    def __init__(self, ast: SpoolAST):
        self.ast = ast
        self.stack: list[Value] = []
        self.global_vars: dict[str, Value] = {}
        self.funcs: dict[str, tuple[list[str], Block]] = {}  # name -> (args, body)

    def run(self) -> Generator:
        yield from self.__run(self.ast.root, ctx_vars=self.global_vars, in_loop=False, in_func=False)

    def __run(self, nodes: Block, ctx_vars: dict, in_loop: bool, in_func: bool) -> Generator:
        for node in nodes.stmts:
            match node:
                case Push(val):
                    self.stack.append(val)

                case BinOp("!!"):
                    if len(self.stack) < 2:
                        raise SpoolRuntimeError(
                            f"Insufficient values on the stack for operation `!!`. Expected >= 2, got {len(self.stack)}."
                        )
                    i, s = self.stack.pop(), self.stack.pop()
                    if not isinstance(s, str):
                        raise SpoolRuntimeError(f"Cannot apply `!!` on value of type {type(s)}.")
                    if not isinstance(i, int):
                        raise SpoolRuntimeError(f"Cannot apply `!!` with index of type {type(i)}.")
                    if not (0 <= i < len(s)):
                        raise SpoolRuntimeError(f"Index {i} is out of bounds for string of len {len(s)}.")
                    self.stack.append(s[i])

                case BinOp(op):
                    b, a = self.stack.pop(), self.stack.pop()
                    self.stack.append(BINOPS[op](a, b))

                case Round(ndigits):
                    if not self.stack:
                        raise SpoolRuntimeError("Stack is empty.")
                    x = self.stack.pop()
                    if not isinstance(x, (int, float)):
                        raise SpoolRuntimeError(f"Cannot apply `round` on value of type {type(x)}.")
                    self.stack.append(round(x, ndigits))

                case Set(var):
                    if not self.stack:
                        raise SpoolRuntimeError("Stack is empty.")
                    ctx_vars[var] = self.stack.pop()

                case Get(var):
                    if var not in ctx_vars:
                        raise SpoolRuntimeError(f"Variable `{var}` is not defined.")
                    self.stack.append(ctx_vars[var])

                case If(true_block, else_block):
                    if not self.stack:
                        raise SpoolRuntimeError("Stack is empty.")
                    if self.stack.pop():
                        yield from self.__run(true_block, ctx_vars=ctx_vars, in_loop=in_loop, in_func=in_func)
                    elif else_block:
                        yield from self.__run(else_block, ctx_vars=ctx_vars, in_loop=in_loop, in_func=in_func)

                case Break():
                    if not in_loop:
                        raise SpoolRuntimeError("'break' outside loop")
                    else:
                        raise SpoolBreak()

                case Return():
                    if not in_func:
                        raise SpoolRuntimeError("'return' outside func")
                    else:
                        raise SpoolReturn()

                case While(cond, body):
                    while True:
                        yield from self.__run(cond, ctx_vars=ctx_vars, in_loop=in_loop, in_func=in_func)
                        if not self.stack.pop():
                            break
                        try:
                            yield from self.__run(body, ctx_vars=ctx_vars, in_loop=True, in_func=in_func)
                        except SpoolBreak:
                            break

                case For(index, body):
                    if (_n := len(self.stack)) < 3:
                        raise SpoolRuntimeError(
                            f"For expects 3 values on the stack: <start> <end> <inc>, but stack size = {_n}."
                        )
                    if not all(isinstance(x, int) for x in self.stack[-3:]):
                        raise SpoolRuntimeError("For expects <start> <end> <inc> all of type int.")

                    inc, end, start = self.stack.pop(), self.stack.pop(), self.stack.pop()
                    for i in range(start, end, inc):  # type: ignore
                        # inject index var into context
                        ctx_vars[index] = i
                        try:
                            yield from self.__run(body, ctx_vars=ctx_vars, in_loop=True, in_func=in_func)
                        except SpoolBreak:
                            break

                case Func(name, args, body):
                    self.funcs[name] = (args, body)

                case Call(func):
                    if func not in self.funcs:
                        raise SpoolRuntimeError(f"Function `{func}` is not defined.")
                    args, func_body = self.funcs[func]
                    arity = len(args)
                    if (_n := len(self.stack)) < arity:
                        raise SpoolRuntimeError(
                            f"Insufficient number of args for function `{func}`. Expected {arity} got {_n}."
                        )
                    try:
                        # prepopulate the ctx with `arity` values from the stack into the given names
                        yield from self.__run(
                            func_body,
                            ctx_vars=self.global_vars | {arg: self.stack.pop() for arg in args[::-1]},
                            in_loop=in_loop,
                            in_func=True,
                        )
                    except SpoolReturn:
                        # return value is on the stack
                        continue

                case Len():
                    if not self.stack:
                        raise SpoolRuntimeError("Stack is empty.")
                    x = self.stack.pop()
                    if not isinstance(x, Sized):
                        raise SpoolRuntimeError(f"Cannot apply `len` on value of type {type(x)}.")
                    self.stack.append(len(x))

                case Swap():
                    # ( a b -- b a )
                    if len(self.stack) < 2:
                        raise SpoolRuntimeError(
                            f"Insufficient values on the stack for operation `swap`. Expected >= 2, got {len(self.stack)}."
                        )
                    b, a = self.stack.pop(), self.stack.pop()
                    self.stack.append(b)
                    self.stack.append(a)

                case Over():
                    # ( a b -- a b a )
                    if len(self.stack) < 2:
                        raise SpoolRuntimeError(
                            f"Insufficient values on the stack for operation `over`. Expected >= 2, got {len(self.stack)}."
                        )
                    self.stack.append(self.stack[-2])

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

                case _:
                    raise AssertionError("should not happen")


def spool_prog(prog: str) -> Generator:
    tokens = SpoolTokenizer(filename="<string>", prog=prog).tokenize()
    ast = SpoolAST(tokens)
    out = SpoolInterpreter(ast).run()
    return out


def spool_file(file: Path) -> Generator:
    try:
        tokens = SpoolTokenizer(filename=str(file), prog=file.read_text()).tokenize()
        ast = SpoolAST(tokens)
        out = SpoolInterpreter(ast).run()
    except (SpoolSyntaxError, SpoolRuntimeError) as e:
        print(e, file=sys.stderr)
        exit(1)

    return out


def main():
    parser = ArgumentParser()
    parser.add_argument("file", type=Path)
    args = parser.parse_args()

    for o in spool_file(args.file):
        print(o)


if __name__ == "__main__":
    main()
