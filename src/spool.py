"""
Simple stack-based PL.
"""

# TODO: arrays
# TODO: impl rule110
# TODO: tests for expected errors
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
class WithLocation:
    filename: str
    line: int
    col: int


@dataclass
class Token(WithLocation):
    val: str


class SpoolBreak(Exception):  # noqa
    pass


class SpoolReturn(Exception):  # noqa
    pass


def try_numeric(x: str) -> int | float | None:
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
        filename=start_tok.filename,
        line=start_tok.line,
        col=start_tok.col,
        message=f"Missing `{keyword}` keyword.",
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
class Node(WithLocation):
    @classmethod
    def from_token(cls, *, token: Token, **kwargs):
        return cls(
            filename=token.filename,
            line=token.line,
            col=token.col,
            **kwargs,
        )


type Block = list[Node]


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


@dataclass
class SpoolError(Exception, WithLocation):
    message: str

    def __str__(self) -> str:
        return f"{self.filename}:{self.line}:{self.col}: {self.__class__.__name__}: {self.message}"


class SpoolSyntaxError(SpoolError):
    # TODO: from_node
    pass


class SpoolRuntimeError(SpoolError):
    def __post_init__(self):
        super().__init__(filename=self.filename, line=self.line, col=self.col, message=self.message)

    @classmethod
    def from_node(cls, *, node: Node, **kwargs):
        return cls(filename=node.filename, line=node.line, col=node.col, **kwargs)


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
                            filename=self.filename,
                            line=line,
                            col=col,
                            message="Unterminated string literal.",
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
        nodes: Block = []

        while pc < len(tokens):
            tok = tokens[pc]

            match tok.val:
                case _ if is_string(tok.val):
                    nodes.append(Push.from_token(val=tok.val.strip(tok.val[0]), token=tok))

                case _ if (num := try_numeric(tok.val)) is not None:
                    nodes.append(Push.from_token(val=num, token=tok))

                case _ if tok.val in BINOPS.keys() | {"!!"}:
                    nodes.append(BinOp.from_token(op=tok.val, token=tok))

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

                    nodes.append(Round.from_token(ndigits=ndigits, token=tok))
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
                            filename=tok.filename,
                            line=tok.line,
                            col=tok.col,
                            message=f"Invalid identifier name `{v}`.",
                        )

                    nodes.append(Set.from_token(var=v, token=tok))

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
                            filename=tok.filename,
                            line=tok.line,
                            col=tok.col,
                            message=f"Invalid identifier name `{v}`.",
                        )

                    nodes.append(Get.from_token(var=v, token=tok))

                case "if":
                    block, pc_end = collect_until(keyword="end", tokens=tokens, index=pc + 1)
                    true_block, else_block = split_else(block)
                    nodes.append(
                        If.from_token(
                            true_block=self.parse(true_block),
                            else_block=self.parse(else_block) if else_block else None,
                            token=tok,
                        )
                    )
                    pc = pc_end

                case "while":
                    # while <cond> do <body> end
                    block_cond, pc_do = collect_until(keyword="do", tokens=tokens, index=pc + 1)
                    block_body, pc_end = collect_until(keyword="end", tokens=tokens, index=pc_do + 1)
                    nodes.append(
                        While.from_token(
                            cond=self.parse(block_cond),
                            body=self.parse(block_body),
                            token=tok,
                        )
                    )
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

                    nodes.append(For.from_token(index=index_tok.val, body=self.parse(block), token=tok))

                    pc = pc_end

                case "break":
                    nodes.append(Break.from_token(token=tok))

                case "ret":
                    nodes.append(Return.from_token(token=tok))

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
                                filename=a.filename,
                                line=a.line,
                                col=a.col,
                                message=f"Invalid arg name `{a.val}`.",
                            )

                    # collect body
                    body, pc_end = collect_until(keyword="end", tokens=tokens, index=pc_do + 1)

                    nodes.append(
                        Func.from_token(
                            name=name_tok.val,
                            args=[a.val for a in args],
                            body=self.parse(body),
                            token=tok,
                        )
                    )

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

                    nodes.append(Call.from_token(func=name_tok.val, token=tok))
                    pc += 1

                # unary ops
                case "len":
                    nodes.append(Len.from_token(token=tok))
                case "swap":
                    nodes.append(Swap.from_token(token=tok))
                case "dup":
                    nodes.append(Dup.from_token(token=tok))
                case "over":
                    nodes.append(Over.from_token(token=tok))
                case "pop":
                    nodes.append(Pop.from_token(token=tok))

                # printing
                case "peek":
                    nodes.append(Peek.from_token(token=tok))
                case "dump":
                    nodes.append(Dump.from_token(token=tok))
                case "vars":
                    nodes.append(Vars.from_token(token=tok))

                # errors
                case _:
                    raise SpoolSyntaxError(
                        filename=tok.filename,
                        line=tok.line,
                        col=tok.col,
                        message=f"Invalid token {tok.val}.",
                    )

            pc += 1

        return nodes


class SpoolInterpreter:
    def __init__(self, ast: SpoolAST):
        self.ast = ast
        self.stack: list[Value] = []
        self.global_vars: dict[str, Value] = {}
        self.funcs: dict[str, tuple[list[str], Block]] = {}  # name -> (args, body)

    def run(self) -> Generator:
        yield from self.__run(self.ast.root, ctx_vars=self.global_vars, in_loop=False, in_func=False)

    def __run(self, nodes: Block, ctx_vars: dict, in_loop: bool, in_func: bool) -> Generator:
        for node in nodes:
            match node:
                case Push():
                    self.stack.append(node.val)

                case BinOp(op="!!"):
                    if len(self.stack) < 2:
                        raise SpoolRuntimeError.from_node(
                            message=f"Insufficient values on the stack for operation `!!`. Expected >= 2, got {len(self.stack)}.",
                            node=node,
                        )
                    i, s = self.stack.pop(), self.stack.pop()
                    if not isinstance(s, str):
                        raise SpoolRuntimeError.from_node(
                            message=f"Cannot apply `!!` on value of type {type(s)}.", node=node
                        )
                    if not isinstance(i, int):
                        raise SpoolRuntimeError.from_node(
                            message=f"Cannot apply `!!` with index of type {type(i)}.", node=node
                        )
                    if not (0 <= i < len(s)):
                        raise SpoolRuntimeError.from_node(
                            message=f"Index {i} is out of bounds for string of len {len(s)}.",
                            node=node,
                        )
                    self.stack.append(s[i])

                case BinOp():
                    if len(self.stack) < 2:
                        raise SpoolRuntimeError.from_node(
                            message=f"Insufficient values on the stack for operation `{node.op}`. Expected >= 2, got {len(self.stack)}.",
                            node=node,
                        )
                    b, a = self.stack.pop(), self.stack.pop()
                    self.stack.append(BINOPS[node.op](a, b))

                case Round():
                    if not self.stack:
                        raise SpoolRuntimeError.from_node(message="Stack is empty.", node=node)
                    x = self.stack.pop()
                    if not isinstance(x, (int, float)):
                        raise SpoolRuntimeError.from_node(
                            message=f"Cannot apply `round` on value of type {type(x)}.", node=node
                        )
                    self.stack.append(round(x, node.ndigits))

                case Set():
                    if not self.stack:
                        raise SpoolRuntimeError.from_node(message="Stack is empty.", node=node)
                    ctx_vars[node.var] = self.stack.pop()

                case Get():
                    if node.var not in ctx_vars:
                        raise SpoolRuntimeError.from_node(message=f"Variable `{node.var}` is not defined.", node=node)
                    self.stack.append(ctx_vars[node.var])

                case If():
                    if not self.stack:
                        raise SpoolRuntimeError.from_node(message="Stack is empty.", node=node)
                    if self.stack.pop():
                        yield from self.__run(node.true_block, ctx_vars=ctx_vars, in_loop=in_loop, in_func=in_func)
                    elif node.else_block:
                        yield from self.__run(node.else_block, ctx_vars=ctx_vars, in_loop=in_loop, in_func=in_func)

                case Break():
                    if not in_loop:
                        raise SpoolRuntimeError.from_node(message="'break' outside loop", node=node)
                    else:
                        raise SpoolBreak()

                case Return():
                    if not in_func:
                        raise SpoolRuntimeError.from_node(message="'return' outside func", node=node)
                    else:
                        raise SpoolReturn()

                case While():
                    while True:
                        yield from self.__run(node.cond, ctx_vars=ctx_vars, in_loop=in_loop, in_func=in_func)
                        if not self.stack.pop():
                            break
                        try:
                            yield from self.__run(node.body, ctx_vars=ctx_vars, in_loop=True, in_func=in_func)
                        except SpoolBreak:
                            break

                case For():
                    if (_n := len(self.stack)) < 3:
                        raise SpoolRuntimeError.from_node(
                            message=f"For expects 3 values on the stack: <start> <end> <inc>, but stack size = {_n}.",
                            node=node,
                        )
                    if not all(isinstance(x, int) for x in self.stack[-3:]):
                        raise SpoolRuntimeError.from_node(
                            message="For expects <start> <end> <inc> all of type int.", node=node
                        )

                    inc, end, start = self.stack.pop(), self.stack.pop(), self.stack.pop()
                    for i in range(start, end, inc):  # type: ignore
                        # inject index var into context
                        ctx_vars[node.index] = i
                        try:
                            yield from self.__run(node.body, ctx_vars=ctx_vars, in_loop=True, in_func=in_func)
                        except SpoolBreak:
                            break

                case Func():
                    self.funcs[node.name] = (node.args, node.body)

                case Call():
                    if node.func not in self.funcs:
                        raise SpoolRuntimeError.from_node(message=f"Function `{node.func}` is not defined.", node=node)
                    args, func_body = self.funcs[node.func]
                    arity = len(args)
                    if (_n := len(self.stack)) < arity:
                        raise SpoolRuntimeError.from_node(
                            message=f"Insufficient number of args for function `{node.func}`. Expected {arity} got {_n}.",
                            node=node,
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
                        raise SpoolRuntimeError.from_node(message="Stack is empty.", node=node)
                    x = self.stack.pop()
                    if not isinstance(x, Sized):
                        raise SpoolRuntimeError.from_node(
                            message=f"Cannot apply `len` on value of type {type(x)}.", node=node
                        )
                    self.stack.append(len(x))

                case Swap():
                    # ( a b -- b a )
                    if len(self.stack) < 2:
                        raise SpoolRuntimeError.from_node(
                            message=f"Insufficient values on the stack for operation `swap`. Expected >= 2, got {len(self.stack)}.",
                            node=node,
                        )
                    b, a = self.stack.pop(), self.stack.pop()
                    self.stack.append(b)
                    self.stack.append(a)

                case Over():
                    # ( a b -- a b a )
                    if len(self.stack) < 2:
                        raise SpoolRuntimeError.from_node(
                            message=f"Insufficient values on the stack for operation `over`. Expected >= 2, got {len(self.stack)}.",
                            node=node,
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


def handler(reraise=False):
    def wrapper(filename: str, prog: str) -> Generator:
        try:
            tokens = SpoolTokenizer(filename=filename, prog=prog).tokenize()
            ast = SpoolAST(tokens)
            yield from SpoolInterpreter(ast).run()
        except (SpoolSyntaxError, SpoolRuntimeError) as e:
            if reraise:  # for tests
                raise e
            else:
                print(e, file=sys.stderr)
                sys.exit(1)

    return wrapper


def spool_prog(prog: str) -> Generator:
    yield from handler(reraise=False)(filename="<string>", prog=prog)


def spool_file(file: Path) -> Generator:
    yield from handler(reraise=False)(filename=str(file), prog=file.read_text())


def main():
    parser = ArgumentParser()
    parser.add_argument("file", type=Path)
    args = parser.parse_args()

    for o in spool_file(args.file):
        print(o)


if __name__ == "__main__":
    main()
