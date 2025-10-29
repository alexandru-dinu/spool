#!/usr/bin/env zsh

SPL_SRC="./src/spool.py"
SPL_FILE="$1"

touch $SPL_FILE

fswatch -o $SPL_FILE | while read num; do
    uv run $SPL_SRC $SPL_FILE
    perl -e 'print "-" x 80, "\n";'
done
