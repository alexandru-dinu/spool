.PHONY: test
test:
	uv run pytest -v src/test_*.py

refresh-readme:
	uv tool run mdup -i README.md

todo:
	@rg -wN "TODO" src/spool.py
