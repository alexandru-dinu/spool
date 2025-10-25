.PHONY: test
test:
	uv run pytest -v src/test_*.py
