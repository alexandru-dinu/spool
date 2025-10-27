.PHONY: all
all:
	make format && make lint && make test && make refresh-readme

.PHONY: test
test:
	@uv run pytest -v src/test_*.py

.PHONY: lint
lint:
	@uv run ruff check src/*.py

.PHONY: format
format:
	@uv tool run autoflake --in-place --remove-all-unused-imports src/*.py \
		&& uv tool run isort src/*.py \
		&& uv tool run black --line-length 120 src/*.py

refresh-readme:
	@uv tool run mdup -i README.md
