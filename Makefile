.PHONY: all
all:
	make format && make lint && make typecheck && make test && make refresh-readme

.PHONY: test
test:
	@uv run pytest -vv src/test_*.py

.PHONY: lint
lint:
	@uv run ruff check src/*.py

.PHONY: typecheck
typecheck:
	@uv run mypy src/spool.py

.PHONY: format
format:
	@uv run autoflake --in-place --remove-all-unused-imports src/*.py \
		&& uv run isort src/*.py \
		&& uv run black --line-length 120 src/*.py

.PHONY: refresh-readme
refresh-readme:
	@uv tool run mdup -i README.md

.PHONY: list-todo
list-todo:
	@rg -wNI "^# TODO:\s*(.*)" src/*.py -r '- $$1'