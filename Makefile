.PHONY: install install-gpu install-torch test lint

install:
	pip install -e .

install-gpu:
	pip install -e '.[gpu]'

install-torch:
	pip install torch --index-url https://download.pytorch.org/whl/cpu
	pip install -e '.[torch]'

test:
	pytest -q

lint:
	flake8 src tests
