.PHONY: setup run lint test verify

setup:
	bash .devcontainer/post-create.sh

lint:
	python -m compileall . -x ".git"

test:
	pytest -q

verify: lint test
