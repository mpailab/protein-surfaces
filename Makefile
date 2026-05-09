.PHONY: setup run lint test verify benchmark-gpu

setup:
	bash .devcontainer/post-create.sh

lint:
	python -m compileall . -x ".git"

test:
	pytest -q

verify: lint test

benchmark-gpu:
	bash scripts/run_gpu_benchmarks.sh
