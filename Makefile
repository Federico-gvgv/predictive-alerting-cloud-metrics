.PHONY: install train eval test clean

install:
	pip install --upgrade pip
	pip install -r requirements.txt

data:
	python scripts/download_nab.py

train:
	python -m src.train --config configs/default.yaml

eval:
	python -m src.eval --config configs/default.yaml

test:
	python -m pytest -q tests/

clean:
	rm -rf outputs/ __pycache__ .pytest_cache
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
