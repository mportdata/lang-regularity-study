.PHONY: fetch fetch-all install

fetch:
	python -m lang_regularity fetch --config configs/latin_tight.yaml

fetch-all:
	python -m lang_regularity fetch-all --config configs/latin_tight.yaml

install:
	uv pip install -e .
