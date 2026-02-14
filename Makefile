.PHONY: fetch fetch-all bpe tokenize pipeline install

fetch:
	python -m lang_regularity fetch --config configs/latin_tight.yaml

fetch-all:
	python -m lang_regularity fetch-all --config configs/latin_tight.yaml

bpe:
	python -m lang_regularity bpe --config configs/latin_tight.yaml

tokenize:
	python -m lang_regularity tokenize --config configs/latin_tight.yaml

pipeline:
	python -m lang_regularity fetch --config configs/latin_tight.yaml
	python -m lang_regularity bpe --config configs/latin_tight.yaml
	python -m lang_regularity tokenize --config configs/latin_tight.yaml

install:
	uv pip install -e .
