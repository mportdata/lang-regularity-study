.PHONY: fetch fetch-all bpe tokenize train pipeline install

fetch:
	uv run python -m lang_regularity fetch --config configs/latin_tight.yaml

fetch-all:
	uv run python -m lang_regularity fetch-all --config configs/latin_tight.yaml

bpe:
	uv run python -m lang_regularity bpe --config configs/latin_tight.yaml

tokenize:
	uv run python -m lang_regularity tokenize --config configs/latin_tight.yaml

train:
	uv run python -m lang_regularity train --config configs/latin_tight.yaml

pipeline:
	uv run python -m lang_regularity pipeline --config configs/latin_tight.yaml $(if $(FORCE),--force,)

install:
	uv pip install -e .
