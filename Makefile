.PHONY: fetch fetch-all bpe tokenize train eval experiment pipeline models generate validate-model results-table install

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

eval:
	uv run python -m lang_regularity eval --config configs/latin_tight.yaml

experiment:
	uv run python -m lang_regularity experiment $(if $(CONFIG),--config $(CONFIG),) $(if $(FORCE),--force,)

pipeline:
	uv run python -m lang_regularity pipeline $(if $(FORCE),--force,)

models:
	uv run python -m lang_regularity models --runs-root runs

generate:
	uv run python -m lang_regularity generate

validate-model:
	uv run python -m lang_regularity validate-model --all --runs-root runs

results-table:
	uv run python scripts/build_results_table.py --runs-root runs --out analysis/results_table.csv

install:
	uv pip install -e .
