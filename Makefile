.PHONY: install sync test lint format train predict demo

install:
	uv sync

sync:
	uv sync --upgrade

test:
	uv run pytest

lint:
	uv run ruff check src/ tests/

format:
	uv run ruff format src/ tests/

train:
	uv run python -m src.training.trainers.neuro_symbolic_trainer

predict:
	uv run python -m src.inference.predictors.realtime_predictor

demo:
	uv run streamlit run app.py