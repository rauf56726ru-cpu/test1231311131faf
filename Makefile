PYTHON?=python3
PIP?=$(PYTHON) -m pip
UVICORN?=$(PYTHON) -m uvicorn
SYMBOL?=BTCUSDT
TF?=1m
CFG?=configs/backtest.yaml
NOTION_PATH?=./notion_export

venv:
	$(PYTHON) -m venv .venv

install:
	$(PIP) install -r requirements.txt

collect:
	$(PYTHON) -m src.cli collect --symbol $(SYMBOL) --interval $(TF)

build_features:
	$(PYTHON) -m src.cli build-features --config configs/backtest.yaml

train_xgb:
	$(PYTHON) -m src.cli train-xgb

train_tcn:
	$(PYTHON) -m src.cli train-tcn

train_all:
	$(PYTHON) -m src.cli train-all --config configs/backtest.yaml

backtest:
	$(PYTHON) -m src.cli backtest --config $(CFG)

api:
	$(UVICORN) src.api.app:app --reload --host 0.0.0.0 --port 8000

paper:
	$(PYTHON) -m src.cli paper

ingest_notion:
	$(PYTHON) -m src.cli ingest-notion --path $(NOTION_PATH)

test:
	$(PYTHON) -m pytest

.PHONY: venv install collect build_features train_xgb train_tcn train_all backtest api paper ingest_notion test
