.PHONY: test smoke

test:
	pytest -q

smoke:
	python scripts/smoke_train_codec.py
	python scripts/smoke_train_planner.py
