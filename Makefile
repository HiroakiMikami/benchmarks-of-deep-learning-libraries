.PHONY: format lint test

format:
	pysen run format

lint:
	pysen run lint

test:
	PYTHONPATH=. pytest test
