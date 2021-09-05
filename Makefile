.PHONY: format lint test

format:
	pysen run format

lint:
	pysen run lint

test:
	pytest baseline/test