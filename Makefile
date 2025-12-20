PYTEST_FILTER ?= test


.PHONY: test

test:  ## Runs tests (limit scope via PYTEST_FILTER=filter)
	@uv run pytest -vk $(PYTEST_FILTER)