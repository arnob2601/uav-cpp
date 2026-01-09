PYTEST_FILTER ?= test

.PHONY: help
help: ## Show this help message
	@echo "Usage: make [target]"
	@echo "Targets:"
	@echo "  help		Show this help message"
	@echo "  test		Runs tests (limit scope via PYTEST_FILTER=filter)"
	@echo "  clean	Clean build artifacts and virtual environment"


.PHONY: setup_dirs
setup_dirs:
	@uv run python -c "import os; os.makedirs('data', exist_ok=True)"

.PHONY: test clean
test: setup_dirs ## Runs tests (limit scope via PYTEST_FILTER=filter)
	@uv run pytest -vk $(PYTEST_FILTER)

clean: ## Clean build artifacts and virtual environment
ifeq ($(OS),Windows_NT)
	@powershell -Command "try { Remove-Item -Recurse -Force -ErrorAction Stop .venv, .pytest_cache, uv.lock } catch { Write-Warning 'Could not fully clean .venv or .pytest_cache (files might be in use/locked)' }"
	@powershell -Command "Get-ChildItem -Path . -Recurse -Include __pycache__, *.egg-info | Remove-Item -Recurse -Force -ErrorAction SilentlyContinue"
else
	@rm -rf .venv .pytest_cache *.egg-info uv.lock
	@find . -type d -name "__pycache__" -exec rm -rf {} +
	@find . -type d -name "*.egg-info" -exec rm -rf {} +
endif