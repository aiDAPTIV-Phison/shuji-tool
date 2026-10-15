IMAGE_VERSION ?= 0.1.0

# Image coordinates. Override to push somewhere else:
#   IMAGE_REGISTRY=ghcr.io/<you> make image
IMAGE_REGISTRY ?= shuji-tool
IMAGE_NAME ?= shuji-tool
IMAGE_TAG := $(IMAGE_REGISTRY)/$(IMAGE_NAME):$(IMAGE_VERSION)

# 0: Run in local environment, 1: Run in uv environment
# Usage: RUN_IN_UV=1 make <target>
RUN_IN_UV ?= 1

ifeq ($(RUN_IN_UV),1)
PIP := uv pip
PY_RUN := uv run
else
PIP := pip
PY_RUN :=
endif

# Build stage. runtime = plain Python (default), dev = + test tooling.
# Usage: STAGE=dev make image
STAGE ?= runtime


# Makefile targets
SHELL := /bin/bash
.PHONY: help init test lint serve image push clean
.ONESHELL:

help: ## Show this help message
	@printf "Utility make commands for project development.\nTo run with specific flags or condition, set the variable and run make. E.g. \`STAGE=dev make image\`.\n"
	@printf "\n"
	@printf "\033[32mVariables:\033[0m\n"
	@printf "  RUN_IN_UV=1          Use uv run / uv pip for Python commands, default: $(RUN_IN_UV)\n"
	@printf "  STAGE=<stage>        Docker build stage: runtime | dev, default: $(STAGE)\n"
	@printf "  IMAGE_REGISTRY=<ns>  Image namespace or registry, default: $(IMAGE_REGISTRY)\n"
	@printf "  IMAGE_VERSION=<tag>  Image tag, default: $(IMAGE_VERSION)\n"
	@printf "\n"
	@printf "\033[32mTargets:\033[0m\n"
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | sort | awk 'BEGIN {FS = ":.*?## "}; {printf "\033[36m%-20s\033[0m %s\n", $$1, $$2}'

init: ## Install dependencies
	$(PIP) install --no-cache-dir -r requirements-dev.txt

test: ## Run the test suite
	PYTHONPATH=src $(PY_RUN) pytest

lint: ## Run pylint over the source tree
	$(PY_RUN) pylint src

serve: ## Run the service locally on APP_PORT (default 7766)
	PYTHONPATH=src $(PY_RUN) python src/api/app.py

image: ## Build the Docker image
	@printf "Building \033[33m%s\033[0m (stage: %s)\n" "$(IMAGE_TAG)" "$(STAGE)"
	@DOCKER_BUILDKIT=1 docker build --target $(STAGE) -t $(IMAGE_TAG) .

push: image ## Build then push the image to IMAGE_REGISTRY
	@docker push $(IMAGE_TAG)

clean: ## Remove caches and transient artifacts
	@rm -rf .pytest_cache temp_files/* logs/*.log
	@find . -type d -name __pycache__ -prune -exec rm -rf {} +
