.PHONY: build build-dev push tests-docker

IMAGE_NAME := ghcr.io/gnosis/prediction-market-agent:$(if $(GITHUB_SHA),$(GITHUB_SHA),test)
IMAGE_NAME_DEV := $(IMAGE_NAME)-dev

build:
	docker build . -t $(IMAGE_NAME)

build-dev:
	docker build . -t $(IMAGE_NAME_DEV) --build-arg INSTALL_DEV=true

push:
	# This should be done from Github's CD pipeline, but keeping it here for any testing purposes required in the future.
	docker push $(IMAGE_NAME)

tests-docker: build-dev
	docker run --env-file .env --rm $(IMAGE_NAME_DEV) pytest tests
