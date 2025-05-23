.PHONY: build push tests-docker

IMAGE_NAME := ghcr.io/gnosis/prediction-market-agent:$(if $(GITHUB_SHA),$(GITHUB_SHA),test)

build:
	docker build . -t $(IMAGE_NAME)

push:
	# This should be done from Github's CD pipeline, but keeping it here for any testing purposes required in the future.
	docker push $(IMAGE_NAME)

tests-docker: build
	docker run --env-file .env --rm $(IMAGE_NAME) pytest tests
