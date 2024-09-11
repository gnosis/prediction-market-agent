.PHONY: build push tests

build:
	docker build . -t ghcr.io/gnosis/prediction-market-agent:test

push:
	# This should be done from Github's CD pipeline, but keeping it here for any testing purposes required in the future.
	docker push ghcr.io/gnosis/prediction-market-agent:test

tests-docker: build
	docker run --env-file .env --rm ghcr.io/gnosis/prediction-market-agent:test pytest tests
