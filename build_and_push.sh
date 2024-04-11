#!/bin/bash
# This script builds the docker image and pushes it to the GitHub Container Registry,
# it should be done from Github's CD pipeline, but keeping it here for any testing purposes required in the future.
docker build . -t ghcr.io/gnosis/prediction-market-agent:test && docker push ghcr.io/gnosis/prediction-market-agent:test
