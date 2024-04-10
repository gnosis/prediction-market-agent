# TODO: This should be automated as part of CI/CD pipeline, but for now, execute it locally on the main branch.
docker build . -t ghcr.io/gnosis/pma:latest && docker push ghcr.io/gnosis/pma:latest
