# TODO: This should be automated as part of CI/CD pipeline, but for now, execute it locally on the main branch.
docker build . -t europe-west1-docker.pkg.dev/gnosis-ai/pma/main && docker push europe-west1-docker.pkg.dev/gnosis-ai/pma/main
