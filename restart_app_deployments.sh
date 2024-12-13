#!/bin/bash
# Ideally, we should have re-deploying of pods (to pick up the latest changes) as part of the CI/CD pipeline.
# As discussed with Giacomo, DevOps will look into it, because first, they need to secure a connection between Github and GKE.
# Until that's done, this script, executed locally on our PC, will restart the deployments, thus picking up the latest changes.
kubectl rollout restart deploy pma-agent -n agents
kubectl rollout restart deploy pma-agent-monitoring  -n agents
kubectl rollout restart deploy autonomous-trader-agent -n agents
kubectl rollout restart deploy deployed-general-agent-viewer -n agents
kubectl rollout restart deploy treasury -n agents
