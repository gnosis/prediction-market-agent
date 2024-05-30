#!/bin/bash
# Ideally, we should have re-deploying of pods (to pick up the latest changes) as part of the CI/CD pipeline.
# As discussed with Giacomo, DevOps will look into it, because first, they need to secure a connection between Github and GKE.
# Until that's done, this script, executed locally on our PC, will delete the pods and they will be re-created automatically by Kube, thus picking up the latest changes.
kubectl delete pod -n agents -l name=pma-agent
kubectl delete pod -n agents -l name=pma-agent-monitoring 
kubectl delete pod -n agents -l name=autonomous-trader-agent
