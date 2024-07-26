# Tool Endpoint

Contains code for deployment of tools-as-endpoints, to enable the creation of a 'Marketplace of Tools'.

Each tool subdirectory has deployment code, and nevermined service creation code.

## Deployment

To deploy a tool using [Modal](https://modal.com/):

```bash
modal serve prediction_market_agent/tools/endpoints/<tool>/deployment.py
```

## Wrapping the deployment with Nevermined

```bash
python prediction_market_agent/tools/endpoints/<tool>/nevermined.py
```
