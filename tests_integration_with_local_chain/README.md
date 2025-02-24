# Integration tests

These tests use a local chain (powered by the ![ape-test](https://docs.apeworx.io/ape/stable/userguides/testing.html) pytest plugin) to enable the accurate testing of functions that execute transactions on the blockchain.

Note that the plugin, regardless of test logic, connects to the RPC of the default ecosystem provider. Hence, tests which do not require a local chain should not be pasted in this folder.

Additionally, note the pytest scope of the fixture `local_web3` for fork duration.

## Requirements

[Anvil](https://www.alchemy.com/dapps/foundry-anvil) is used for the local testnet node. It is installed as part of `Foundry`.

Installation instructions from [here](https://book.getfoundry.sh/getting-started/installation):

```bash
# Download foundry installer
curl -L https://foundry.paradigm.xyz | bash 

# Follow instructions on-screen, and run the installer
foundryup
```
