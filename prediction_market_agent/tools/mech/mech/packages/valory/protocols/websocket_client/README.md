# Websocket Client Protocol

## Description

This is a protocol for communicating with websocket servers.

## Specification

```yaml
---
name: websocket_client
author: valory
version: 0.1.0
description: A protocol for websocket client.
license: Apache-2.0
aea_version: '>=1.0.0, <2.0.0'
protocol_specification_id: valory/websocket_client:1.0.0
speech_acts:
  subscribe:
    url: pt:str
    subscription_id: pt:str
    subscription_payload: pt:optional[pt:str]
  subscription:
    alive: pt:bool
    subscription_id: pt:str
  check_subscription:
    alive: pt:bool
    subscription_id: pt:str
  send:
    payload: pt:str
    subscription_id: pt:str
  send_success:
    send_length: pt:int
    subscription_id: pt:str
  recv:
    data: pt:str
    subscription_id: pt:str
  error:
    alive: pt:bool
    message: pt:str
    subscription_id: pt:str
...
---
initiation: [subscribe,check_subscription,send]
reply:
  subscribe: [subscription, recv, error]
  subscription: []
  check_subscription: [subscription, error]
  send: [send_success, recv, error]
  send_success: []
  recv: []
  error: []
termination: [recv,send_success,subscription,error]
roles: {skill, connection}
end_states: [successful]
keep_terminal_state_dialogues: false
...
```

## Links

