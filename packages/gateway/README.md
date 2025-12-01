# LOCUS Gateway Module

MQTT, ZeroMQ, WebSocket, REST 등 외부 입출력 브릿지를 한 곳에서 관리합니다. 기존에 `packages/ai/realtime`에 흩어져 있던 브릿지 코드를 옮겨와 운용/배포를 단순화했습니다.

## 구조
```
packages/gateway/
├── README.md
├── audio_bridge_win.py      # Windows → ZeroMQ 오디오 브릿지
├── policy_bridge.py         # FR3 → FR5 정책 브릿지
└── visual_bridge_win.py     # Windows → ZeroMQ 비주얼 브릿지
```

각 스크립트는 독립 실행이 가능하며, `apps/iot-gateway/bridge_server.py`를 통해 조합 실행할 수도 있습니다.
