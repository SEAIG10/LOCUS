# LOCUS IoT Gateway

실시간 브릿지 역할(MQTT ↔ ZMQ, WebSocket ↔ ZMQ 등)을 `apps/backend`에서 분리하여 독립 서비스로 구성했습니다. 
`packages/gateway` 모듈의 브릿지 클래스를 조합해 필요한 파이프라인을 구성할 수 있습니다.

## 구성
```
apps/iot-gateway/
├── README.md
├── bridge_server.py
└── requirements.txt
```

## 실행 방법
```bash
cd apps/iot-gateway
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
python bridge_server.py --policy --audio --visual
```

위 명령은 정책 브릿지(FR3→FR5)와 Windows 오디오/비주얼 브릿지를 동시에 기동합니다. 필요에 따라 플래그를 조합하세요.
