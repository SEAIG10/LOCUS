# LOCUS Unified Workspace

로봇 청소기의 온디바이스 컨텍스트 인식 · 예측형 청소 프로젝트 **LOCUS** 모듈을 하나의 코드베이스로 모았습니다.  
`packages/`에는 센서 융합 · 정책 · FedPer 연합학습과 같은 파이썬 모듈이, `apps/`에는 위치 수집 및 사용자 인터페이스용 서비스가 들어 있습니다.

## 디렉터리 레이아웃

```
LOCUS/
├── packages/
│   ├── ai/              # 멀티모달 센서 융합 + GRU 학습 & 정책 파이프라인
│   ├── config/          # 공용 설정/엔드포인트 (ZMQ, zones 등)
│   ├── federated/       # Mosquitto + FedPer 연합학습 서버/클라이언트
│   ├── dashboard/       # 로컬 전용 FR1~FR5 CLI 대시보드
│   └── gateway/         # MQTT/ZMQ/WS 브릿지 모음
└── apps/
    ├── backend/         # Fastify + Prisma 기반 LOCUS 백엔드 (MQTT/Socket 브리지)
    ├── iot-gateway/     # 실시간 IoT 브릿지를 구동하는 독립 Python 앱
    ├── mobile-tracker/  # React 모바일 위치 추적기 + WebSocket 브로드캐스터
    └── tracker-expo/    # Expo / React Native Tracker 실험 앱
```

각 서브프로젝트의 README와 스크립트는 원본 저장소의 내용을 그대로 유지했습니다. 아래 요약을 참고해 필요한 모듈을 골라 실행하세요.

## packages/

### `packages/ai`
- [README](packages/ai/README.md)에 FR1~FR4 (공간 매핑, 멀티모달 센서, GRU 학습, Policy) 워크플로가 정리되어 있습니다.
- ZeroMQ 기반 실시간 파이프라인(`realtime/`), 정책 모듈(`src/policy/`), GRU 학습 스크립트(`src/train_gru.py`)를 포함합니다.
- `models/gru/gru_model.keras`는 `packages/federated`와 공유되는 베이스 가중치 위치입니다.

### `packages/federated`
- [README](packages/federated/README.md)에 ZeroMQ → MQTT → FedPer 서버/클라이언트가 정리되어 있습니다.
- `run_fl_server.py`, `run_fl_client.py`에서 `--model-path packages/ai/models/gru/gru_model.keras`를 기본으로 사용하도록 업데이트했습니다.
- 대시보드 관련 코드는 제거되었으며, 시각화는 `packages/dashboard` CLI 스크립트를 사용합니다.

## apps/

### `apps/backend`
- Fastify + TypeScript + Prisma 백엔드. `npm install` 후 `npm run dev`로 로컬 개발 서버를 띄웁니다.
- 사용자/홈/라벨 등 REST API만 다루며, 실시간 브릿지는 `apps/iot-gateway`에서 담당합니다.

### `packages/dashboard`
- `python packages/dashboard/frX_dashboard.py` 형태의 CLI로 FR1~FR5 대시보드를 띄웁니다.
- 모든 데이터는 `packages/dashboard/data/dashboard_samples.json`에서 로컬로만 읽어와 개인정보 유출 우려가 없습니다.

### `packages/gateway`
- MQTT/ZeroMQ/WebSocket 브릿지를 한 곳에서 관리하는 모듈입니다.
- `packages/gateway/policy_bridge.py`, `audio_bridge_win.py`, `visual_bridge_win.py` 등을 다른 앱에서 import 해 사용할 수 있습니다.

### `apps/mobile-tracker`
- [README](apps/mobile-tracker/README.md)에 모바일 위치 추적 워크플로가 정리되어 있습니다.
- `npm run host`로 브라우저 기반 트래커를 실행하고, `server/` 디렉터리의 WebSocket 서버(`npm run dev`)와 연동합니다.

### `apps/tracker-expo`
- Expo 기반 실험용 모바일 앱 템플릿. `npm install` 후 `npx expo start`.
- `expo-arkit` 모듈 샘플이 포함되어 있어 ARKit 확장 실험을 할 수 있습니다.

### `apps/iot-gateway`
- MQTT/WS ↔ ZeroMQ 파이프라인을 전담하는 Python 앱입니다.
- `python apps/iot-gateway/bridge_server.py --policy --audio --visual` 형태로 필요한 브릿지를 선택 실행합니다.

## 통합 개발 플로우

1. **센서/정책 (`packages/ai`)**  
   - ZeroMQ 스트림을 발행하도록 `realtime/` 센서 스크립트를 실행하고, `src/policy` 모듈을 통해 청소 정책을 적용합니다.
   - 학습/추론 결과는 `packages/ai/results/`, `runs/` 등에 저장됩니다.
2. **연합학습 (`packages/federated`)**  
   - 같은 Python 3.12 가상환경에서 `pip install -r requirements.txt` 후 `run_fl_server.py`, `run_fl_client.py`를 실행합니다.  
   - 기본 모델 경로는 `packages/ai/models/gru/gru_model.keras`이며 필요 시 `--model-path`로 대체합니다.
3. **대시보드 (`packages/dashboard`)**  
   - ZeroMQ/HTTP/MQTT 없이 로컬 JSON만으로 FR1~FR5 상태를 검토합니다.

각 패키지는 독립적으로 실행되지만 동일 루트 아래 배치되어 있으므로 상대 경로(`packages/ai/models/...`)를 그대로 활용할 수 있습니다. 추가 설정이나 의존성 정보는 각 서브 디렉터리의 README를 참고하세요.
