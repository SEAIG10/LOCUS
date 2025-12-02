# LOCUS — AI-Driven Household Context Awareness

LOCUS는 스마트폰과 노트북을 로봇청소기의 온디바이스 엣지로 가정하고, 집 구조 · 시각 · 청각 · 위치 정보를 멀티모달로 수집하여 **예측형 청소**를 실현하는 프로젝트입니다. ZeroMQ 기반 비동기 파이프라인과 TimeSyncBuffer가 입력을 정렬하고, 30-타임스텝 GRU가 5~30분 후 구역별 오염 확률을 예측합니다. 모든 로직은 온디바이스에서 처리되며 로우 이미지·오디오, 개인화된 GRU 헤드는 외부로 전송하지 않습니다.

프로젝트 전반은 Flower 기반 FedPer 연합학습으로 묶여 있으며, 실험 상태는 `packages/dashboard`의 로컬 CLI 대시보드로 확인합니다.

---

## 📋 Table of Contents

- [System Architecture](#-system-architecture)
- [Module Overview](#-module-overview)
- [Functional Requirements](#-functional-requirements)
- [Repository Layout](#-repository-layout)
- [Setup & Environment](#-setup--environment)
- [Runtime Workflow](#-runtime-workflow)
- [Federated Learning Workflow](#-federated-learning-workflow)
- [Datasets & Training](#-datasets--training)
- [Dashboards & Telemetry](#-dashboards--telemetry)
- [Configuration & Messaging](#-configuration--messaging)
- [Troubleshooting](#-troubleshooting)
- [Team](#-team)

---

## 🌟 Highlights

- RoomPlan, YOLOv8n, YAMNet, 위치 센서가 ZeroMQ 스트림으로 융합되어 160차원 컨텍스트 벡터를 생성합니다.
- TimeSyncBuffer + GRU는 30 타임스텝 시퀀스로 각 zone의 오염 확률을 예측하고, Policy Engine이 즉시 청소/연기/알림을 의사결정합니다.
- FedPer 연합학습 서버/클라이언트는 Flower gRPC 채널을 통해 base layer만 집계하고, 개인화 헤드는 디바이스 내부에 유지합니다.
- FastAPI + ZeroMQ 대시보드는 FL 라운드, 타임싱크 상태, 텔레메트리를 실시간으로 모니터링합니다.

---

## 🔧 System Architecture

```
┌──────────────────────────────┐
│ 1. Home Structure & Location │─┐  smartphone RoomPlan + geolocation
└──────────────────────────────┘ │
┌──────────────────────────────┐ │ YOLO detections
│ 2. Visual Context (YOLOv8n)  │─┤
└──────────────────────────────┘ │
┌──────────────────────────────┐ │ YAMNet audio events
│ 3. Audio Context (YAMNet)    │─┘
└──────────────────────────────┘
                │  ZeroMQ streams (timestamped packets)
                ▼
┌──────────────────────────────┐
│ 4. TimeSyncBuffer            │  ±100ms window, feature fusion
└──────────────────────────────┘
                │  [30, F] sequence
                ▼
┌──────────────────────────────┐
│ 5. Sequential GRU Predictor  │  zone contamination probability
└──────────────────────────────┘
                │
      ┌─────────┴─────────┐
      ▼                   ▼
┌──────────────┐  ┌────────────────┐
│ 6. FedPer FL │  │ 7. Policy      │
└──────────────┘  │    Engine      │
                  └────────────────┘
```

### Module Overview

| # | Module | Key Responsibilities | Key Files |
|---|--------|----------------------|-----------|
| 1 | Home Structure & Location Intelligence | RoomPlan 기반 3D Mesh, 라벨링, smartphone pose를 ZeroMQ topic `locus.location`으로 송신 | `src/spatial_mapping/location_intelligence.py`, `packages/config/zones_config.json` |
| 2 | Visual Context | YOLOv8n으로 zone별 객체/행동 감지, visual embedding 생성 (실기기/녹화 영상) | `realtime/sensor_visual.py`, `src/context_fusion/visual_processor.py` |
| 3 | Audio Context | YAMNet + 17-class head로 실내 소리 인식 및 확률 벡터 송신 | `realtime/sensor_audio.py`, `src/audio_recognition/yamnet_processor.py` |
| 4 | TimeSyncBuffer & Context Encoder | 멀티모달 메시지를 ±100ms 윈도우로 정렬, AttentionContextEncoder로 160차원 벡터 생성 | `src/context_fusion/time_sync_buffer.py`, `src/context_fusion/attention_context_encoder.py`, `src/context_fusion/context_vector.py` |
| 5 | Sequential GRU + Policy | 30-step 시퀀스로 zone contamination probability 예측 후 정책 이벤트 생성 | `realtime/gru_predictor.py`, `realtime/utils.py`, `src/policy/policy_engine.py` |
| 6 | Federated Learning (FedPer) | Flower 제어 서버/클라이언트, base GRU 공유, head 로컬 유지 | `server.py`, `client.py`, `run_fl_server.py`, `run_fl_client.py` |
| 7 | Dataset Builder & Scenario Simulator | 시나리오에서 (X, y) 시퀀스 생성, synthetic 데이터로 GRU 학습 지원 | `src/dataset/dataset_builder.py`, `src/dataset/scenario_generator.py` |

---

## ✅ Functional Requirements

| 번호 | 설명 | Entry Points |
|------|------|--------------|
| FR1 | RoomPlan 기반 구조 생성 + 실시간 위치 송신 | `src/spatial_mapping/location_intelligence.py`, `packages/config/zones_config.json` |
| FR2 | YOLO 기반 Visual Context 파이프라인 | `src/context_fusion/visual_processor.py`, `realtime/sensor_visual.py` |
| FR3 | YAMNet 기반 Audio Context 파이프라인 | `src/audio_recognition/yamnet_processor.py`, `realtime/sensor_audio.py` |
| FR4 | TimeSyncBuffer (timestamp 정렬, 컨텍스트 벡터, 30-step 시퀀스) | `src/context_fusion/time_sync_buffer.py`, `src/context_fusion/context_types.py` |
| FR5 | Sequential GRU 예측 및 정책 후처리 | `realtime/gru_predictor.py`, `src/policy/policy_engine.py` |
| FR6 | Federated Learning (Base 공유, Head 로컬) | `server.py`, `client.py`, `run_fl_server.py`, `run_fl_client.py` |
| FR7 | Central Policy Engine + Dashboard Bridge | `src/policy/policy_engine.py`, `src/context_fusion/dashboard_bridge.py` |

### Non-Functional Goals

- **Privacy-first**: 로우 이미지/오디오 및 GRU Head 파라미터는 디바이스 내에만 저장
- **Low-latency on-device processing**: Raspberry Pi 5 + 노트북 조합에서 실시간 동작
- **Robust time alignment**: TimeSyncBuffer의 ±100ms 매칭과 최근값 보간
- **Edge-grade deployment**: YOLOv8n/YAMNet TFLite, ZeroMQ 메시징, Flower gRPC FedPer

---

## 🗂 Repository Layout

```
.
├── README.md
├── config.py                 # 글로벌 상수 (Flower, ZMQ, GRU 설정)
├── config/                   # zone 정의 및 추가 JSON 설정
├── client.py / server.py     # Flower FedPer 핵심 로직
├── run_fl_client.py / run_fl_server.py  # CLI 엔트리포인트
├── realtime/                 # FR3 → FR4 ZeroMQ ingest 도구
├── src/
│   ├── spatial_mapping/      # FR1: RoomPlan & 위치 인텔리전스
│   ├── context_fusion/       # TimeSync, encoders, policy bridge
│   ├── audio_recognition/    # YAMNet 기반 오디오 파이프라인
│   └── dataset/              # Scenario → (X, y) 빌더
├── tests/                    # (기타) 유닛/통합 테스트
├── data/, results/, runs/            # 데이터·모델·실험 산출물
└── requirements.txt
```

> **Pretrained GRU**: `config.PRETRAINED_MODEL_PATH`는 리포지토리 바깥 sibling 디렉터리 `../ai/models/gru/gru_model.keras`를 가리킵니다. 새로운 모델을 학습했다면 동일 경로에 덮어쓰면 됩니다.

---

## ⚙️ Setup & Environment

1. **Python 환경**
   ```bash
   cd packages/federated
   python3 -m venv venv
   source venv/bin/activate
   pip install --upgrade pip
   pip install -r requirements.txt
   ```
2. **Flower 서버 주소**
   - `config.py`의 `FLOWER_SERVER_ADDRESS`(기본: `0.0.0.0:8080`)를 환경에 맞게 조정하세요.
3. **ZeroMQ IPC 권한**
   - 기본 IPC 경로는 `/tmp/locus.*`입니다. 필요 시 `config.ZMQ_ENDPOINTS`로 수정하세요.
4. **Pretrained GRU 확인**
   ```bash
   ls ../ai/models/gru/gru_model.keras
   ```
   없을 경우 별도 학습 산출물을 복사하거나 팀에서 공유받으세요.

---

## 🛠 Runtime Workflow

### 1. Location / Visual / Audio Publishers

- **Location Intelligence**: RoomPlan 라벨이 준비되었다면,
  ```bash
  PYTHONPATH=. python -m src.spatial_mapping.location_intelligence \
    --labels data/roomplan_labels.json
  ```
  zone 라벨은 `packages/config/zones_config.json`과 동기화해 주세요.
- **모바일/엣지 센서**: `apps/mobile-tracker` 또는 `apps/tracker-expo`를 실행해 위치/방문 로그를 WebSocket → ZeroMQ로 전달할 수 있습니다.
- **Sensor scripts**: 실제 센서가 준비되지 않았거나 빠르게 테스트하고 싶다면 `realtime` 스크립트로 모의 데이터를 생성할 수 있습니다.
  ```bash
  python realtime/launcher.py
  ```
  위 명령은 GRU Predictor, Visual/Audio/Context 센서를 순차 실행합니다. 개별로 실행하려면:
  ```bash
  python realtime/gru_predictor.py
  python realtime/sensor_visual.py --interval 1.0
  python realtime/sensor_audio.py --interval 1.0 --duration 0.975
  python realtime/sensor_context.py --interval 1.0 --zone living_room
  ```
  각 스크립트는 `ipc:///tmp/locus_sensors.ipc`로 PUB/SUB 통신을 수행합니다.

### 2. TimeSyncBuffer + Context Emission

ZeroMQ 토픽(`locus.location`, `locus.visual`, `locus.audio`)이 활성화되었다면 TimeSyncBuffer를 실행합니다.
```bash
PYTHONPATH=. python -m src.context_fusion.time_sync_buffer
```
해당 모듈은 `config.ZMQ_ENDPOINTS`에 따라 컨텍스트 시퀀스를 `ipc:///tmp/locus.context`로 퍼블리시합니다.

### 3. GRU Inference + Policy

- 실시간 데모: `realtime/gru_predictor.py`가 30개의 타임스텝을 모으면 `../ai/models/gru/gru_model.keras`를 가져와 예측하고 `realtime/utils.py`로 결과를 시각화합니다.
- 정책 후처리:
  ```bash
  PYTHONPATH=. python -m src.policy.policy_engine
  PYTHONPATH=. python -m src.context_fusion.dashboard_bridge
  ```
  Policy Engine은 contamination probability를 즉시 청소/알림 이벤트로 변환하고, Dashboard Bridge가 ZMQ/HTTP/gRPC 등 외부 시스템으로 전달합니다.

---

## 🤝 Federated Learning Workflow

1. **Server (Flower FedAvg)**  
   ```bash
   python -m run_fl_server \
     --server-address 0.0.0.0:8080 \
     --rounds 3 \
     --clients-per-round 1 \
     --model-path ../ai/models/gru/gru_model.keras
   ```
   - 글로벌 가중치는 `results/fl_global/round_<n>.keras`로 저장됩니다.
   - `LocusFedAvg` 전략은 base GRU layer만 집계하고, Flower gRPC 채널로 새 round 파라미터를 브로드캐스트합니다.

2. **Clients (각 현장 디바이스)**  
   ```bash
   python -m run_fl_client \
     --server-address 127.0.0.1:8080 \
     --client-id home_001 \
     --dataset-path ../ai/data/training_dataset.npz \
     --model-path ../ai/models/gru/gru_model.keras
   ```
   - `client.py`는 `.npz` 데이터셋을 로드해 `LOCAL_EPOCHS`, `LOCAL_BATCH_SIZE`, `LR`에 따라 파인튜닝합니다.
   - 학습 후 base layer 가중치만 Flower 서버로 송신하며, `results/fl_local/<client_id>_round_<n>.keras`로 로컬 스냅샷을 유지합니다.

3. **ZeroMQ Ingest (FR3 → FR4)**  
   GRU Predictor가 송신하는 실시간 컨텍스트 시퀀스를 수집하려면 아래 브리지를 실행하세요.
   ```bash
   python -m realtime.zmq_ingest --output-dir results/zmq_stream
   ```
   생성된 `.npz/.json` 파일은 FedPer 학습 샘플로 재사용하거나 Flower 클라이언트에서 바로 로드할 수 있습니다.

4. **Gateway 브릿지**  
   MQTT/WS ↔ ZMQ 브릿지는 `packages/gateway` 모듈과 `apps/iot-gateway/bridge_server.py`에서 관리합니다.
   ```bash
   python apps/iot-gateway/bridge_server.py --policy
   ```
   위 명령은 FR3 → FR5 정책 브릿지를 로컬에서 구동합니다.

## 📦 Datasets & Training

- **시나리오 기반 Synthetic 데이터**
  ```bash
  PYTHONPATH=. python -m src.dataset.dataset_builder
  ```
  위 테스트 스크립트는 시나리오를 생성해 `data/test_dataset.npz`에 저장합니다. 커스텀 데이터셋은 `DatasetBuilder.build_dataset()` / `save_dataset()`으로 생성한 뒤 `data/training_dataset.npz`로 저장하면 FL 클라이언트가 자동으로 로드합니다.

- **모델 학습**
  - 현재 리포지토리는 학습된 GRU를 외부 `../ai/models/gru/gru_model.keras`에서 로드합니다.
  - 새 모델을 학습하려면 DatasetBuilder 산출물을 사용해 별도 Keras 스크립트에서 학습 후 동일 경로에 저장하세요.

- **시뮬레이션/리플레이**
  - `scripts/reorganize_hd10k.py`, `scripts/merge_datasets.py`, `scripts/validate_labels.py`를 활용하여 실데이터와 시나리오를 정리할 수 있습니다.

---

## 📊 Dashboards & Telemetry

HTTP/ZeroMQ 기반 대시보드는 모두 제거되었으며, 개인정보 보호를 위해 **로컬 전용 CLI 대시보드**만 제공합니다.  
`packages/dashboard/README.md`를 참고하여 `python packages/dashboard/frX_dashboard.py` 명령으로 FR1~FR5 상태를 확인하세요.

---

## 🧩 Configuration & Messaging

- `config.py`
  - **ZeroMQ**: `ZMQ_ENDPOINTS`에 location/visual/audio/context/telemetry 엔드포인트가 정의되어 있습니다.
  - **Sequence/Vector**: `SEQUENCE_LENGTH=30`, `CONTEXT_DIM=160`, `TIMESYNC_WINDOW_MS=100`.
  - **Federated**: `FLOWER_SERVER_ADDRESS`, `CLIENTS_PER_ROUND`, `SERVER_ROUNDS`, `LOCAL_EPOCHS`, `LR`, `LOCAL_BATCH_SIZE`.
  - **Zones**: `ZONE_NAMES`와 `packages/config/zones_config.json`이 구역 인덱스를 공유합니다.
- **ZeroMQ Topics**
  - `locus.location`, `locus.visual`, `locus.audio`, `locus.context`, `locus.telemetry`.
- **Context Vector Schema**
  - AttentionContextEncoder output: 5 modalities × 64-head fusion → 160 dims.
  - 패킷에는 `zone_id`, `timestamp`, `latency_ms`, `vector.tobytes()`가 포함됩니다.

---

## 🧪 Simulation & Testing

- **Sensor Dry-run**: `realtime/launcher.py`를 실행하면 visual/audio/context 퍼블리셔와 GRU Predictor가 동시에 구동되어 로그를 빠르게 검증할 수 있습니다.
- **Unit-style checks**: `realtime/launcher.py`는 프로세스 상태를 모니터링하며, 예측기가 중단되면 전체를 종료하여 재현성을 확보합니다.

---

## 🚑 Troubleshooting

- **`FileNotFoundError: gru_model.keras`**  
  → `../ai/models/gru/gru_model.keras`가 존재하는지 확인하고, 새 모델을 동일 경로에 배치하세요.
- **Flower 연결 실패 (`grpc_status: UNAVAILABLE`)**  
  → `run_fl_server`가 실행 중인지 확인하고, `FLOWER_SERVER_ADDRESS`에 방화벽/포트가 허용되어 있는지 점검하세요.
- **ZeroMQ IPC Permission**  
  → `/tmp` 대신 사용자 홈 디렉터리 아래 경로를 `config.ZMQ_ENDPOINTS`에 지정하거나 `chmod`로 권한을 조정하세요.
- **Dataset 누락**  
  → `client.py`가 `data/training_dataset.npz`를 찾지 못하면 `DatasetBuilder.save_dataset()`을 실행하여 기본 세트를 생성하십시오.
---

## 👥 Team

| Name          | Organization                                  | Email |
|---------------|-----------------------------------------------|-------|
| Hanyeong Go   | Hanyang Univ. Information Systems             | lilla9907@hanyang.ac.kr |
| Junhyung Kim  | Hanyang Univ. Information Systems             | combe4259@hanyang.ac.kr |
| Dayeon Lee    | Hanyang Univ. Sports Science                  | ldy21@hanyang.ac.kr |
| Seunghwan Lee | Hanyang Univ. Information Systems             | shlee5820@hanyang.ac.kr |

필요 시 `소웨공_문서.pdf`와 본 README를 함께 참고하여 모듈 경계를 유지하면서 기능을 확장해 주세요.
