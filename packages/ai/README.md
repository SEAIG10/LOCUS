# LOCUS - Personalized Robot Cleaning System

FedPer 기반 개인화 로봇 청소 시스템 (실기기 멀티모달 센서 + 예측형 정책)

https://four-starflower-749.notion.site/LOCUS-AI-Driven-Household-Context-Awareness-for-Predictive-Cleaning-2b139484d2c0806592aaf4e31005259c?source=copy_link

## Project Structure

```
packages/ai/
├── core/                        # 공용 모듈 (context_fusion, dataset, model, policy 등)
├── data/                        # 학습 데이터셋 및 컨텍스트 DB
├── models/                      # 학습된 GRU/Encoder 가중치
├── realtime/                    # ZeroMQ 센서 퍼블리셔 + GRU 추론 + 정책 실행
├── training/                    # 학습 파이프라인 (prepare_data/train_*)
├── results/, runs/              # 실험 로그
├── scripts/                     # 데이터/텔레메트리 스크립트
└── test_tracker.py
```

## Functional Requirements

- **FR1**: Semantic Spatial Mapping (의미론적 공간 매핑)
- **FR2**: Multimodal Context Awareness (멀티모달 컨텍스트 인식)
- **FR3**: Sequential Pattern Learning (GRU 기반 청소 필요 예측)
- **FR4**: Personalized Federated Learning (FedPer 연합학습)

## Quick Start

```bash
# 1. 가상환경 활성화
source venv/bin/activate

# 2. GRU 모델 학습
python src/train_gru.py

# 3. ZeroMQ 실시간 파이프라인 실행
python realtime/launcher.py

# (선택) 개별 센서/정책 실행
python realtime/sensor_visual.py
python realtime/sensor_audio.py
python realtime/sensor_context.py
python realtime/gru_predictor.py          # FR3 (예측/발행)
python packages/gateway/policy_bridge.py  # FR5 (정책/청소 실행)
```

## ZeroMQ Buses

| Link (FR ↔ FR) | Endpoint | Publisher | Subscriber |
|----------------|----------|-----------|------------|
| FR1/FR2 → FR3 (센서) | `ipc:///tmp/locus_sensors.ipc` | `realtime/sensor_context.py`, `sensor_visual.py`, `sensor_audio.py` | `realtime/gru_predictor.py` (내부 `TimeSyncBuffer`) |
| FR3 → FR4 (연합학습) | `ipc:///tmp/locus_federated.ipc` | `realtime/gru_predictor.py` | `packages/federated/realtime/zmq_ingest.py` or 맞춤형 FedPer 브리지 |
| FR3 → FR5 (정책) | `ipc:///tmp/locus_policy.ipc` | `realtime/gru_predictor.py` | `packages/gateway/policy_bridge.py` |

`src/context_fusion/time_sync_buffer.py`는 YOLO · YAMNet · Pose류 메시지의 타임스탬프를 ±0.5초 오차로 정렬해 GRU에 항상 동기화된 시퀀스를 공급합니다.

## Model Architecture

**FedPer GRU Model**:
- Base Layer (공유): GRU(64) → GRU(32) [42.8K params]
- Head Layer (개인화): Dense(16) → Dense(7) [0.6K params]
- Input: (30, 108) - 30 timesteps of 108-dim context vectors
- Output: (7,) - Pollution probability for 7 semantic zones

## Technologies

- Python 3.11
- TensorFlow/Keras (GRU model)
- 모바일 RoomPlan/ARKit (공간/위치)
- YOLOv8 (Object detection)
- Yamnet (Audio recognition)
- SQLite (Context database)
- ZeroMQ, MQTT

## 👥 Group Members

| Name          | Organization                                  | Email                   |
|---------------|------------------------------------------------|-------------------------|
| Hanyeong Ko  | Department of Information Systems, Hanyang University | lilla9907@hanyang.ac.kr   |
| Junhyung Kim   | Department of Information Systems, Hanyang University | combe4259@hanyang.ac.kr |
| Dayeon Lee | Department of Sports Science, Hanyang University | ldy21@hanyang.ac.kr  |
| Seunghwan Lee  | Department of Information Systems, Hanyang University | shlee5820@hanyang.ac.kr |
