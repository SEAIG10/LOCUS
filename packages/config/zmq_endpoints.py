"""
Shared ZeroMQ endpoint definitions for LOCUS modules.

Each FR module communicates over dedicated buses to decouple producers/consumers:

- SENSOR_STREAM:  FR1/FR2 → FR3 (raw sensor payloads)
- FEDERATED_STREAM: FR3 → FR4 (context windows + predictions for FL ingestion)
- POLICY_STREAM: FR3 → FR5 (predictions for policy/cleaning decisions)
"""
# WSL 내부 센서 버스 (기존 GRU가 구독하던 버스)
SENSOR_STREAM = "ipc:///tmp/locus_sensors.ipc"

# Windows에서 오는 raw 스트림
WIN_AUDIO_STREAM = "tcp://192.168.68.246:6000"
WIN_VISUAL_STREAM = "tcp://192.168.68.246:6001"

FEDERATED_STREAM = "ipc:///tmp/locus_federated.ipc"
POLICY_STREAM = "ipc:///tmp/locus_policy.ipc"
