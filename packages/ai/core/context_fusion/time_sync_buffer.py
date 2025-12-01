"""
TimeSyncBuffer
==============
FR2(Visual/Audio/Pose) 센서에서 들어오는 타임스탬프가 다른 메시지들을
ROS의 ApproximateTimeSynchronizer 방식으로 정렬해 줍니다.

FR3(GRU Predictor)는 이 버퍼를 통해 항상 동기화된 멀티모달 컨텍스트를 전달받습니다.
"""

from __future__ import annotations

from collections import deque
from typing import Callable, Dict, Iterable, Optional


class TimeSyncBuffer:
    def __init__(
        self,
        required_sensors: Iterable[str],
        queue_size: int = 10,
        slop: float = 0.5,
        on_sync: Optional[Callable[[Dict[str, object], float], None]] = None,
    ):
        self.queues = {
            sensor: deque(maxlen=queue_size) for sensor in required_sensors
        }
        self.slop = slop
        self.on_sync = on_sync
        self.queue_size = queue_size
        self.dropped = 0

    def push(self, sensor_type: str, timestamp: float, data):
        if sensor_type not in self.queues:
            return
        self.queues[sensor_type].append({"timestamp": timestamp, "data": data})
        self._try_synchronize()

    def _try_synchronize(self):
        if not all(len(q) > 0 for q in self.queues.values()):
            return

        pivot = max(q[0]["timestamp"] for q in self.queues.values())

        candidates = {}
        for sensor_type, queue in self.queues.items():
            best_msg = min(queue, key=lambda msg: abs(msg["timestamp"] - pivot))
            candidates[sensor_type] = best_msg

        timestamps = [msg["timestamp"] for msg in candidates.values()]
        span = max(timestamps) - min(timestamps)

        if span <= self.slop:
            sensor_data = {
                sensor: msg["data"] for sensor, msg in candidates.items()
            }
            avg_timestamp = sum(timestamps) / len(timestamps)

            if self.on_sync:
                self.on_sync(sensor_data, avg_timestamp)

            for sensor_type, matched in candidates.items():
                queue = self.queues[sensor_type]
                self.queues[sensor_type] = deque(
                    (msg for msg in queue if msg["timestamp"] != matched["timestamp"]),
                    maxlen=self.queue_size,
                )
        else:
            # drop the oldest sample to keep queues moving
            oldest_sensor, queue = min(
                self.queues.items(),
                key=lambda item: item[1][0]["timestamp"]
                if len(item[1]) > 0
                else float("inf"),
            )
            if len(queue) > 0:
                queue.popleft()
                self.dropped += 1
