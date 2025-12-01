import React, { useEffect, useState } from "react";
import TimelineItem from "../components/TimelineItem";

interface Fr2YoloItem {
  label: string;
  confidence: number;
}

interface Fr2YamnetItem {
  label: string;
  score: number;
}

interface Fr2SyncItem {
  step: string;
  latency: number;
}

interface Fr2Snapshot {
  yolo: Fr2YoloItem[];
  yamnet: Fr2YamnetItem[];
  sync: Fr2SyncItem[];
}

const FR2ContextAwarenessPage: React.FC = () => {
  const [data, setData] = useState<Fr2Snapshot | null>(null);
  const [error, setError] = useState<string | null>(null);

  // 초기 더미 값 (fetch 실패 시 fallback)
  const fallback: Fr2Snapshot = {
    yolo: [
      { label: "person_cooking", confidence: 0.92 },
      { label: "knife", confidence: 0.88 },
      { label: "dog", confidence: 0.73 }
    ],
    yamnet: [
      { label: "Dishes", score: 0.81 },
      { label: "Speech", score: 0.63 },
      { label: "Music", score: 0.41 }
    ],
    sync: [
      { step: "t-3s", latency: 45 },
      { step: "t-2s", latency: 38 },
      { step: "t-1s", latency: 40 },
      { step: "t", latency: 42 }
    ]
  };

  useEffect(() => {
    let cancelled = false;

    const load = async () => {
      try {
        const res = await fetch("/fr2_live.json", { cache: "no-store" });
        if (!res.ok) {
          throw new Error(`status ${res.status}`);
        }
        const json = (await res.json()) as Fr2Snapshot;
        if (!cancelled) {
          setData(json);
          setError(null);
        }
      } catch (e) {
        if (!cancelled) {
          console.warn("FR2 live fetch failed, using fallback.", e);
          setData(fallback);
          setError("라이브 데이터 불러오기 실패, 샘플로 대체 중");
        }
      }
    };

    // 즉시 한 번 로드
    load();
    // 2초마다 갱신
    const id = setInterval(load, 2000);
    return () => {
      cancelled = true;
      clearInterval(id);
    };
  }, []);

  const snapshot = data ?? fallback;

  return (
    <div className="page">
      <h1>FR2 · Visual &amp; Audio Context Awareness</h1>
      <p className="page-subtitle">
        YOLO / YAMNet가 인식한 시각·청각 이벤트를 동시에 모니터링하고, TimeSyncBuffer로
        정렬되는 과정을 시각화합니다.
      </p>
      {error && <p className="page-error">{error}</p>}

      <section className="section section-flex-2">
        <div>
          <h2>YOLO Detection Stream</h2>
          <div className="panel">
            <div className="panel-header">실시간 객체 감지</div>
            <ul className="data-list">
              {snapshot.yolo.map((d) => (
                <li key={d.label}>
                  <span>{d.label}</span>
                  <span>{Math.round(d.confidence * 100)}%</span>
                </li>
              ))}
            </ul>
          </div>
        </div>
        <div>
          <h2>YAMNet Audio Classes</h2>
          <div className="panel">
            <div className="panel-header">상위 음향 이벤트</div>
            <ul className="data-list">
              {snapshot.yamnet.map((c) => (
                <li key={c.label}>
                  <span>{c.label}</span>
                  <span>{Math.round(c.score * 100)}%</span>
                </li>
              ))}
            </ul>
          </div>
        </div>
      </section>

      <section className="section">
        <h2>TimeSync &amp; GRU Pipeline</h2>
        <p className="section-description">
          ZeroMQ로 비동기 수신된 YOLO / YAMNet / 위치 스트림을 TimeSyncBuffer가 타임스탬프
          기준으로 정렬하고, 30×160 시퀀스로 GRU에 전달합니다.
        </p>
        <div className="timeline timeline-horizontal">
          {snapshot.sync.map((s, idx) => (
            <TimelineItem
              key={s.step + String(s.latency)}
              time={s.step}
              label={`Latency: ${s.latency} ms`}
              highlight={idx === snapshot.sync.length - 1}
            />
          ))}
        </div>
      </section>
    </div>
  );
};

export default FR2ContextAwarenessPage;
