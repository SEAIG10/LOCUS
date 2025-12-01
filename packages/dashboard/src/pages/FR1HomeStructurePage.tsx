import React, { useEffect, useState } from "react";
import StatCard from "../components/StatCard";
import TimelineItem from "../components/TimelineItem";

interface Fr1Zone {
  name: string;
  occupancy: string;
  humidity?: string;
  lighting?: string;
  last_seen?: string;
}

interface Fr1Tracker {
  status: string;
  device: string;
  zone?: string;
  x?: number;
  y?: number;
  last_seen?: string;
  latency_ms?: number;
}

interface Fr1Event {
  time: string;
  label: string;
}

interface Fr1Snapshot {
  zones: Fr1Zone[];
  tracker: Fr1Tracker;
  events: Fr1Event[];
}

const FR1HomeStructurePage: React.FC = () => {
  const [data, setData] = useState<Fr1Snapshot | null>(null);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    let cancelled = false;

    const load = async () => {
      try {
        const res = await fetch("/fr1_live.json", { cache: "no-store" });
        if (!res.ok) throw new Error(`status ${res.status}`);
        const json = (await res.json()) as Fr1Snapshot;
        if (!cancelled) {
          setData(json);
          setError(null);
        }
      } catch (e) {
        if (!cancelled) {
          console.warn("FR1 live fetch failed", e);
          setError("라이브 데이터 불러오기 실패, 마지막 스냅샷 사용 중");
        }
      }
    };

    load();
    const id = setInterval(load, 2000);
    return () => {
      cancelled = true;
      clearInterval(id);
    };
  }, []);

  if (!data) {
    return <p>FR1 라이브 데이터 로딩 중...</p>;
  }

  const tracker = data.tracker;
  const zones = data.zones;
  const events = data.events;

  return (
    <div className="page">
      <h1>FR1 · Home Structure &amp; Location Intelligence</h1>
      <p className="page-subtitle">
        RoomPlan / 모바일 트래커로 수집한 집 구조와 실시간 위치 정보를 요약해서 보여줍니다.
      </p>
      {error && <p className="page-error">{error}</p>}

      <section className="section-grid section-grid--3">
        <StatCard
          label="현재 위치 존"
          value={tracker.zone ?? "-"}
          sub={`마지막 업데이트: ${tracker.last_seen ?? "-"}`}
        />
        <StatCard
          label="트래커 상태"
          value={tracker.status ?? "-"}
          sub={tracker.device ?? ""}
        />
        <StatCard
          label="위치 업데이트 지연"
          value={tracker.latency_ms != null ? `${tracker.latency_ms} ms` : "-"}
          sub="Gateway · ZeroMQ hop 포함"
        />
      </section>

      <section className="section">
        <h2>Zones</h2>
        <table className="table">
          <thead>
            <tr>
              <th>Zone</th>
              <th>Occupancy</th>
              <th>Humidity</th>
              <th>Lighting</th>
              <th>Last seen</th>
            </tr>
          </thead>
          <tbody>
            {zones.map((z) => (
              <tr key={z.name}>
                <td>{z.name}</td>
                <td>{z.occupancy}</td>
                <td>{z.humidity ?? "-"}</td>
                <td>{z.lighting ?? "-"}</td>
                <td>{z.last_seen ?? "-"}</td>
              </tr>
            ))}
          </tbody>
        </table>
      </section>

      <section className="section section-flex-2">
        <div>
          <h2>Mobile Tracker</h2>
          <ul className="info-list">
            <li>
              <span className="info-label">상태</span>
              <span>{tracker.status ?? "-"}</span>
            </li>
            <li>
              <span className="info-label">디바이스</span>
              <span>{tracker.device ?? "-"}</span>
            </li>
            <li>
              <span className="info-label">현재 존</span>
              <span>{tracker.zone ?? "-"}</span>
            </li>
            <li>
              <span className="info-label">마지막 업데이트</span>
              <span>{tracker.last_seen ?? "-"}</span>
            </li>
            <li>
              <span className="info-label">지연</span>
              <span>
                {tracker.latency_ms != null ? `${tracker.latency_ms} ms` : "-"}
              </span>
            </li>
          </ul>
        </div>
        <div>
          <h2>Recent Events</h2>
          <div className="timeline">
            {events.map((e) => (
              <TimelineItem
                key={e.time + e.label}
                time={e.time}
                label={e.label}
              />
            ))}
          </div>
        </div>
      </section>
    </div>
  );
};

export default FR1HomeStructurePage;
