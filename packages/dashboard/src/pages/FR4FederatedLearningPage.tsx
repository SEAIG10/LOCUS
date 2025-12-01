// src/pages/FR4FederatedLearningPage.tsx
import React, { useEffect, useState } from "react";
import TimelineItem from "../components/TimelineItem";
import StatCard from "../components/StatCard";

interface Fr4Summary {
  global_round: number;
  total_clients: number;
  online_clients: number;
  avg_loss: number | null;
  last_updated: string;
}

interface Fr4Client {
  id: string;
  name: string;
  status: string;
  latency_ms: number | null;
  loss: number | null;
  rounds: number | null;
}

interface Fr4Event {
  time: string;
  source: string;
  event: string;
}

interface Fr4Snapshot {
  summary: Fr4Summary;
  clients: Fr4Client[];
  flow: string[];
  events: Fr4Event[];
}

const FR4FederatedLearningPage: React.FC = () => {
  const [data, setData] = useState<Fr4Snapshot | null>(null);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    let cancelled = false;

    const load = async () => {
      try {
        const res = await fetch("/fr4_live.json", { cache: "no-store" });
        if (!res.ok) throw new Error(`status ${res.status}`);
        const json = (await res.json()) as Fr4Snapshot;
        if (!cancelled) {
          setData(json);
          setError(null);
        }
      } catch (e) {
        if (!cancelled) {
          console.warn("FR4 live fetch failed", e);
          setError("라이브 FL 상태 불러오기 실패, 마지막 스냅샷 또는 기본값 사용 중");
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
    return <p>FR4 연합학습 라이브 데이터 로딩 중...</p>;
  }

  const { summary, clients, flow, events } = data;

  return (
    <div className="page">
      <h1>FR4 · Personalized Federated Learning</h1>
      <p className="page-subtitle">
        각 집(클라이언트)의 Head Layer는 로컬에 유지한 채, Base Layer만 서버와 교환하는
        FedAvg 진행 상황을 모니터링합니다.
      </p>
      {error && <p className="page-error">{error}</p>}

      <section className="section section-grid section-grid--3">
        <StatCard
          label="현재 Round"
          value={String(summary.global_round ?? "-")}
          sub={`마지막 업데이트: ${summary.last_updated || "-"}`}
        />
        <StatCard
          label="참여 클라이언트"
          value={`${summary.online_clients}/${summary.total_clients}`}
          sub="온라인 / 전체"
        />
        <StatCard
          label="Global Val Loss"
          value={
            summary.avg_loss != null ? summary.avg_loss.toFixed(3) : "-"
          }
          sub="FedAvg 이후 검증 기준"
        />
      </section>

      <section className="section">
        <h2>Client Nodes</h2>
        <table className="table">
          <thead>
            <tr>
              <th>Client ID</th>
              <th>Name</th>
              <th>Status</th>
              <th>Latency</th>
              <th>Loss</th>
              <th>Rounds</th>
            </tr>
          </thead>
          <tbody>
            {clients.map((c) => (
              <tr key={c.id}>
                <td>{c.id}</td>
                <td>{c.name}</td>
                <td>{c.status}</td>
                <td>
                  {c.latency_ms != null ? `${c.latency_ms} ms` : "-"}
                </td>
                <td>
                  {c.loss != null ? c.loss.toFixed(3) : "-"}
                </td>
                <td>{c.rounds ?? "-"}</td>
              </tr>
            ))}
          </tbody>
        </table>
      </section>

      <section className="section section-flex-2">
        <div>
          <h2>연합 학습 Flow</h2>
          <ul className="bullet-list">
            {flow.map((line, idx) => (
              <li key={idx}>{line}</li>
            ))}
          </ul>
        </div>
        <div>
          <h2>Server ↔ Client 이벤트 타임라인</h2>
          <div className="timeline">
            {events.map((e) => (
              <TimelineItem
                key={e.time + e.source + e.event}
                time={e.time}
                label={`${e.source}: ${e.event}`}
              />
            ))}
          </div>
        </div>
      </section>
    </div>
  );
};

export default FR4FederatedLearningPage;
