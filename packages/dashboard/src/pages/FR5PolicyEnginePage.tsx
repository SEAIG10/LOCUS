// src/pages/FR5PolicyDecisionPage.tsx
import React, { useEffect, useState } from "react";
import StatCard from "../components/StatCard";
import TimelineItem from "../components/TimelineItem";
import BarChart, { BarDatum } from "../components/BarChart";

interface MapRow {
  zone: string;
  score: number;
}

interface PathRow {
  zone: string;
  eta: number;
  prob: number;
}

interface Fr5Snapshot {
  action: string;
  reason: string;
  eta: number;
  battery: number;
  map: MapRow[];
  path: PathRow[];
  notes: string[];
}

const FR5PolicyDecisionPage: React.FC = () => {
  const [data, setData] = useState<Fr5Snapshot | null>(null);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    let cancelled = false;

    const load = async () => {
      try {
        const res = await fetch("/fr5_live.json", { cache: "no-store" });
        if (!res.ok) throw new Error(`status ${res.status}`);
        const json = (await res.json()) as Fr5Snapshot;
        if (!cancelled) {
          setData(json);
          setError(null);
        }
      } catch (e) {
        if (!cancelled) {
          console.warn("FR5 live fetch failed", e);
          setError("라이브 정책 결정을 불러오지 못했습니다. 마지막 스냅샷 또는 기본값을 표시 중입니다.");
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
    return <p>FR5 정책 엔진 라이브 데이터 로딩 중...</p>;
  }

  const mapBarData: BarDatum[] = data.map.map((m) => ({
    label: m.zone,
    value: m.score,
  }));

  const pathBarData: BarDatum[] = data.path.map((p) => ({
    label: p.zone,
    value: p.prob,
  }));

  return (
    <div className="page">
      <h1>FR5 · Policy Engine &amp; Cleaning Decision</h1>
      <p className="page-subtitle">
        FR1–FR4에서 추론한 맥락을 기반으로, 지금 로봇을 움직일지/언제 움직일지를 최종 결정합니다.
      </p>
      {error && <p className="page-error">{error}</p>}

      <section className="section section-grid section-grid--3">
        <StatCard
          label="현재 액션"
          value={data.action}
          sub={data.reason}
        />
        <StatCard
          label="예상 청소 시간"
          value={`${data.eta}분`}
          sub="이번 라운드 기준"
        />
        <StatCard
          label="배터리"
          value={`${data.battery}%`}
          sub="결정 시점 기준"
        />
      </section>

      <section className="section section-flex-2">
        <div>
          <h2>존별 오염 점수</h2>
          <BarChart
            title="Map Pollution Scores"
            data={mapBarData}
            valueFormat={(v) => `${Math.round(v * 100)}%`}
          />
          <p className="helper-text">
            FR3 예측 + FR1/FR2 맥락을 합쳐 각 존의 “지금 청소 필요도”를 점수화한 값입니다.
          </p>
        </div>

        <div>
          <h2>이번 청소 경로</h2>
          <table className="table">
            <thead>
              <tr>
                <th>Zone</th>
                <th>ETA</th>
                <th>선택 확률</th>
              </tr>
            </thead>
            <tbody>
              {data.path.map((p) => (
                <tr key={p.zone}>
                  <td>{p.zone}</td>
                  <td>{p.eta} 분</td>
                  <td>{(p.prob * 100).toFixed(1)}%</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </section>

      <section className="section">
        <h2>정책 메모</h2>
        <div className="timeline">
          {data.notes.map((note, idx) => (
            <TimelineItem
              key={idx}
              time={`Note ${idx + 1}`}
              label={note}
            />
          ))}
        </div>
      </section>
    </div>
  );
};

export default FR5PolicyDecisionPage;
