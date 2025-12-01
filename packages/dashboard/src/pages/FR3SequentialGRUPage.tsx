import React, { useEffect, useState } from "react";
import TimelineItem from "../components/TimelineItem";
import BarChart, { BarDatum } from "../components/BarChart";

interface TimelineRow { time: string; summary: string; }
interface AttentionRow { label: string; weight: number; }
interface PredictionRow { zone: string; value: number; }

interface Fr3Snapshot {
  timeline: TimelineRow[];
  attention: AttentionRow[];
  prediction: PredictionRow[];
}

const FR3SequentialGRUPage: React.FC = () => {
  const [data, setData] = useState<Fr3Snapshot | null>(null);

  useEffect(() => {
    const load = async () => {
      try {
        const r = await fetch("/fr3_live.json", { cache: "no-store" });
        if (r.ok) {
          const json = await r.json();
          setData(json);
        }
      } catch (_) {
        // ignore, keep old data
      }
    };

    load();
    const id = setInterval(load, 1000);
    return () => clearInterval(id);
  }, []);

  if (!data) return <p>Loading FR3 live data...</p>;

  return (
    <div className="page">
      <h1>FR3 · Sequential GRU Prediction</h1>

      <section className="section">
        <h2>Context Timeline</h2>
        <div className="timeline">
          {data.timeline.map((item, idx) => (
            <TimelineItem
              key={item.time}
              time={item.time}
              label={item.summary}
              highlight={idx === data.timeline.length - 1}
            />
          ))}
        </div>
      </section>

      <section className="section section-flex-2">
        <div>
          <h2>Attention Weights</h2>
          <ul className="bullet-list">
            {data.attention.map((a) => (
              <li key={a.label}>{`${a.label}: ${(a.weight * 100).toFixed(1)}%`}</li>
            ))}
          </ul>
        </div>

        <div>
          <BarChart
            title="Zone별 오염 확률"
            data={data.prediction.map(p => ({
              label: p.zone,   // 변환!
              value: p.value
            }))}
            valueFormat={(v) => `${Math.round(v * 100)}%`}
          />
        </div>
      </section>
    </div>
  );
};

export default FR3SequentialGRUPage;
