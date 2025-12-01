import React from "react";

export interface BarDatum {
  label: string;
  value: number;
}

interface BarChartProps {
  title?: string;
  data: BarDatum[];
  valueFormat?: (value: number) => string;
}

const defaultFormat = (value: number) => `${Math.round(value * 100)}%`;

const BarChart: React.FC<BarChartProps> = ({ title, data, valueFormat = defaultFormat }) => {
  const safeMax = data.length ? Math.max(...data.map((d) => d.value), 0.0001) : 1;

  return (
    <div className="barchart">
      {title && <div className="barchart-title">{title}</div>}
      <div className="barchart-body">
        {data.map((datum) => (
          <div key={datum.label} className="barchart-row">
            <div className="barchart-label">{datum.label}</div>
            <div className="barchart-bar-wrapper">
              <div
                className="barchart-bar"
                style={{ width: `${(datum.value / safeMax) * 100}%` }}
              />
            </div>
            <div className="barchart-value">{valueFormat(datum.value)}</div>
          </div>
        ))}
      </div>
    </div>
  );
};

export default BarChart;
