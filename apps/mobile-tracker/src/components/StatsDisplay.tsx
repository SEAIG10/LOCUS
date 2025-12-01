import { Stats } from '../types';
import '../styles/Stats.css';

interface StatsDisplayProps {
  stats: Stats;
}

export const StatsDisplay: React.FC<StatsDisplayProps> = ({ stats }) => {
  return (
    <section className="stats-section">
      <h2>전송 통계</h2>
      <div className="stats-grid">
        <div className="stat-item">
          <div className="stat-value">{stats.sentCount}</div>
          <div className="stat-label">전송 횟수</div>
        </div>
        <div className="stat-item">
          <div className="stat-value">{stats.errorCount}</div>
          <div className="stat-label">오류 횟수</div>
        </div>
      </div>
    </section>
  );
};
