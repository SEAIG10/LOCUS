import { useEffect, useRef } from 'react';
import { LogEntry } from '../types';
import { formatTime } from '../utils/format';
import '../styles/LogViewer.css';

interface LogViewerProps {
  logs: LogEntry[];
  onClear: () => void;
}

export const LogViewer: React.FC<LogViewerProps> = ({ logs, onClear }) => {
  const containerRef = useRef<HTMLDivElement>(null);

  // 새 로그가 추가되면 자동 스크롤
  useEffect(() => {
    if (containerRef.current) {
      containerRef.current.scrollTop = containerRef.current.scrollHeight;
    }
  }, [logs]);

  return (
    <section className="log-section">
      <h2>로그</h2>
      <div ref={containerRef} className="log-container">
        {logs.length === 0 ? (
          <div className="log-entry info">
            <span className="timestamp">[--:--:--]</span>
            로그가 없습니다.
          </div>
        ) : (
          logs.map((log) => (
            <div key={log.id} className={`log-entry ${log.level}`}>
              <span className="timestamp">[{formatTime(log.timestamp)}]</span>
              {log.message}
            </div>
          ))
        )}
      </div>
      <button onClick={onClear} className="btn btn-secondary">
        로그 지우기
      </button>
    </section>
  );
};
