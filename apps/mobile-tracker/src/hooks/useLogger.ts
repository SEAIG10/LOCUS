import { useCallback, useState } from 'react';
import { LogEntry, LogLevel } from '../types';

const MAX_LOGS = 100;

export const useLogger = () => {
  const [logs, setLogs] = useState<LogEntry[]>([]);

  const addLog = useCallback((level: LogLevel, message: string) => {
    const entry: LogEntry = {
      id: `${Date.now()}-${Math.random()}`,
      level,
      message,
      timestamp: new Date(),
    };

    setLogs((prev) => {
      const newLogs = [...prev, entry];
      // 최대 로그 수 제한
      if (newLogs.length > MAX_LOGS) {
        return newLogs.slice(-MAX_LOGS);
      }
      return newLogs;
    });
  }, []);

  const clearLogs = useCallback(() => {
    setLogs([]);
    addLog('info', '로그 초기화');
  }, [addLog]);

  return {
    logs,
    addLog,
    clearLogs,
  };
};
