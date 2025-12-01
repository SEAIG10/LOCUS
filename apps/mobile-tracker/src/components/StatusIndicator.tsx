import { ConnectionStatus, TrackingStatus } from '../types';
import '../styles/StatusIndicator.css';

interface StatusIndicatorProps {
  connectionStatus: ConnectionStatus;
  trackingStatus: TrackingStatus;
}

export const StatusIndicator: React.FC<StatusIndicatorProps> = ({
  connectionStatus,
  trackingStatus,
}) => {
  const getStatusText = (): string => {
    if (trackingStatus === 'tracking') return '추적 중';
    if (connectionStatus === 'connected') return '연결됨';
    if (connectionStatus === 'connecting') return '연결 중...';
    if (connectionStatus === 'error') return '연결 오류';
    return '연결 대기중';
  };

  const getStatusClass = (): string => {
    if (trackingStatus === 'tracking') return 'tracking';
    if (connectionStatus === 'connected') return 'connected';
    return '';
  };

  return (
    <div className="status">
      <span className={`status-dot ${getStatusClass()}`} />
      <span className="status-text">{getStatusText()}</span>
    </div>
  );
};
