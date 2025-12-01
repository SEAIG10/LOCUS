import { TrackingStatus } from '../types';
import '../styles/ControlButtons.css';

interface ControlButtonsProps {
  trackingStatus: TrackingStatus;
  onStart: () => void;
  onStop: () => void;
  disabled: boolean;
}

export const ControlButtons: React.FC<ControlButtonsProps> = ({
  trackingStatus,
  onStart,
  onStop,
  disabled,
}) => {
  const isTracking = trackingStatus === 'tracking';

  return (
    <section className="control-section">
      <button
        onClick={onStart}
        disabled={disabled || isTracking}
        className="btn btn-start"
      >
        ▶️ 추적 시작
      </button>
      <button
        onClick={onStop}
        disabled={!isTracking}
        className="btn btn-stop"
      >
        ⏹️ 추적 중지
      </button>
    </section>
  );
};
