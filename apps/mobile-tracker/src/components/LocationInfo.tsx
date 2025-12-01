import { ExtendedPosition } from '../types';
import { formatCoordinate, formatAccuracy, formatTime } from '../utils/format';
import '../styles/LocationInfo.css';

interface LocationInfoProps {
  position: ExtendedPosition | null;
}

export const LocationInfo: React.FC<LocationInfoProps> = ({ position }) => {
  if (!position) {
    return (
      <section className="location-section">
        <h2>현재 위치</h2>
        <div className="location-info">
          <div className="info-row">
            <span className="label">위도:</span>
            <span className="value">-</span>
          </div>
          <div className="info-row">
            <span className="label">경도:</span>
            <span className="value">-</span>
          </div>
          <div className="info-row">
            <span className="label">정확도:</span>
            <span className="value">-</span>
          </div>
          <div className="info-row">
            <span className="label">업데이트:</span>
            <span className="value">-</span>
          </div>
        </div>
      </section>
    );
  }

  const { latitude, longitude, accuracy } = position.coords;
  const timestamp = new Date(position.timestamp);

  return (
    <section className="location-section">
      <h2>현재 위치</h2>
      <div className="location-info">
        <div className="info-row">
          <span className="label">위도:</span>
          <span className="value">{formatCoordinate(latitude)}</span>
        </div>
        <div className="info-row">
          <span className="label">경도:</span>
          <span className="value">{formatCoordinate(longitude)}</span>
        </div>
        <div className="info-row">
          <span className="label">정확도:</span>
          <span className="value">{formatAccuracy(accuracy)}</span>
        </div>
        <div className="info-row">
          <span className="label">업데이트:</span>
          <span className="value">{formatTime(timestamp)}</span>
        </div>
      </div>
    </section>
  );
};
