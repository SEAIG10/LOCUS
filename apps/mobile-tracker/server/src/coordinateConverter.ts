/**
 * GPS 좌표를 3D 공간 좌표로 변환하는 유틸리티
 * + 칼만 필터로 노이즈 제거
 */

export interface GPSCoordinate {
  latitude: number;
  longitude: number;
}

export interface Position3D {
  x: number;
  y: number;
  z: number;
}

/**
 * 집의 기준점 설정
 * 이 좌표가 3D 공간의 (0, 0, 0)이 됩니다
 */
let REFERENCE_POINT: GPSCoordinate = {
  latitude: 37.563517,   // TODO: 집의 실제 위도로 변경
  longitude: 127.079571, // TODO: 집의 실제 경도로 변경
};

/**
 * 좌표 변환 스케일
 * 
 * 한국(위도 37도 부근) 기준:
 * - 위도 1도 = 약 111km
 * - 경도 1도 = 약 88.8km
 */
const SCALE = {
  latToZ: -111000,  // 위도 → Z축 (남북)
  lonToX: 88800,    // 경도 → X축 (동서)
};

/**
 * 1차원 칼만 필터
 * GPS 노이즈를 줄여 부드러운 위치 추적
 */
class KalmanFilter {
  private x: number = 0; // 추정 위치
  private p: number = 1; // 추정 오차 공분산
  private q: number; // 프로세스 노이즈
  private r: number; // 측정 노이즈

  constructor(processNoise = 0.001, measurementNoise = 0.05) {
    this.q = processNoise;
    this.r = measurementNoise;
  }

  update(measurement: number): number {
    // 예측 단계
    this.p = this.p + this.q;

    // 칼만 게인 계산
    const k = this.p / (this.p + this.r);

    // 업데이트 단계
    this.x = this.x + k * (measurement - this.x);
    this.p = (1 - k) * this.p;

    return this.x;
  }

  reset(): void {
    this.x = 0;
    this.p = 1;
  }

  getValue(): number {
    return this.x;
  }
}

// 각 축마다 독립적인 칼만 필터
const kalmanX = new KalmanFilter(0.001, 0.05);
const kalmanZ = new KalmanFilter(0.001, 0.05);

/**
 * Moving Average 필터
 */
const positionHistory: Position3D[] = [];
const HISTORY_SIZE = 3;

/**
 * GPS 좌표를 3D 좌표로 변환 (기본 - 필터 없음)
 * 
 * @param gps GPS 좌표 (위도, 경도)
 * @returns 3D 좌표 (x, y, z) in meters
 */
export function gpsTo3D(gps: GPSCoordinate): Position3D {
  const latDiff = gps.latitude - REFERENCE_POINT.latitude;
  const lonDiff = gps.longitude - REFERENCE_POINT.longitude;

  const x = lonDiff * SCALE.lonToX;
  const z = latDiff * SCALE.latToZ;
  const y = 0;

  return { x, y, z };
}

/**
 * GPS 좌표를 3D 좌표로 변환 + 칼만 필터 적용 ⭐
 * 
 * @param gps GPS 좌표
 * @param accuracy GPS 정확도 (미터) - 선택적
 * @returns 필터링된 3D 좌표
 */
export function gpsTo3DSmooth(gps: GPSCoordinate, accuracy?: number): Position3D {
  if (accuracy && accuracy > 20) {
    console.warn(`⚠️  GPS 정확도 낮음: ±${accuracy}m (칼만 필터 적용 중)`);
  }

  const latDiff = gps.latitude - REFERENCE_POINT.latitude;
  const lonDiff = gps.longitude - REFERENCE_POINT.longitude;

  const rawX = lonDiff * SCALE.lonToX;
  const rawZ = latDiff * SCALE.latToZ;

  // 칼만 필터 적용
  const x = kalmanX.update(rawX);
  const z = kalmanZ.update(rawZ);
  const y = 0;

  return { x, y, z };
}

/**
 * GPS 좌표를 3D 좌표로 변환 + Moving Average 필터 ⭐
 * 
 * @param gps GPS 좌표
 * @returns 평균화된 3D 좌표
 */
export function gpsTo3DAverage(gps: GPSCoordinate): Position3D {
  const latDiff = gps.latitude - REFERENCE_POINT.latitude;
  const lonDiff = gps.longitude - REFERENCE_POINT.longitude;

  const rawPosition = {
    x: lonDiff * SCALE.lonToX,
    y: 0,
    z: latDiff * SCALE.latToZ,
  };

  positionHistory.push(rawPosition);
  if (positionHistory.length > HISTORY_SIZE) {
    positionHistory.shift();
  }

  const avgX = positionHistory.reduce((sum, p) => sum + p.x, 0) / positionHistory.length;
  const avgZ = positionHistory.reduce((sum, p) => sum + p.z, 0) / positionHistory.length;

  return { x: avgX, y: 0, z: avgZ };
}

/**
 * 칼만 필터 + Moving Average 병합 (최고 품질) ⭐⭐⭐
 * 
 * @param gps GPS 좌표
 * @param accuracy GPS 정확도 (미터)
 * @returns 이중 필터링된 3D 좌표
 */
export function gpsTo3DHybrid(gps: GPSCoordinate, accuracy?: number): Position3D {
  // 1단계: 칼만 필터
  const kalmanFiltered = gpsTo3DSmooth(gps, accuracy);

  // 2단계: Moving Average
  positionHistory.push(kalmanFiltered);
  if (positionHistory.length > HISTORY_SIZE) {
    positionHistory.shift();
  }

  const avgX = positionHistory.reduce((sum, p) => sum + p.x, 0) / positionHistory.length;
  const avgZ = positionHistory.reduce((sum, p) => sum + p.z, 0) / positionHistory.length;

  return { x: avgX, y: 0, z: avgZ };
}

/**
 * 3D 좌표를 GPS 좌표로 역변환 (디버깅용)
 */
export function position3DToGPS(position: Position3D): GPSCoordinate {
  const latitude = REFERENCE_POINT.latitude + position.z / SCALE.latToZ;
  const longitude = REFERENCE_POINT.longitude + position.x / SCALE.lonToX;

  return { latitude, longitude };
}

/**
 * 기준점 업데이트
 */
export function updateReferencePoint(newReference: GPSCoordinate): void {
  REFERENCE_POINT = { ...newReference };
  
  // 필터 초기화
  kalmanX.reset();
  kalmanZ.reset();
  positionHistory.length = 0;
  
  console.log(`📍 기준점 업데이트: (${newReference.latitude}, ${newReference.longitude})`);
}

/**
 * 현재 기준점 조회
 */
export function getReferencePoint(): GPSCoordinate {
  return { ...REFERENCE_POINT };
}

/**
 * 모든 필터 초기화
 */
export function resetFilters(): void {
  kalmanX.reset();
  kalmanZ.reset();
  positionHistory.length = 0;
  console.log('🔄 필터 초기화 완료');
}

/**
 * 두 GPS 좌표 간의 거리 계산 (미터)
 * Haversine formula 사용
 */
export function calculateDistance(
  coord1: GPSCoordinate,
  coord2: GPSCoordinate
): number {
  const R = 6371000; // 지구 반지름 (미터)
  const lat1 = (coord1.latitude * Math.PI) / 180;
  const lat2 = (coord2.latitude * Math.PI) / 180;
  const deltaLat = ((coord2.latitude - coord1.latitude) * Math.PI) / 180;
  const deltaLon = ((coord2.longitude - coord1.longitude) * Math.PI) / 180;

  const a =
    Math.sin(deltaLat / 2) * Math.sin(deltaLat / 2) +
    Math.cos(lat1) *
      Math.cos(lat2) *
      Math.sin(deltaLon / 2) *
      Math.sin(deltaLon / 2);

  const c = 2 * Math.atan2(Math.sqrt(a), Math.sqrt(1 - a));

  return R * c;
}

/**
 * GPS 정확도 평가
 */
export function evaluateAccuracy(accuracy: number): 'excellent' | 'good' | 'fair' | 'poor' {
  if (accuracy <= 5) return 'excellent';
  if (accuracy <= 10) return 'good';
  if (accuracy <= 20) return 'fair';
  return 'poor';
}