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
 * 
 * 설정 방법:
 * 1. 집의 중앙(예: 거실 중앙)에 서서 GPS 좌표 확인
 * 2. 그 좌표를 여기에 입력
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
  latToZ: -111000,  // 위도 → Z축 (남북), 음수는 남쪽이 +Z
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

  /**
   * 새 측정값으로 필터 업데이트
   * @param measurement 측정값 (예: GPS에서 변환된 x 또는 z 좌표)
   * @returns 필터링된 추정값
   */
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

  /**
   * 필터 초기화 (기준점 변경 시 사용)
   */
  reset(): void {
    this.x = 0;
    this.p = 1;
  }

  /**
   * 현재 추정값 조회
   */
  getValue(): number {
    return this.x;
  }
}

// 각 축마다 독립적인 칼만 필터
const kalmanX = new KalmanFilter(0.001, 0.05);
const kalmanZ = new KalmanFilter(0.001, 0.05);

/**
 * Moving Average 필터 (칼만 필터와 병행 사용 가능)
 */
const positionHistory: Position3D[] = [];
const HISTORY_SIZE = 3; // 최근 3개 평균

/**
 * GPS 좌표를 3D 좌표로 변환 (기본)
 * 
 * @param gps GPS 좌표 (위도, 경도)
 * @returns 3D 좌표 (x, y, z) in meters
 */
export function gpsTo3D(gps: GPSCoordinate): Position3D {
  // 기준점으로부터의 차이 계산
  const latDiff = gps.latitude - REFERENCE_POINT.latitude;
  const lonDiff = gps.longitude - REFERENCE_POINT.longitude;

  // 3D 좌표로 변환
  const x = lonDiff * SCALE.lonToX; // 경도 차이 → X축 (동서)
  const z = latDiff * SCALE.latToZ; // 위도 차이 → Z축 (남북)
  const y = 0; // 지면 높이

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
  // 정확도가 너무 낮으면 경고 (하지만 무시하지는 않음)
  if (accuracy && accuracy > 20) {
    console.warn(`⚠️ GPS 정확도 매우 낮음: ±${accuracy}m`);
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
 * (칼만 필터보다 단순하지만 효과적)
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

  // 히스토리에 추가
  positionHistory.push(rawPosition);
  if (positionHistory.length > HISTORY_SIZE) {
    positionHistory.shift();
  }

  // 평균 계산
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
 * 
 * @param position 3D 좌표 (x, y, z)
 * @returns GPS 좌표 (위도, 경도)
 */
export function position3DToGPS(position: Position3D): GPSCoordinate {
  const latitude = REFERENCE_POINT.latitude + position.z / SCALE.latToZ;
  const longitude = REFERENCE_POINT.longitude + position.x / SCALE.lonToX;

  return { latitude, longitude };
}

/**
 * 기준점 업데이트 (런타임에 변경 가능)
 * 
 * @param newReference 새로운 기준점
 */
export function updateReferencePoint(newReference: GPSCoordinate): void {
  REFERENCE_POINT = { ...newReference };
  
  // 필터 초기화 (기준점이 바뀌었으므로)
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

  return R * c; // 미터 단위 거리
}

/**
 * GPS 정확도 평가
 * @param accuracy 정확도 (미터)
 * @returns 품질 레벨
 */
export function evaluateAccuracy(accuracy: number): 'excellent' | 'good' | 'fair' | 'poor' {
  if (accuracy <= 5) return 'excellent';
  if (accuracy <= 10) return 'good';
  if (accuracy <= 20) return 'fair';
  return 'poor';
}

/**
 * 필터 통계 조회 (디버깅용)
 */
export function getFilterStats() {
  return {
    kalmanX: kalmanX.getValue(),
    kalmanZ: kalmanZ.getValue(),
    historySize: positionHistory.length,
  };
}