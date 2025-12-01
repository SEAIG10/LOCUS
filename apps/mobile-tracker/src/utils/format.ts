// 시간 포맷팅
export const formatTime = (date: Date): string => {
  return date.toLocaleTimeString('ko-KR', {
    hour: '2-digit',
    minute: '2-digit',
    second: '2-digit',
  });
};

// 날짜 포맷팅
export const formatDateTime = (date: Date): string => {
  return date.toLocaleString('ko-KR', {
    year: 'numeric',
    month: '2-digit',
    day: '2-digit',
    hour: '2-digit',
    minute: '2-digit',
    second: '2-digit',
  });
};

// 좌표 포맷팅
export const formatCoordinate = (value: number, decimals: number = 6): string => {
  return value.toFixed(decimals);
};

// 정확도 포맷팅
export const formatAccuracy = (accuracy: number): string => {
  return `±${accuracy.toFixed(1)}m`;
};

// Geolocation 에러 메시지
export const getGeolocationErrorMessage = (error: GeolocationPositionError): string => {
  switch (error.code) {
    case error.PERMISSION_DENIED:
      return '위치 권한이 거부되었습니다.';
    case error.POSITION_UNAVAILABLE:
      return '위치 정보를 사용할 수 없습니다.';
    case error.TIMEOUT:
      return '위치 요청 시간 초과';
    default:
      return '알 수 없는 오류 발생';
  }
};

// Geolocation 지원 여부 확인
export const isGeolocationSupported = (): boolean => {
  return 'geolocation' in navigator;
};
