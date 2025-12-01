/**
 * WebSocket Handler (서버)
 * GPS → 3D 변환 시 칼만 필터 적용
 */

import { gpsTo3DSmooth, gpsTo3DHybrid, evaluateAccuracy } from './coordinateUtils';

// 위치 업데이트 처리 함수
function handleLocationUpdate(data: any, ws: WebSocket) {
  const { latitude, longitude, accuracy } = data;

  // 🎯 방법 1: 칼만 필터만 (빠르고 반응성 좋음)
  const position3D = gpsTo3DSmooth(
    { latitude, longitude },
    accuracy
  );

  // 🎯 방법 2: 하이브리드 (칼만 + Moving Average, 가장 부드러움)
  // const position3D = gpsTo3DHybrid(
  //   { latitude, longitude },
  //   accuracy
  // );

  // 정확도 평가
  const quality = evaluateAccuracy(accuracy);
  
  console.log(`📍 위치 업데이트:`, {
    raw: `(${latitude}, ${longitude})`,
    filtered: `(${position3D.x.toFixed(2)}, ${position3D.z.toFixed(2)})`,
    accuracy: `±${accuracy.toFixed(1)}m`,
    quality,
  });

  // 정확도가 너무 낮으면 경고만 하고 데이터는 전송
  if (quality === 'poor') {
    console.warn(`⚠️ 낮은 GPS 정확도 (±${accuracy}m) - 칼만 필터로 보정 중`);
  }

  // 클라이언트로 전송
  const message = {
    type: 'location_update',
    data: {
      latitude,
      longitude,
      accuracy,
      position3D,
      quality, // 품질 정보 추가
      timestamp: new Date().toISOString(),
    },
  };

  // 모든 뷰어에게 브로드캐스트
  broadcastToViewers(message);
}

// 뷰어들에게 브로드캐스트하는 함수 (예시)
function broadcastToViewers(message: any) {
  // WebSocket 클라이언트 목록에서 viewer만 필터링해서 전송
  // 실제 구현은 서버 구조에 따라 다름
}

export { handleLocationUpdate };