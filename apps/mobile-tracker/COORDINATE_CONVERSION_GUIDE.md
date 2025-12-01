# 좌표 변환 시스템 사용 가이드

GPS 좌표를 3D 공간 좌표로 변환하는 시스템입니다.

## 🎯 개요

```
[모바일 앱] GPS (위도, 경도)
    ↓
[WebSocket 서버] GPS → 3D 좌표 변환
    ↓
[LOCUS_CLIENT] 3D 좌표로 로봇 위치 표시
```

## 📍 1단계: 기준점 설정

### 방법 1: 코드에서 직접 설정

`server/src/coordinateConverter.ts` 파일에서:

```typescript
const REFERENCE_POINT: GPSCoordinate = {
  latitude: 37.563517,   // 집의 실제 위도
  longitude: 127.079571, // 집의 실제 경도
};
```

### 방법 2: API로 동적 설정

서버가 실행 중일 때:

```bash
curl -X POST http://localhost:8080/reference-point \
  -H "Content-Type: application/json" \
  -d '{
    "latitude": 37.563517,
    "longitude": 127.079571
  }'
```

### 기준점 확인

```bash
curl http://localhost:8080/reference-point
```

## 🏠 기준점 찾는 방법

1. **집의 중앙**(예: 거실 중앙)에 서기
2. 모바일 앱 실행
3. 서버 연결 및 추적 시작
4. 로그에서 GPS 좌표 확인:
   ```
   위도: 37.563517
   경도: 127.079571
   ```
5. 이 좌표를 기준점으로 설정

## 📊 좌표 시스템 이해

### GPS 좌표
```
위도 (latitude): 북위/남위 (37.563517)
경도 (longitude): 동경/서경 (127.079571)
```

### 3D 좌표
```
X축: 동서 방향 (경도)
  - 동쪽: +X
  - 서쪽: -X

Y축: 높이 (현재는 0으로 고정)
  - 위: +Y
  - 아래: -Y

Z축: 남북 방향 (위도)
  - 북쪽: -Z
  - 남쪽: +Z
```

### 변환 예시

기준점: (37.563517, 127.079571)

현재 위치: (37.563527, 127.079581)

차이:
- 위도 차이: 0.00001도 = 약 1.11m
- 경도 차이: 0.00001도 = 약 0.89m

3D 좌표:
- X: 0.89m (동쪽으로)
- Y: 0m
- Z: -1.11m (북쪽으로)

## 🔧 스케일 조정

집이 너무 크거나 작게 보이면 `coordinateConverter.ts`에서 스케일 조정:

```typescript
const SCALE = {
  latToZ: -111000,  // 기본값: -111000
  lonToX: 88800,    // 기본값: 88800
};
```

예시:
- 집이 2배 크게 보이게: `222000`, `177600`
- 집이 0.5배 작게 보이게: `55500`, `44400`

## 📡 서버 API

### GET /status
서버 상태 및 통계

### GET /locations?limit=50
최근 위치 히스토리 (3D 좌표 포함)

### GET /reference-point
현재 기준점 조회

### POST /reference-point
기준점 설정

```json
{
  "latitude": 37.563517,
  "longitude": 127.079571
}
```

## 💻 LOCUS_CLIENT 연동

### WebSocket 연결

```typescript
const ws = new WebSocket('wss://your-server.ngrok-free.app');

ws.onopen = () => {
  // 뷰어로 식별
  ws.send(JSON.stringify({
    type: 'identify',
    clientType: 'viewer'
  }));
};
```

### 위치 업데이트 수신

```typescript
ws.onmessage = (event) => {
  const message = JSON.parse(event.data);
  
  if (message.type === 'location_update') {
    const { position3D, latitude, longitude, accuracy } = message.data;
    
    console.log('GPS:', latitude, longitude);
    console.log('3D:', position3D.x, position3D.y, position3D.z);
    console.log('정확도:', accuracy);
    
    // 로봇 마커 위치 업데이트
    updateRobotPosition(position3D);
  }
};
```

### RobotMarker 업데이트

```typescript
function updateRobotPosition(position3D: { x: number; y: number; z: number }) {
  // Three.js 객체 위치 설정
  robotMarker.position.set(
    position3D.x,
    position3D.y,
    position3D.z
  );
  
  // 카메라가 로봇 따라가기 (선택사항)
  camera.lookAt(robotMarker.position);
}
```

## 📝 서버 로그 예시

```
[수신] 클라이언트 #1:
  위도: 37.563527
  경도: 127.079581
  정확도: ±15.0m
  시간: 2025. 11. 18. 오후 6:33:55
  3D 좌표: (0.89, 0.00, -1.11)
  기준점 거리: 1.42m
  → 1개 클라이언트에게 브로드캐스트
```

## 🐛 문제 해결

### 로봇이 너무 멀리 떨어져 보임
→ 기준점이 잘못 설정됨. 집 중앙에서 GPS 좌표 다시 측정

### 로봇이 너무 크거나 작음
→ SCALE 값 조정

### 로봇이 반대 방향으로 움직임
→ SCALE 값에 음수 부호 추가/제거

### 3D 좌표가 이상함
→ 서버 로그에서 "3D 좌표" 확인
→ `/reference-point` API로 기준점 확인

## 🎯 테스트 방법

### 1. 서버 시작

```bash
cd server
npm run dev
```

### 2. 기준점 설정 확인

```bash
curl http://localhost:8080/reference-point
```

### 3. 모바일 앱 연결

ngrok URL로 접속 및 추적 시작

### 4. 로그 확인

서버 터미널에서 3D 좌표가 올바르게 출력되는지 확인

### 5. LOCUS_CLIENT 연결

WebSocket으로 연결하여 실시간 위치 확인

## 💡 팁

1. **실내 GPS는 정확도가 낮아요** (±5~50m)
   - 야외에서 먼저 테스트 추천
   - Wi-Fi를 켜면 실내 정확도 향상

2. **기준점은 집 중앙에 설정하세요**
   - 거실, 복도 등 중심점 추천
   - 벽이나 모퉁이는 피하기

3. **스케일은 천천히 조정하세요**
   - 처음엔 기본값 사용
   - 나중에 실제 집 크기에 맞춰 미세 조정

4. **로그를 자주 확인하세요**
   - 서버 로그에 모든 정보가 출력됨
   - 3D 좌표가 이상하면 바로 확인 가능

## 📄 타입 정의

```typescript
// GPS 좌표
interface GPSCoordinate {
  latitude: number;
  longitude: number;
}

// 3D 좌표
interface Position3D {
  x: number;
  y: number;
  z: number;
}

// 위치 업데이트 메시지
interface LocationUpdateMessage {
  type: 'location_update';
  data: {
    clientId: number;
    latitude: number;
    longitude: number;
    accuracy: number;
    timestamp: string;
    receivedAt: string;
    position3D: Position3D;  // 추가된 필드
  };
}
```

---

궁금한 점이 있으면 서버 README.md를 참조하세요! 📖
