# LOCUS Tracker WebSocket Server

TypeScript로 작성된 WebSocket 서버입니다. 모바일 트래커와 LOCUS_CLIENT를 연결합니다.

## 🚀 빠른 시작

### 의존성 설치

```bash
npm install
```

### 개발 모드 실행

```bash
npm run dev
```

### 프로덕션 빌드 및 실행

```bash
npm run build
npm start
```

## 📡 API

### WebSocket

```
ws://localhost:8080
```

### HTTP 엔드포인트

#### GET /status

서버 상태 및 통계

```json
{
  "status": "running",
  "stats": {
    "totalConnections": 10,
    "currentConnections": 2,
    "messagesReceived": 150,
    "messagesSent": 300,
    "uptime": 3600,
    "locationHistoryCount": 100
  }
}
```

#### GET /locations?limit=50

최근 위치 데이터 조회

```json
{
  "count": 100,
  "locations": [
    {
      "clientId": 1,
      "latitude": 37.5665,
      "longitude": 126.9780,
      "accuracy": 15.5,
      "timestamp": "2025-11-18T08:30:00.000Z",
      "receivedAt": "2025-11-18T08:30:00.123Z"
    }
  ]
}
```

## 📨 WebSocket 메시지 프로토콜

자세한 내용은 메인 README를 참조하세요.

## 🔧 환경 변수

```bash
PORT=8080  # 서버 포트 (기본값: 8080)
```

## 🛠️ 기술 스택

- TypeScript
- Node.js
- Express
- ws (WebSocket)

## 📝 타입 정의

`src/types.ts`에서 모든 메시지 타입을 정의합니다.

## 🐛 문제 해결

### 포트 이미 사용 중

```bash
# 다른 포트 사용
PORT=3000 npm run dev
```

### 의존성 오류

```bash
rm -rf node_modules package-lock.json
npm install
```

## 📄 라이선스

MIT
