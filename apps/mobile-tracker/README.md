# LOCUS Mobile Tracker

React + TypeScript 기반 모바일 위치 추적 웹앱입니다. 로봇청소기 대신 휴대폰을 사용해 실시간 위치 데이터를 수집하고 WebSocket을 통해 전송합니다.

## 🎯 프로젝트 구조

```
apps/mobile-tracker/
├── src/                      # React 앱 소스
│   ├── components/          # React 컴포넌트
│   ├── hooks/               # 커스텀 훅
│   ├── types/               # TypeScript 타입 정의
│   ├── utils/               # 유틸리티 함수
│   ├── styles/              # CSS 파일
│   ├── App.tsx              # 메인 앱 컴포넌트
│   └── main.tsx             # 엔트리 포인트
├── server/                   # WebSocket 서버
│   ├── src/
│   │   ├── server.ts        # 서버 메인 코드
│   │   └── types.ts         # 서버 타입 정의
│   ├── package.json
│   └── tsconfig.json
├── index.html
├── package.json
├── tsconfig.json
├── vite.config.ts
└── README.md
```

## 🚀 빠른 시작

### 1. 프로젝트 클론 및 의존성 설치

```bash
# 프로젝트 디렉토리로 이동
cd apps/mobile-tracker

# 클라이언트 의존성 설치
npm install

# 서버 의존성 설치
cd server
npm install
cd ..
```

### 2. WebSocket 서버 실행

```bash
cd server
npm run dev
```

서버가 `ws://0.0.0.0:8080`에서 실행됩니다.

### 3. 모바일 앱 실행

새 터미널에서:

```bash
cd apps/mobile-tracker
npm run host
```

앱이 `http://0.0.0.0:5173`에서 실행됩니다.

### 4. 휴대폰에서 접속

컴퓨터의 로컬 IP 주소를 확인하고 휴대폰 브라우저에서 접속:

```
http://192.168.x.x:5173
```

## 📱 사용 방법

1. **서버 연결**
   - WebSocket URL 입력 (예: `ws://192.168.x.x:8080`)
   - "연결하기" 버튼 클릭

2. **위치 추적 시작**
   - "추적 시작" 버튼 클릭
   - 브라우저 위치 권한 허용
   - 집 안에서 휴대폰을 들고 돌아다니기

3. **실시간 모니터링**
   - 현재 좌표 확인
   - 전송 통계 확인
   - 로그에서 상세 정보 확인

## 🎨 주요 기능

### 모바일 앱
- ✅ React + TypeScript로 타입 안전성 보장
- ✅ 커스텀 훅으로 로직 분리 (`useWebSocket`, `useGeolocation`, `useLogger`)
- ✅ 실시간 GPS 위치 추적
- ✅ WebSocket 실시간 통신
- ✅ 깔끔한 컴포넌트 구조
- ✅ 반응형 모바일 UI

### WebSocket 서버
- ✅ TypeScript로 작성
- ✅ Express + ws 라이브러리
- ✅ 클라이언트 관리 (tracker/viewer 구분)
- ✅ 위치 히스토리 저장 (최근 100개)
- ✅ HTTP API 제공 (`/status`, `/locations`)
- ✅ Heartbeat로 연결 관리

## 🛠️ 기술 스택

### Frontend
- React 18
- TypeScript 5
- Vite
- CSS Modules

### Backend
- Node.js
- TypeScript
- Express
- ws (WebSocket)

## 📊 데이터 형식

### 위치 데이터 (Location Data)

```typescript
interface LocationData {
  type: 'location';
  latitude: number;
  longitude: number;
  accuracy: number;
  timestamp: string;
  altitude: number | null;
  heading: number | null;
  speed: number | null;
}
```

### 위치 업데이트 브로드캐스트

```typescript
interface LocationUpdateMessage {
  type: 'location_update';
  data: LocationData & {
    clientId: number;
    receivedAt: string;
  };
}
```

## 🔗 LOCUS_CLIENT 연동

LOCUS_CLIENT에서 WebSocket 서버에 연결하여 실시간 위치 데이터를 수신할 수 있습니다:

```typescript
const ws = new WebSocket('ws://localhost:8080');

// 뷰어로 식별
ws.onopen = () => {
  ws.send(JSON.stringify({
    type: 'identify',
    clientType: 'viewer'
  }));
};

// 위치 업데이트 수신
ws.onmessage = (event) => {
  const message = JSON.parse(event.data);
  
  if (message.type === 'location_update') {
    const { latitude, longitude } = message.data;
    // 3D 뷰에서 로봇 마커 위치 업데이트
    updateRobotPosition(latitude, longitude);
  }
};
```

## 🔧 개발 스크립트

### 클라이언트

```bash
npm run dev      # 개발 서버 실행 (localhost만)
npm run host     # 개발 서버 실행 (네트워크 노출)
npm run build    # 프로덕션 빌드
npm run preview  # 빌드 미리보기
```

### 서버

```bash
npm run dev      # 개발 모드 (자동 재시작)
npm run build    # TypeScript 컴파일
npm start        # 프로덕션 실행
```

## 🌐 네트워크 설정

### 같은 네트워크에서 접속

1. 컴퓨터의 로컬 IP 확인:
   ```bash
   # Mac/Linux
   ifconfig | grep "inet "
   
   # Windows
   ipconfig
   ```

2. 방화벽에서 포트 허용:
   - 클라이언트: 5173
   - 서버: 8080

### HTTPS 사용 (프로덕션)

Geolocation API는 HTTPS가 아닌 환경에서 제한될 수 있습니다. 프로덕션 환경에서는:
- Let's Encrypt로 SSL 인증서 발급
- WSS(보안 WebSocket) 사용

## 🐛 문제 해결

### 위치 권한 거부
- 브라우저 설정에서 위치 권한 확인
- iOS: 설정 > Safari > 위치 서비스
- Android: 설정 > 앱 > Chrome > 권한

### WebSocket 연결 실패
- 서버가 실행 중인지 확인
- 방화벽 설정 확인
- 올바른 IP 주소와 포트 확인

### 위치 정확도 낮음
- GPS 신호가 좋은 야외에서 테스트
- Wi-Fi를 켜면 실내 위치 정확도 향상
- `enableHighAccuracy: true` 옵션 사용 중

## 📝 타입 안전성

TypeScript를 사용하여 전체 프로젝트에서 타입 안전성을 보장합니다:

- 클라이언트와 서버 간 메시지 타입 공유
- 컴파일 타임에 오류 감지
- IDE 자동완성 지원
- 리팩토링 안전성

## 🔒 보안 고려사항

- 위치 데이터는 민감 정보입니다
- 프로덕션에서는 HTTPS/WSS 사용 필수
- 인증/인가 시스템 추가 권장
- CORS 설정 필요시 추가

## 📄 라이선스

MIT

## 🤝 기여

이슈나 PR은 언제든 환영합니다!

---

Made with ❤️ for LOCUS Project
