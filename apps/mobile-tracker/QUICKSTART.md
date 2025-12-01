# LOCUS Mobile Tracker - 빠른 시작 가이드

## 📦 1. 압축 해제 및 의존성 설치

```bash
# 압축 해제
unzip locus-mobile-tracker.zip
cd locus-mobile-tracker

# 클라이언트 의존성 설치
npm install

# 서버 의존성 설치
cd server
npm install
cd ..
```

## 🚀 2. 서버 실행

터미널 1:
```bash
cd server
npm run dev
```

✅ 서버가 `ws://0.0.0.0:8080`에서 실행됩니다.

## 📱 3. 모바일 앱 실행

터미널 2:
```bash
cd locus-mobile-tracker
npm run host
```

✅ 앱이 `http://0.0.0.0:5173`에서 실행됩니다.

## 🌐 4. 휴대폰에서 접속

### 컴퓨터 IP 확인

**Mac/Linux:**
```bash
ifconfig | grep "inet "
# 또는
hostname -I
```

**Windows:**
```cmd
ipconfig
```

### 휴대폰 브라우저에서 접속

```
http://192.168.x.x:5173
```

예: `http://192.168.0.10:5173`

## 🔌 5. 서버 연결

1. 앱에서 WebSocket URL 입력:
   ```
   ws://192.168.x.x:8080
   ```
   
2. "연결하기" 버튼 클릭

3. "추적 시작" 버튼 클릭

4. 위치 권한 허용

5. 집 안에서 휴대폰을 들고 돌아다니기! 🚶‍♂️

## 🐛 문제 해결

### "Cannot connect to server"
- 서버가 실행 중인지 확인
- 방화벽에서 8080 포트 허용
- 같은 Wi-Fi 네트워크에 연결되어 있는지 확인

### "Location permission denied"
- 브라우저 설정에서 위치 권한 허용
- iOS: 설정 > Safari > 위치 서비스
- Android: 설정 > 앱 > Chrome > 권한

### 페이지가 안 열림
- 컴퓨터와 휴대폰이 같은 Wi-Fi에 있는지 확인
- IP 주소가 정확한지 확인
- 방화벽에서 5173 포트 허용

## 📊 서버 상태 확인

브라우저에서:
```
http://192.168.x.x:8080/status
```

## 🎯 다음 단계

LOCUS_CLIENT와 연동하려면 메인 README.md를 참조하세요!

---

궁금한 점이 있으면 README.md를 확인하세요! 📖
