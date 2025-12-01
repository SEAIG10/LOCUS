📊 LOCUS Dashboard

Real-time Monitoring for FR1–FR5 Modules

LOCUS Dashboard는 LOCUS 시스템의 5개 모듈(FR1~FR5)에서 생성되는 실시간 상태를
웹 브라우저에서 시각적으로 확인하기 위한 React + Vite 기반 대시보드이다.

각 모듈(FR1~FR5)은 자체 “Probe(스냅샷 생성기)”가 JSON 형태의 라이브 데이터를 업데이트하고,
대시보드 UI는 해당 JSON 파일을 주기적으로 가져와 화면을 갱신하는 구조이다.

대시보드는 ZeroMQ/MQTT와 직접 통신하지 않는다.
모든 실시간 데이터는 packages/dashboard/public/*.json 파일로 표현된다.

🧩 Dashboard Architecture Overview
┌─────────────┐       ┌──────────────────────┐
│   FR1–FR5    │       │   Snapshot Probes    │
│  (System)    │       │ (fr1~, fr2~, fr5~)   │
└──────┬──────┘       └────────┬─────────────┘
       │   JSON snapshot write   │
       ▼                         ▼
packages/dashboard/public/  ←  *live.json files*
       │
       ▼
React Web Dashboard (Vite dev server)

📁 Directory Structure
packages/dashboard
│
├── README.md              # 현재 문서
├── vite.config.ts         # Vite dev 서버 설정(host=0.0.0.0)
├── public/
│   ├── fr1_live.json      # FR1 최신 상태 스냅샷
│   ├── fr2_live.json      # FR2 최신 상태 스냅샷
│   ├── fr3_live.json      # FR3 최신 상태 스냅샷
│   ├── fr4_live.json      # FR4 최신 상태 스냅샷
│   └── fr5_live.json      # FR5 정책 엔진 결정 스냅샷
│
├── src/
│   ├── main.tsx               # React App entry
│   ├── App.tsx                # 상단 탭 네비게이션 등
│   ├── pages/
│   │   ├── FR1HomePage.tsx
│   │   ├── FR2ContextPage.tsx
│   │   ├── FR3SequentialGRUPage.tsx
│   │   ├── FR4FederatedLearningPage.tsx
│   │   └── FR5PolicyDecisionPage.tsx
│   │
│   └── components/            # BarChart, StatCard, TimelineItem 등
│
└── package.json

🔥 Live Data Flow per FR Module

대시보드는 각 FR 모듈과 직접 소켓 통신하지 않는다.
모든 데이터는 **Probe(프로브)**가 생성한 JSON 스냅샷을 통해 반영된다.

FR1 — Home Structure & Location Intelligence
구성 요소	내용
입력	Region Learner, Mobile Tracker 등
프로브 위치	packages/ai/realtime/fr1_dashboard_probe.py
출력 파일	public/fr1_live.json
UI	존 정보, 센서 상태, 최근 이벤트
FR2 — Visual & Audio Context Awareness
구성 요소	내용
입력	YOLO(시각), YAMNet(음성), TimeSync
프로브 위치	packages/ai/realtime/fr2_dashboard_probe.py
출력 파일	public/fr2_live.json
UI	감지된 객체, 음성 태그, 지연/동기화 타임라인
FR3 — Sequential GRU Predictor
구성 요소	내용
입력	FR1/FR2 융합 Context Vector
프로브 위치	packages/ai/realtime/fr3_dashboard_probe.py
출력 파일	public/fr3_live.json
UI	Attention Timeline, 예측 확률, 요약된 Context 흐름
FR4 — Personalized Federated Learning (FedPer)
구성 요소	내용
입력	FL Server/Client 이벤트 로그
로그 파일	packages/federated/logs/fl_events.log.jsonl
프로브 위치	packages/federated/realtime/fr4_dashboard_probe.py
출력 파일	public/fr4_live.json
UI	round 진행 상황, 클라이언트 상태, loss 변화, 통신 지연
FR5 — Policy Engine & Cleaning Decision
구성 요소	내용
입력	PolicyBridge에서 기록하는 정책 결정 로그
로그 파일	packages/gateway/logs/policy_events.log.jsonl
프로브 위치	packages/gateway/fr5_dashboard_probe.py
출력 파일	public/fr5_live.json
UI	행동(action), 이유(reason), ETA, 배터리, 경로/오염도 지도
🛠️ Development & Run
1️⃣ Install dependencies
cd packages/dashboard
npm install

2️⃣ Start Vite dev server

WSL2 환경에서는 반드시 host: "0.0.0.0" 설정이 필요하다.

npm run dev


출력:

  ➜  Local:   http://localhost:5174/
  ➜  Network: http://<WSL-IP>:5174/


Windows 브라우저에서는 WSL IP로 접속해야 한다:

http://172.20.x.x:5174/

3️⃣ Run Probes (FR1~FR5)

예:

python -m packages.federated.realtime.fr4_dashboard_probe
python -m packages.gateway.fr5_dashboard_probe


각 Probe는 1초 간격으로 public/*.json 파일을 자동 업데이트한다.

✨ Adding a New Module

대시보드에 새 모듈을 추가하는 방법:

/packages/dashboard/public/<module>_live.json 생성

모듈 프로브에서 해당 JSON 파일을 주기적으로 업데이트

React pages/에 새로운 UI 페이지 추가

App.tsx 탭에 포함

대시보드는 네트워크 기술(MQTT/ZeroMQ/HTTP)에 의존하지 않기 때문에
새 모듈을 추가해도 UI 안정성이 높다.

🎯 Key Design Principles

대시보드는 백엔드/AI 모듈과 직접 통신하지 않음
→ 시스템이 복잡해져도 UI는 항상 단순하고 안전함

Probe가 모든 책임을 가짐
→ ZeroMQ/MQTT/DB/모듈 내부 코드 변경도 Probe만 수정하면 UI 유지됨

JSON 기반 스냅샷 구조
→ React 빌드 없이도 파일만 덮어쓰면 최신 데이터를 볼 수 있음

각 FR 모듈의 상태를 독립적으로 모니터링 가능
→ 로봇 내부 AI 파이프라인의 흐름을 전체적으로 이해하기 쉬움