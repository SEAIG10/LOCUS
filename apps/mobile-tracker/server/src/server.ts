import { WebSocketServer, WebSocket } from 'ws';
import express from 'express';
import { createServer } from 'http';
import axios from 'axios';
import {
  ClientInfo,
  LocationRecord,
  ServerStats,
  WebSocketMessage,
  LocationData,
  ARKitLocationData,
  LocationUpdateMessage,
} from './types.js';

// --- 🔥 [설정] 메인 백엔드 주소 (LocusBackend) ---
// 이제 배치가 아니라 '단건'으로 보냅니다. 백엔드 쪽 API 주소가 달라질 수 있습니다.
// 예: /api/log/single 또는 기존 /api/log/batch가 배열을 받는지 확인 필요
const MAIN_BACKEND_URL = process.env.MAIN_BACKEND_URL || 'http://localhost:4000/api/log/record';

// Express 앱 설정
const app = express();
const server = createServer(app);

// WebSocket 서버 설정
const wss = new WebSocketServer({ server });

// 연결된 클라이언트 관리
const clients = new Map<number, ClientInfo>();
let clientIdCounter = 0;

// 위치 데이터 저장 (최근 100개 - 실시간 뷰어용)
const locationHistory: LocationRecord[] = [];
const MAX_HISTORY = 100;

// 통계
const stats: ServerStats = {
  totalConnections: 0,
  currentConnections: 0,
  messagesReceived: 0,
  messagesSent: 0,
  startTime: new Date(),
};

// JSON 파싱 미들웨어
app.use(express.json());

// 상태 확인 엔드포인트
app.get('/status', (req, res) => {
  res.json({
    status: 'running',
    stats: {
      ...stats,
      uptime: Math.floor(
        (new Date().getTime() - stats.startTime.getTime()) / 1000,
      ),
      currentConnections: wss.clients.size,
      locationHistoryCount: locationHistory.length,
    },
    clients: Array.from(clients.values()).map((client) => ({
      id: client.id,
      type: client.type,
      connectedAt: client.connectedAt,
    })),
  });
});

// 위치 히스토리 조회 엔드포인트
app.get('/locations', (req, res) => {
  const limit = parseInt(req.query.limit as string) || 50;
  res.json({
    count: locationHistory.length,
    locations: locationHistory.slice(-limit),
  });
});

// ==========================================
// 🔥 [수정] 즉시 전송 함수 (Fire and Forget)
// ==========================================
async function forwardToBackend(record: LocationRecord) {
  try {
    // await를 쓰지 않고 프로미스 체인으로 처리하여 WS 루프 지연 방지 (Fire & Forget)
    // 백엔드 API가 단건 객체를 받도록 설계되어야 함
    axios.post(MAIN_BACKEND_URL, record).catch((error) => {
      // 전송 실패 로그는 남기되, 서버를 멈추지는 않음
      console.error(`❌ [Relay] 백엔드 전송 실패: ${error.message}`);
    });
  } catch (error: any) {
    console.error(`❌ [Relay] 전송 오류: ${error.message}`);
  }
}

// WebSocket 연결 처리
wss.on('connection', (ws: WebSocket, req) => {
  const clientId = ++clientIdCounter;
  const clientIp = req.socket.remoteAddress || 'unknown';

  const clientInfo: ClientInfo = {
    id: clientId,
    ws,
    type: 'unknown',
    connectedAt: new Date(),
    ip: clientIp,
    lastActivity: new Date(),
  };

  clients.set(clientId, clientInfo);
  stats.totalConnections++;
  stats.currentConnections++;

  console.log(`\n[연결] 클라이언트 #${clientId} (${clientIp})`);
  console.log(`현재 연결: ${stats.currentConnections}개`);

  // 환영 메시지
  sendMessage(ws, {
    type: 'welcome',
    clientId,
    message: 'LOCUS Tracker Server에 연결되었습니다.',
    serverTime: new Date().toISOString(),
  });

  // 메시지 수신 처리
  ws.on('message', (data: Buffer) => {
    try {
      const raw = JSON.parse(data.toString()) as WebSocketMessage;
      clientInfo.lastActivity = new Date();
      stats.messagesReceived++;

      switch (raw.type) {
        case 'arkit_location':
          handleARKitLocation(clientId, raw as ARKitLocationData);
          break;

        case 'location':
          handleLegacyLocation(clientId, raw as LocationData);
          break;

        case 'identify':
          clientInfo.type = raw.clientType || 'unknown';
          console.log(`클라이언트 #${clientId} 타입: ${clientInfo.type}`);
          break;

        case 'ping':
          sendMessage(ws, {
            type: 'pong',
            timestamp: new Date().toISOString(),
          });
          break;

        default:
          console.log(`알 수 없는 메시지 타입: ${raw.type}`);
      }
    } catch (error) {
      const err = error as Error;
      console.error(
        `[오류] 메시지 파싱 실패 (클라이언트 #${clientId}):`,
        err.message,
      );
    }
  });

  // 연결 종료 처리
  ws.on('close', () => {
    clients.delete(clientId);
    stats.currentConnections--;
    console.log(`\n[종료] 클라이언트 #${clientId}`);
    console.log(`현재 연결: ${stats.currentConnections}개`);
  });

  // 에러 처리
  ws.on('error', (error) => {
    console.error(`[오류] 클라이언트 #${clientId}:`, error.message);
  });

  // Heartbeat
  (ws as any).isAlive = true;
  ws.on('pong', () => {
    (ws as any).isAlive = true;
  });
});

/**
 * ✅ ARKit 3D 위치 처리
 */
function handleARKitLocation(clientId: number, message: ARKitLocationData) {
  const {
    position3D,
    accuracy,
    timestamp,
  } = message.data;

  const record: LocationRecord = {
    clientId,
    receivedAt: new Date().toISOString(),
    position3D,
    accuracy,
    timestamp,
  };

  // 1. [실시간] 뷰어용 히스토리 저장 (메모리)
  locationHistory.push(record);
  if (locationHistory.length > MAX_HISTORY) {
    locationHistory.shift();
  }

  // 2. [실시간] 뷰어들에게 브로드캐스트
  const updateMessage: LocationUpdateMessage = {
    type: 'location_update',
    data: record,
  };
  broadcastToViewers(updateMessage);

  // 3. 🔥 [릴레이] 백엔드로 즉시 전송
  forwardToBackend(record);
}

/**
 * (선택) 옛날 GPS 기반 tracker 처리
 */
function handleLegacyLocation(clientId: number, locationData: LocationData) {
  const {
    latitude,
    longitude,
    accuracy,
    timestamp,
    altitude,
    heading,
    speed,
  } = locationData;

  const record: LocationRecord = {
    clientId,
    receivedAt: new Date().toISOString(),
    position3D: { x: 0, y: 0, z: 0 },
    latitude,
    longitude,
    accuracy,
    timestamp,
    altitude,
    heading,
    speed,
  };

  locationHistory.push(record);
  if (locationHistory.length > MAX_HISTORY) {
    locationHistory.shift();
  }

  const updateMessage: LocationUpdateMessage = {
    type: 'location_update',
    data: record,
  };

  broadcastToViewers(updateMessage);
  
  // 🔥 [릴레이] 백엔드로 즉시 전송
  forwardToBackend(record);
}

// 뷰어 클라이언트들에게 브로드캐스트
function broadcastToViewers(message: LocationUpdateMessage) {
  let sentCount = 0;

  clients.forEach((client) => {
    if (client.type === 'viewer' || client.type === 'unknown') {
      if (client.ws.readyState === WebSocket.OPEN) {
        sendMessage(client.ws, message);
        sentCount++;
      }
    }
  });
}

// 메시지 전송 헬퍼
function sendMessage(ws: WebSocket, message: WebSocketMessage | any) {
  if (ws.readyState === WebSocket.OPEN) {
    ws.send(JSON.stringify(message));
    stats.messagesSent++;
  }
}

// 주기적으로 연결 상태 확인 (heartbeat)
const heartbeatInterval = setInterval(() => {
  wss.clients.forEach((ws) => {
    const typedWs = ws as any;
    if (typedWs.isAlive === false) {
      return ws.terminate();
    }

    typedWs.isAlive = false;
    ws.ping();
  });
}, 30000); // 30초마다

wss.on('close', () => {
  clearInterval(heartbeatInterval);
});

// 서버 시작
const PORT = process.env.PORT || 8080;

server.listen(PORT, () => {
  console.log('='.repeat(50));
  console.log('🚀 LOCUS Tracker Relay Server');
  console.log('='.repeat(50));
  console.log(`✅ WebSocket 서버: ws://0.0.0.0:${PORT}`);
  console.log(`✅ HTTP 상태 확인: http://0.0.0.0:${PORT}/status`);
  console.log(`🔗 Main Backend: ${MAIN_BACKEND_URL}`);
  console.log('='.repeat(50));
  console.log('\n대기 중...\n');
});

// 종료 처리
process.on('SIGINT', () => {
  console.log('\n\n서버 종료 중...');

  const shutdownMessage: WebSocketMessage = {
    type: 'server_shutdown',
    message: '서버가 종료됩니다.',
  } as any;

  wss.clients.forEach((ws) => {
    sendMessage(ws, shutdownMessage);
    ws.close();
  });

  server.close(() => {
    console.log('서버가 정상적으로 종료되었습니다.');
    process.exit(0);
  });

  setTimeout(() => {
    console.error('강제 종료됩니다.');
    process.exit(1);
  }, 5000);
});