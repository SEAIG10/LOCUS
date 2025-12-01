/**
 * src/server.ts
 * Fastify + Socket.IO + MQTT 통합 실행
 */
import { buildApp } from './app';
import { env } from './config/env';
import { connectDB } from './config/db';
import { Server as SocketIOServer } from 'socket.io'; // ✅ 추가
import { connectMQTT, disconnectMQTT } from './modules/mqtt/mqtt.service';
import { initSocket } from './modules/socket/socket.service'; // ✅ 추가

async function start() {
  const app = buildApp();

  try {
    // 1. DB 연결
    await connectDB();

    // 2. Fastify 플러그인 로딩 대기 (중요: app.server에 접근하기 위해 필요)
    await app.ready();

    // 3. Socket.IO 서버 생성 및 Fastify HTTP 서버에 부착
    // cors: origin '*'은 개발용입니다. 배포 시 보안 설정 필요.
    const io = new SocketIOServer(app.server, {
      cors: {
        origin: "*", 
        methods: ["GET", "POST"]
      }
    });

    // 4. 소켓 서비스 초기화 (EventBus 구독 시작)
    initSocket(io);

    // 5. MQTT 서비스 시작 (DB저장 -> EventBus 발행 시작)
    connectMQTT();

    // 6. 서버 리슨 시작
    await app.listen({ port: env.PORT, host: '0.0.0.0' });
    console.log('='.repeat(50));
    console.log(`🚀 Server running on http://localhost:${env.PORT}`);
    console.log(`🔌 Socket.IO is attached to the same port`);
    console.log('='.repeat(50));

  } catch (err) {
    app.log.error(err);
    process.exit(1);
  }
}

// Graceful shutdown
const shutdown = () => {
  console.log('Shutting down gracefully...');
  disconnectMQTT();
  process.exit(0);
};

process.on('SIGTERM', shutdown);
process.on('SIGINT', shutdown);

start();