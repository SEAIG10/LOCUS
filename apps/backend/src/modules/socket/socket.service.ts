import { Server as SocketIOServer, Socket } from 'socket.io';
import { eventBus, EVENTS } from '../../lib/eventBus';

let io: SocketIOServer | null = null;

export const initSocket = (socketIoInstance: SocketIOServer) => {
  io = socketIoInstance;

  io.on('connection', (socket: Socket) => {
    console.log(`🔌 [Socket] Client connected: ${socket.id}`);

    socket.on('join_home', (homeId: string) => {
      const roomName = `home_${homeId}`;
      socket.join(roomName);
      console.log(`[Socket] ${socket.id} joined ${roomName}`);
    });

    socket.on('disconnect', () => {
      console.log(`🔌 [Socket] Client disconnected: ${socket.id}`);
    });
  });

  // ============================================================
  // 🔥 [EventBus Listener] 내부 이벤트 수신 -> 프론트엔드 전송
  // ============================================================

  // 1. 오염 예측 데이터 (MQTT -> DB -> EventBus -> 여기)
  eventBus.on(EVENTS.NEW_POLLUTION_PREDICTION, ({ homeId, data }) => {
    if (io) {
      io.to(`home_${homeId}`).emit('pollution_update', data);
      console.log(`📡 [Socket] Pushed pollution_update to home_${homeId}`);
    }
  });

  // 2. 센서/청소 이벤트 (MQTT -> DB -> EventBus -> 여기)
  eventBus.on(EVENTS.NEW_SENSOR_EVENT, ({ homeId, data }) => {
    if (io) {
      io.to(`home_${homeId}`).emit('new_event', data);
      console.log(`📡 [Socket] Pushed new_event to home_${homeId}`);
    }
  });

  // 3. 로봇 위치 정보 (Mobile App -> HTTP -> EventBus -> 여기)
  eventBus.on(EVENTS.NEW_ROBOT_LOCATION, (record) => {
    if (io) {
      // 위치 정보는 'robot_position' 이벤트로 전체 브로드캐스트
      // (특정 homeId 룸으로 보내려면 record에 homeId가 포함되어야 함)
      io.emit('robot_position', record);
      // console.log(`📡 [Socket] Pushed robot position`); // 로그 너무 많으면 주석 처리
    }
  });
};