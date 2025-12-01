// WebSocket 메시지 타입
export type MessageType = 
  | 'location'
  | 'identify'
  | 'ping'
  | 'pong'
  | 'welcome'
  | 'location_update'
  | 'server_shutdown';

// 위치 데이터
export interface LocationData {
  type: 'location';
  latitude: number;
  longitude: number;
  accuracy: number;
  timestamp: string;
  altitude: number | null;
  heading: number | null;
  speed: number | null;
}

// 클라이언트 식별
export interface IdentifyMessage {
  type: 'identify';
  clientType: 'tracker' | 'viewer';
}

// Ping/Pong
export interface PingMessage {
  type: 'ping';
}

export interface PongMessage {
  type: 'pong';
  timestamp: string;
}

// 환영 메시지
export interface WelcomeMessage {
  type: 'welcome';
  clientId: number;
  message: string;
  serverTime: string;
}

// 위치 업데이트 브로드캐스트
export interface LocationUpdateMessage {
  type: 'location_update';
  data: LocationData & {
    clientId: number;
    receivedAt: string;
  };
}

// 서버 종료 알림
export interface ServerShutdownMessage {
  type: 'server_shutdown';
  message: string;
}

// 모든 메시지 타입
export type WebSocketMessage =
  | LocationData
  | IdentifyMessage
  | PingMessage
  | PongMessage
  | WelcomeMessage
  | LocationUpdateMessage
  | ServerShutdownMessage;

// 연결 상태
export type ConnectionStatus = 
  | 'disconnected'
  | 'connecting'
  | 'connected'
  | 'error';

// 추적 상태
export type TrackingStatus = 
  | 'idle'
  | 'tracking'
  | 'paused'
  | 'error';

// 로그 레벨
export type LogLevel = 'info' | 'success' | 'error' | 'warning';

// 로그 엔트리
export interface LogEntry {
  id: string;
  level: LogLevel;
  message: string;
  timestamp: Date;
}

// 통계
export interface Stats {
  sentCount: number;
  errorCount: number;
  startTime: Date | null;
}

// Geolocation Position 확장
export interface ExtendedPosition extends GeolocationPosition {
  coords: GeolocationCoordinates;
}
