import { WebSocket } from 'ws';

export type MessageType =
  | 'location'          // (기존 GPS, 지금은 안 씀)
  | 'arkit_location'    // ✅ ARKit용
  | 'identify'
  | 'ping'
  | 'pong'
  | 'welcome'
  | 'location_update'
  | 'server_shutdown';

export type ClientType = 'tracker' | 'viewer' | 'unknown';

/**
 * 기존 GPS 기반 위치 데이터
 *  - 지금은 안 쓰지만, 혹시 나중에 다시 쓸 수도 있으니 유지
 */
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

/**
 * ARKit에서 바로 오는 3D 좌표용 메시지
 */
export interface ARKitLocationData {
  type: 'arkit_location';
  data: {
    position3D: Position3D;
    accuracy: number;
    timestamp: number; // ms
  };
}

export interface IdentifyMessage {
  type: 'identify';
  clientType: ClientType;
}

export interface PingMessage {
  type: 'ping';
}

export interface PongMessage {
  type: 'pong';
  timestamp: string;
}

export interface WelcomeMessage {
  type: 'welcome';
  clientId: number;
  message: string;
  serverTime: string;
}

export interface Position3D {
  x: number;
  y: number;
  z: number;
}

/**
 * 뷰어/히스토리에서 공통으로 쓰는 위치 데이터 형태
 *  - 최소 요구: clientId, receivedAt, position3D
 *  - GPS 관련 필드는 있어도 되고 없어도 됨
 */
export interface LocationUpdatePayload {
  clientId: number;
  receivedAt: string;
  position3D: Position3D;

  // 아래는 선택
  latitude?: number;
  longitude?: number;
  accuracy?: number;
  timestamp?: string | number;
  altitude?: number | null;
  heading?: number | null;
  speed?: number | null;
}

export interface LocationUpdateMessage {
  type: 'location_update';
  data: LocationUpdatePayload;
}

export interface ServerShutdownMessage {
  type: 'server_shutdown';
  message: string;
}

export type WebSocketMessage =
  | LocationData          // legacy GPS
  | ARKitLocationData     // ✅ ARKit 3D
  | IdentifyMessage
  | PingMessage
  | PongMessage
  | WelcomeMessage
  | LocationUpdateMessage
  | ServerShutdownMessage;

export interface ClientInfo {
  id: number;
  ws: WebSocket;
  type: ClientType;
  connectedAt: Date;
  ip: string;
  lastActivity: Date;
}

/**
 * 서버 내부에서 쓰는 히스토리 타입
 *  - LocationUpdatePayload와 동일
 */
export interface LocationRecord extends LocationUpdatePayload {}

export interface ServerStats {
  totalConnections: number;
  currentConnections: number;
  messagesReceived: number;
  messagesSent: number;
  startTime: Date;
}
