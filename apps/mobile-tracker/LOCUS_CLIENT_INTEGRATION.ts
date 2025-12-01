/**
 * LOCUS_CLIENT WebSocket 연동 예제
 * 
 * 이 파일을 LOCUS_CLIENT 프로젝트에 추가하세요.
 */

// 타입 정의
interface Position3D {
  x: number;
  y: number;
  z: number;
}

interface LocationUpdateData {
  clientId: number;
  latitude: number;
  longitude: number;
  accuracy: number;
  timestamp: string;
  receivedAt: string;
  position3D: Position3D;
  altitude: number | null;
  heading: number | null;
  speed: number | null;
}

interface LocationUpdateMessage {
  type: 'location_update';
  data: LocationUpdateData;
}

interface WelcomeMessage {
  type: 'welcome';
  clientId: number;
  message: string;
  serverTime: string;
}

type WebSocketMessage = LocationUpdateMessage | WelcomeMessage;

// WebSocket 연결 관리 클래스
class RobotTracker {
  private ws: WebSocket | null = null;
  private robotMarker: THREE.Object3D | null = null;
  private onLocationUpdate?: (data: LocationUpdateData) => void;

  constructor(
    robotMarker: THREE.Object3D,
    onLocationUpdate?: (data: LocationUpdateData) => void
  ) {
    this.robotMarker = robotMarker;
    this.onLocationUpdate = onLocationUpdate;
  }

  // WebSocket 연결
  connect(url: string) {
    if (this.ws) {
      console.warn('WebSocket already connected');
      return;
    }

    console.log(`연결 시도: ${url}`);
    this.ws = new WebSocket(url);

    this.ws.onopen = () => {
      console.log('✅ WebSocket 연결 성공');

      // 뷰어로 식별
      this.send({
        type: 'identify',
        clientType: 'viewer',
      });
    };

    this.ws.onmessage = (event) => {
      try {
        const message: WebSocketMessage = JSON.parse(event.data);
        this.handleMessage(message);
      } catch (error) {
        console.error('메시지 파싱 오류:', error);
      }
    };

    this.ws.onerror = (error) => {
      console.error('❌ WebSocket 오류:', error);
    };

    this.ws.onclose = () => {
      console.log('🔌 WebSocket 연결 끊김');
      this.ws = null;
    };
  }

  // WebSocket 연결 해제
  disconnect() {
    if (this.ws) {
      this.ws.close();
      this.ws = null;
      console.log('WebSocket 연결 해제');
    }
  }

  // 메시지 전송
  private send(message: any) {
    if (this.ws && this.ws.readyState === WebSocket.OPEN) {
      this.ws.send(JSON.stringify(message));
    }
  }

  // 메시지 처리
  private handleMessage(message: WebSocketMessage) {
    switch (message.type) {
      case 'welcome':
        console.log(`환영 메시지: ${message.message}`);
        break;

      case 'location_update':
        this.handleLocationUpdate(message.data);
        break;

      default:
        console.log('알 수 없는 메시지 타입:', message);
    }
  }

  // 위치 업데이트 처리
  private handleLocationUpdate(data: LocationUpdateData) {
    const { position3D, latitude, longitude, accuracy } = data;

    console.log('📍 위치 업데이트:', {
      GPS: `(${latitude.toFixed(6)}, ${longitude.toFixed(6)})`,
      '3D': `(${position3D.x.toFixed(2)}, ${position3D.y.toFixed(2)}, ${position3D.z.toFixed(2)})`,
      정확도: `±${accuracy.toFixed(1)}m`,
    });

    // 로봇 마커 위치 업데이트
    if (this.robotMarker) {
      this.robotMarker.position.set(position3D.x, position3D.y, position3D.z);
    }

    // 콜백 실행
    if (this.onLocationUpdate) {
      this.onLocationUpdate(data);
    }
  }

  // 연결 상태 확인
  isConnected(): boolean {
    return this.ws !== null && this.ws.readyState === WebSocket.OPEN;
  }
}

// 사용 예제
export function initRobotTracking(
  scene: THREE.Scene,
  robotMarker: THREE.Object3D
) {
  // RobotTracker 인스턴스 생성
  const tracker = new RobotTracker(
    robotMarker,
    (data) => {
      // 위치 업데이트 시 추가 작업
      console.log('로봇 위치:', data.position3D);

      // 예: 이동 경로 기록
      // trackHistory.push(data.position3D);

      // 예: UI 업데이트
      // updateLocationUI(data);
    }
  );

  // WebSocket 서버 연결
  // ngrok URL 또는 로컬 URL 사용
  tracker.connect('wss://your-server.ngrok-free.app');
  // 또는
  // tracker.connect('ws://localhost:8080'); // HTTPS가 아닌 경우만

  // 페이지 언로드 시 연결 해제
  window.addEventListener('beforeunload', () => {
    tracker.disconnect();
  });

  return tracker;
}

// React Three Fiber 사용 시
export function RobotTrackerComponent({ robotRef }: { robotRef: React.RefObject<THREE.Object3D> }) {
  React.useEffect(() => {
    if (!robotRef.current) return;

    const tracker = new RobotTracker(robotRef.current);
    tracker.connect('wss://your-server.ngrok-free.app');

    return () => {
      tracker.disconnect();
    };
  }, [robotRef]);

  return null;
}

// LOCUS_CLIENT 기존 코드에 통합하는 방법
export function integrateWithLocusClient(
  scene: THREE.Scene,
  camera: THREE.Camera,
  robotMarker: THREE.Object3D
) {
  const tracker = new RobotTracker(robotMarker, (data) => {
    // 카메라가 로봇 따라가기 (선택사항)
    // camera.lookAt(robotMarker.position);

    // 로봇 회전 업데이트 (heading이 있는 경우)
    if (data.heading !== null) {
      robotMarker.rotation.y = (data.heading * Math.PI) / 180;
    }

    // 정확도에 따라 마커 크기 조정 (선택사항)
    const scale = Math.max(0.5, 1 - data.accuracy / 100);
    robotMarker.scale.setScalar(scale);
  });

  // 연결
  tracker.connect('wss://your-server.ngrok-free.app');

  // 전역에서 접근 가능하도록 (디버깅용)
  (window as any).robotTracker = tracker;

  return tracker;
}

export default RobotTracker;
