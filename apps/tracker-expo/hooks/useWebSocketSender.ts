// useWebSocketSender.ts

import { useEffect, useRef, useState } from 'react';

export interface ARPosition {
  x: number;
  y: number;
  z: number;
}

interface WebSocketConfig {
  serverUrl: string;
  autoConnect?: boolean;
  reconnectInterval?: number; // ms
}

/**
 * iOS → 서버로 보내는 메시지 포맷
 *  - type: 'arkit_location'
 *  - data: { position3D, accuracy, timestamp }
 */
interface ARKitLocationMessage {
  type: 'arkit_location';
  data: {
    position3D: ARPosition;
    accuracy: number;   // m 단위라고 가정
    timestamp: number;  // Date.now()
  };
}

export function useWebSocketSender(config: WebSocketConfig) {
  const {
    serverUrl,
    autoConnect = true,
    reconnectInterval = 3000,
  } = config;

  const ws = useRef<WebSocket | null>(null);
  const [isConnected, setIsConnected] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const reconnectTimer = useRef<ReturnType<typeof setTimeout> | null>(null);

  // 🔥 여기부터가 교체할 connect 함수
  const connect = () => {
    if (ws.current?.readyState === WebSocket.OPEN) {
      console.log('⚠️ 이미 연결되어 있습니다');
      return;
    }

    try {
      // http/https 들어와도 ws/wss로 자동 변환
      const wsUrl = serverUrl.startsWith('http')
        ? serverUrl.replace(/^http/, 'ws')
        : serverUrl;

      console.log(`🔌 서버 연결 시도: ${wsUrl}`);
      ws.current = new WebSocket(wsUrl);

      ws.current.onopen = () => {
        console.log('✅ 서버 연결 성공');
        setIsConnected(true);
        setError(null);

        // 이 클라이언트가 tracker(폰)이라는 걸 알림
        ws.current?.send(
          JSON.stringify({
            type: 'identify',
            clientType: 'tracker',
          }),
        );
      };

      ws.current.onmessage = (event) => {
        try {
          const message = JSON.parse(event.data);
          console.log('📨 서버 메시지:', message.type ?? message);
        } catch (err) {
          console.error('메시지 파싱 오류:', err);
        }
      };

      ws.current.onerror = (event) => {
        console.error('❌ WebSocket 오류:', event);
        setError('연결 오류가 발생했습니다');
      };

      ws.current.onclose = () => {
        console.log('🔌 서버 연결 끊김');
        setIsConnected(false);

        // 자동 재연결
        if (reconnectTimer.current) {
          clearTimeout(reconnectTimer.current);
        }
        reconnectTimer.current = setTimeout(() => {
          console.log('🔄 재연결 시도...');
          connect();
        }, reconnectInterval);
      };
    } catch (err) {
      const errorMsg = `연결 실패: ${err}`;
      console.error(errorMsg);
      setError(errorMsg);
    }
  };
  // 🔥 여기까지가 교체

  const disconnect = () => {
    if (reconnectTimer.current) {
      clearTimeout(reconnectTimer.current);
      reconnectTimer.current = null;
    }

    if (ws.current) {
      ws.current.close();
      ws.current = null;
      setIsConnected(false);
    }
  };

  // ARKit 3D 위치 데이터 전송
  const sendPosition = (position: ARPosition, accuracy: number) => {
    if (!isConnected || ws.current?.readyState !== WebSocket.OPEN) {
      console.warn('⚠️ 서버에 연결되지 않음');
      return;
    }

    const message: ARKitLocationMessage = {
      type: 'arkit_location',
      data: {
        position3D: position,
        accuracy,
        timestamp: Date.now(),
      },
    };

    try {
      ws.current.send(JSON.stringify(message));
      // console.log('📤 ARKit 위치 전송:', message.data);
    } catch (err) {
      console.error('전송 실패:', err);
    }
  };

  // 자동 연결
  useEffect(() => {
    if (autoConnect) {
      connect();
    }

    return () => {
      disconnect();
    };
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [serverUrl, autoConnect]);

  return {
    isConnected,
    error,
    connect,
    disconnect,
    sendPosition,
  };
}
