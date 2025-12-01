import { useCallback, useEffect, useState } from 'react';
import { StatusIndicator } from './components/StatusIndicator';
import { ServerConfig } from './components/ServerConfig';
import { LocationInfo } from './components/LocationInfo';
import { ControlButtons } from './components/ControlButtons';
import { StatsDisplay } from './components/StatsDisplay';
import { LogViewer } from './components/LogViewer';
import { useWebSocket } from './hooks/useWebSocket';
import { useGeolocation } from './hooks/useGeolocation';
import { useLogger } from './hooks/useLogger';
import { LocationData, Stats, WebSocketMessage } from './types';
import { getGeolocationErrorMessage, isGeolocationSupported } from './utils/format';
import './styles/App.css';

function App() {
  const [stats, setStats] = useState<Stats>({
    sentCount: 0,
    errorCount: 0,
    startTime: null,
  });

  const { logs, addLog, clearLogs } = useLogger();

  const handleWebSocketMessage = useCallback(
    (message: WebSocketMessage) => {
      if (message.type === 'welcome') {
        addLog('success', `서버 연결 성공: ${message.message}`);
      } else if (message.type === 'pong') {
        addLog('info', 'Pong 수신');
      } else if (message.type === 'server_shutdown') {
        addLog('warning', message.message);
      } else {
        addLog('info', `서버 메시지: ${message.type}`);
      }
    },
    [addLog]
  );

  const {
    status: connectionStatus,
    connect,
    disconnect,
    send,
    isConnected,
  } = useWebSocket({
    onMessage: handleWebSocketMessage,
    onOpen: () => {
      addLog('success', '서버 연결 성공');
      // 트래커로 식별
      send({
        type: 'identify',
        clientType: 'tracker',
      });
    },
    onClose: () => {
      addLog('info', '서버 연결이 끊어졌습니다');
    },
    onError: () => {
      addLog('error', 'WebSocket 연결 오류');
      setStats((prev) => ({ ...prev, errorCount: prev.errorCount + 1 }));
    },
  });

  const handlePositionUpdate = useCallback(
    (position: GeolocationPosition) => {
      const { latitude, longitude, accuracy, altitude, heading, speed } = position.coords;

      // 서버로 전송
      if (isConnected) {
        const locationData: LocationData = {
          type: 'location',
          latitude,
          longitude,
          accuracy,
          timestamp: new Date(position.timestamp).toISOString(),
          altitude,
          heading,
          speed,
        };

        const success = send(locationData);
        if (success) {
          setStats((prev) => ({ ...prev, sentCount: prev.sentCount + 1 }));
          addLog(
            'success',
            `위치 전송: (${latitude.toFixed(4)}, ${longitude.toFixed(4)})`
          );
        } else {
          setStats((prev) => ({ ...prev, errorCount: prev.errorCount + 1 }));
          addLog('error', '위치 전송 실패: WebSocket 연결 안 됨');
        }
      }
    },
    [isConnected, send, addLog]
  );

  const handlePositionError = useCallback(
    (error: GeolocationPositionError) => {
      const message = getGeolocationErrorMessage(error);
      addLog('error', `위치 오류: ${message}`);
      setStats((prev) => ({ ...prev, errorCount: prev.errorCount + 1 }));
    },
    [addLog]
  );

  const {
    status: trackingStatus,
    position,
    startTracking,
    stopTracking,
  } = useGeolocation({
    onPosition: handlePositionUpdate,
    onError: handlePositionError,
  });

  // 앱 초기화
  useEffect(() => {
    if (!isGeolocationSupported()) {
      addLog('error', '이 브라우저는 Geolocation을 지원하지 않습니다');
    } else {
      addLog('info', 'LOCUS Mobile Tracker 시작');
    }
  }, [addLog]);

  const handleConnect = useCallback(
    (url: string) => {
      addLog('info', `연결 시도: ${url}`);
      connect(url);
    },
    [connect, addLog]
  );

  const handleDisconnect = useCallback(() => {
    if (trackingStatus === 'tracking') {
      stopTracking();
    }
    disconnect();
  }, [disconnect, stopTracking, trackingStatus]);

  const handleStartTracking = useCallback(() => {
    addLog('info', '위치 추적 시작');
    setStats((prev) => ({ ...prev, startTime: new Date() }));
    startTracking();
  }, [startTracking, addLog]);

  const handleStopTracking = useCallback(() => {
    addLog('info', '위치 추적 중지');
    stopTracking();
  }, [stopTracking, addLog]);

  return (
    <div className="container">
      <header>
        <h1>🏠 LOCUS Tracker</h1>
        <StatusIndicator
          connectionStatus={connectionStatus}
          trackingStatus={trackingStatus}
        />
      </header>

      <main>
        <ServerConfig
          onConnect={handleConnect}
          onDisconnect={handleDisconnect}
          status={connectionStatus}
        />

        <LocationInfo position={position} />

        <ControlButtons
          trackingStatus={trackingStatus}
          onStart={handleStartTracking}
          onStop={handleStopTracking}
          disabled={!isConnected}
        />

        <StatsDisplay stats={stats} />

        <LogViewer logs={logs} onClear={clearLogs} />
      </main>
    </div>
  );
}

export default App;
