import React, { useState, useEffect, useRef } from 'react';
import './FR2Page.css';

const WEBSOCKET_URL = 'ws://localhost:8080';

interface YoloDetection {
  label: string;
  confidence: number;
}

interface AudioClass {
  label: string;
  score: number;
}

interface ZmqMessage {
  timestamp: string;
  sensor: string;
  status: string;
}

const FR2Page: React.FC = () => {
  const [yoloDetections, setYoloDetections] = useState<YoloDetection[]>([]);
  const [audioClasses, setAudioClasses] = useState<AudioClass[]>([]);
  const [zmqMessages, setZmqMessages] = useState<ZmqMessage[]>([]);
  const [wsConnected, setWsConnected] = useState(false);
  const wsRef = useRef<WebSocket | null>(null);

  const [stats, setStats] = useState({
    visualMsgCount: 0,
    audioMsgCount: 0,
    syncedCount: 0,
    droppedCount: 0,
    latencyMs: 0
  });

  // WebSocket 연결 및 실시간 데이터 수신
  useEffect(() => {
    const connectWebSocket = () => {
      try {
        const ws = new WebSocket(WEBSOCKET_URL);
        wsRef.current = ws;

        ws.onopen = () => {
          console.log('[FR2 WS] Connected to WebSocket bridge');
          setWsConnected(true);
        };

        ws.onmessage = (event) => {
          try {
            const message = JSON.parse(event.data);

            if (message.type === 'welcome') return;

            const timestamp = new Date().toLocaleTimeString();

            // Visual 센서 데이터 (YOLO)
            if (message.type === 'visual') {
              setStats(prev => ({ ...prev, visualMsgCount: prev.visualMsgCount + 1 }));

              // YOLO detection 시뮬레이션 (실제로는 message.data에서 파싱)
              const detections: YoloDetection[] = [
                { label: 'person', confidence: Math.random() * 0.3 + 0.7 },
                { label: 'object', confidence: Math.random() * 0.3 + 0.5 }
              ];
              setYoloDetections(detections);

              setZmqMessages(prev => [...prev, {
                timestamp,
                sensor: 'Visual',
                status: 'Received'
              }].slice(-5));
            }

            // Audio 센서 데이터 (YAMNet)
            if (message.type === 'audio') {
              setStats(prev => ({ ...prev, audioMsgCount: prev.audioMsgCount + 1 }));

              // YAMNet classification 시뮬레이션
              const classes: AudioClass[] = [
                { label: 'Environment', score: Math.random() * 0.4 + 0.4 },
                { label: 'Speech', score: Math.random() * 0.3 + 0.3 },
                { label: 'Music', score: Math.random() * 0.2 + 0.1 }
              ];
              setAudioClasses(classes);

              setZmqMessages(prev => [...prev, {
                timestamp,
                sensor: 'Audio',
                status: 'Received'
              }].slice(-5));
            }

            // Pose 센서 데이터
            if (message.type === 'pose') {
              setZmqMessages(prev => [...prev, {
                timestamp,
                sensor: 'Pose',
                status: 'Received'
              }].slice(-5));
            }

            // 동기화된 데이터
            if (message.type === 'synced') {
              setStats(prev => ({
                ...prev,
                syncedCount: prev.syncedCount + 1,
                latencyMs: Math.floor(Math.random() * 20 + 30)
              }));

              setZmqMessages(prev => [...prev, {
                timestamp,
                sensor: 'All',
                status: 'Sent to GRU'
              }].slice(-5));
            }

          } catch (err) {
            console.error('[FR2 WS] Message parsing error:', err);
          }
        };

        ws.onerror = (error) => {
          console.error('[FR2 WS] WebSocket error:', error);
          setWsConnected(false);
        };

        ws.onclose = () => {
          console.log('[FR2 WS] Disconnected');
          setWsConnected(false);
          wsRef.current = null;
          setTimeout(connectWebSocket, 3000);
        };

      } catch (err) {
        console.error('[FR2 WS] Connection error:', err);
      }
    };

    connectWebSocket();

    return () => {
      if (wsRef.current) {
        wsRef.current.close();
      }
    };
  }, []);

  return (
    <div className="page">
      <h1 className="page-title" style={{ color: 'var(--accent)' }}>FR2 · Visual & Audio Context Awareness</h1>
      <p className="page-subtitle">
        YOLO 객체 감지 + YAMNet 오디오 분류 실시간 스트림 및 ZeroMQ 동기화
      </p>

      {/* 상단: YOLO + YAMNet */}
      <div className="grid grid-2">
        {/* 왼쪽: YOLO 비디오 스트림 */}
        <div className="card">
          <div className="card-header">YOLO Detection Stream</div>
          <div className="video-container">
            <img
              src="http://localhost:5001/video_feed"
              alt="YOLO Live Stream"
              style={{ width: '100%', height: 'auto', borderRadius: '12px' }}
              onError={(e) => {
                e.currentTarget.style.display = 'none';
                e.currentTarget.nextElementSibling?.setAttribute('style', 'display: flex');
              }}
            />
            <div className="video-placeholder" style={{ display: 'none' }}>
              <p>YOLOv11n-Pose 실시간 스트림 (카메라 연결 대기 중)</p>
            </div>
          </div>
          <div className="detection-list">
            {yoloDetections.map((det, idx) => (
              <div key={idx} className="detection-item">
                <span className="detection-label">{det.label}</span>
                <div className="detection-bar-container">
                  <div
                    className="detection-bar"
                    style={{ width: `${det.confidence * 100}%` }}
                  />
                </div>
                <span className="detection-value">{(det.confidence * 100).toFixed(0)}%</span>
              </div>
            ))}
          </div>
        </div>

        {/* 오른쪽: YAMNet 오디오 분류 */}
        <div className="card">
          <div className="card-header">YAMNet Audio Classification</div>
          <div className="audio-viz">
            {audioClasses.map((audio, idx) => (
              <div key={idx} className="audio-class-item" style={{ height: `${audio.score * 170}px` }}>
                <div className="audio-bar" style={{ height: '100%' }} />
                <div className="audio-label">{audio.label}</div>
                <div className="audio-score">{(audio.score * 100).toFixed(0)}%</div>
              </div>
            ))}
          </div>
        </div>
      </div>

      {/* Bridge connection + ZeroMQ logs */}
      <div className="grid grid-2" style={{ marginTop: '1rem' }}>
        <div className="card">
          <div className="card-header">ZeroMQ Bridge Status</div>
          <div style={{ display: 'flex', alignItems: 'center', gap: '1rem' }}>
            <div style={{
              width: '14px',
              height: '14px',
              borderRadius: '50%',
              background: wsConnected ? 'var(--success)' : 'var(--danger)',
              boxShadow: wsConnected ? '0 0 10px rgba(78,204,163,0.6)' : '0 0 10px rgba(255,99,72,0.5)'
            }}></div>
            <div>
              <div style={{ fontWeight: 600, fontSize: '1.05rem', color: 'var(--text-primary)' }}>
                {wsConnected ? '🟢 WebSocket Connected' : '🔴 Waiting for websocket_bridge.py'}
              </div>
              <div style={{ fontSize: '0.85rem', color: 'var(--text-secondary)' }}>
                {wsConnected
                  ? `Latency ${stats.latencyMs}ms · ${stats.syncedCount} synced batches`
                  : 'ws://localhost:8080 로 연결을 시도하는 중입니다.'}
              </div>
            </div>
          </div>
        </div>

        <div className="card">
          <div className="card-header">Latest ZeroMQ Messages</div>
          {zmqMessages.length === 0 ? (
            <div style={{ color: 'var(--text-secondary)', fontSize: '0.85rem' }}>
              아직 수신된 메시지가 없습니다. 센서를 실행하면 이 영역에 5개의 최근 메시지가 표시됩니다.
            </div>
          ) : (
            <div style={{ display: 'flex', flexDirection: 'column', gap: '0.75rem' }}>
              {zmqMessages.map((msg, idx) => (
                <div key={`${msg.timestamp}-${idx}`} style={{ display: 'flex', justifyContent: 'space-between', fontSize: '0.85rem' }}>
                  <div style={{ color: 'var(--text-secondary)', minWidth: 70 }}>{msg.timestamp}</div>
                  <div style={{ fontWeight: 600, color: 'var(--text-primary)' }}>{msg.sensor}</div>
                  <div style={{ color: msg.status.includes('Sent') ? 'var(--accent)' : 'var(--success)', fontWeight: 600 }}>
                    {msg.status}
                  </div>
                </div>
              ))}
            </div>
          )}
        </div>
      </div>

      {/* 하단: ROS ApproximateTimeSynchronizer 동기화 */}
      <div className="card" style={{ marginTop: '1rem' }}>
        <div className="card-header">ROS ApproximateTimeSynchronizer - Sensor Fusion</div>
        <p style={{ color: 'var(--text-secondary)', marginBottom: '1rem', fontSize: '0.875rem' }}>
          Visual, Audio, Pose 센서 스트림을 <span style={{ color: 'var(--accent)', fontWeight: '600' }}>±500ms 허용 오차</span> 윈도우로 동기화하여 Context Encoder로 전송
        </p>

        {/* 센서 동기화 타임라인 */}
        <div style={{ background: '#f8f9fa', padding: '1.25rem', borderRadius: '12px', marginBottom: '1rem' }}>
          <div style={{ fontFamily: 'monospace', fontSize: '0.85rem', lineHeight: '1.8' }}>
            <div style={{ display: 'flex', alignItems: 'center', gap: '0.5rem' }}>
              <span style={{ width: '80px', color: 'var(--text-secondary)' }}>Visual</span>
              <div style={{ flex: 1, height: '2px', background: '#e0e0e0', position: 'relative' }}>
                <div style={{
                  position: 'absolute',
                  left: '40%',
                  top: '-6px',
                  width: '12px',
                  height: '12px',
                  borderRadius: '50%',
                  background: 'var(--accent)'
                }}></div>
              </div>
              <span style={{ fontSize: '0.8rem', color: 'var(--text-secondary)' }}>{stats.visualMsgCount} msgs</span>
            </div>
            <div style={{ display: 'flex', alignItems: 'center', gap: '0.5rem' }}>
              <span style={{ width: '80px', color: 'var(--text-secondary)' }}>Audio</span>
              <div style={{ flex: 1, height: '2px', background: '#e0e0e0', position: 'relative' }}>
                <div style={{
                  position: 'absolute',
                  left: '45%',
                  top: '-6px',
                  width: '12px',
                  height: '12px',
                  borderRadius: '50%',
                  background: 'var(--accent)'
                }}></div>
              </div>
              <span style={{ fontSize: '0.8rem', color: 'var(--text-secondary)' }}>{stats.audioMsgCount} msgs</span>
            </div>
            <div style={{ display: 'flex', alignItems: 'center', gap: '0.5rem' }}>
              <span style={{ width: '80px', color: 'var(--text-secondary)' }}>Pose</span>
              <div style={{ flex: 1, height: '2px', background: '#e0e0e0', position: 'relative' }}>
                <div style={{
                  position: 'absolute',
                  left: '42%',
                  top: '-6px',
                  width: '12px',
                  height: '12px',
                  borderRadius: '50%',
                  background: 'var(--accent)'
                }}></div>
              </div>
              <span style={{ fontSize: '0.8rem', color: 'var(--text-secondary)' }}>Latest</span>
            </div>
            <div style={{
              display: 'flex',
              alignItems: 'center',
              gap: '0.5rem',
              marginTop: '0.75rem',
              paddingTop: '0.75rem',
              borderTop: '2px dashed var(--accent)'
            }}>
              <span style={{ width: '80px', color: 'var(--accent)', fontWeight: '700' }}>Synced</span>
              <div style={{ flex: 1, height: '3px', background: 'var(--accent)', position: 'relative', borderRadius: '2px' }}>
                <div style={{
                  position: 'absolute',
                  left: '43%',
                  top: '-7px',
                  width: '16px',
                  height: '16px',
                  borderRadius: '50%',
                  background: 'var(--accent)',
                  border: '3px solid white',
                  boxShadow: '0 2px 8px rgba(165, 0, 52, 0.3)'
                }}></div>
              </div>
              <span style={{ fontSize: '0.8rem', color: 'var(--accent)', fontWeight: '600' }}>{stats.syncedCount} synced</span>
            </div>
            {/* 파이프라인 플로우 */}
            <div style={{ marginTop: '1.25rem', paddingTop: '1rem', borderTop: '1px solid #e0e0e0' }}>
              <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'center', gap: '0.75rem', flexWrap: 'wrap' }}>
                {/* Step 1: Synced Data */}
                <div style={{
                  padding: '0.75rem 1rem',
                  background: 'white',
                  border: '2px solid var(--accent)',
                  borderRadius: '10px',
                  textAlign: 'center',
                  minWidth: '110px'
                }}>
                  <div style={{ fontSize: '0.65rem', color: 'var(--text-secondary)', marginBottom: '0.2rem' }}>STEP 1</div>
                  <div style={{ fontSize: '0.8rem', fontWeight: '600', color: 'var(--accent)' }}>Synced Data</div>
                  <div style={{ fontSize: '0.65rem', color: 'var(--text-secondary)', marginTop: '0.2rem' }}>3 sensors</div>
                </div>

                {/* Arrow */}
                <div style={{ fontSize: '1.25rem', color: 'var(--accent)' }}>→</div>

                {/* Step 2: Context Encoder */}
                <div style={{
                  padding: '0.75rem 1rem',
                  background: 'white',
                  border: '2px solid #e0e0e0',
                  borderRadius: '10px',
                  textAlign: 'center',
                  minWidth: '110px'
                }}>
                  <div style={{ fontSize: '0.65rem', color: 'var(--text-secondary)', marginBottom: '0.2rem' }}>STEP 2</div>
                  <div style={{ fontSize: '0.8rem', fontWeight: '600', color: 'var(--text-primary)' }}>Context Encoder</div>
                  <div style={{ fontSize: '0.65rem', color: 'var(--text-secondary)', marginTop: '0.2rem' }}>Attention (160-dim)</div>
                </div>

                {/* Arrow */}
                <div style={{ fontSize: '1.25rem', color: 'var(--text-secondary)' }}>→</div>

                {/* Step 3: Buffer */}
                <div style={{
                  padding: '0.75rem 1rem',
                  background: 'white',
                  border: '2px solid #e0e0e0',
                  borderRadius: '10px',
                  textAlign: 'center',
                  minWidth: '110px'
                }}>
                  <div style={{ fontSize: '0.65rem', color: 'var(--text-secondary)', marginBottom: '0.2rem' }}>STEP 3</div>
                  <div style={{ fontSize: '0.8rem', fontWeight: '600', color: 'var(--text-primary)' }}>Buffer</div>
                  <div style={{ fontSize: '0.65rem', color: 'var(--text-secondary)', marginTop: '0.2rem' }}>30 timesteps</div>
                </div>

                {/* Arrow */}
                <div style={{ fontSize: '1.25rem', color: 'var(--text-secondary)' }}>→</div>

                {/* Step 4: GRU Model */}
                <div style={{
                  padding: '0.75rem 1rem',
                  background: 'white',
                  border: '2px solid #e0e0e0',
                  borderRadius: '10px',
                  textAlign: 'center',
                  minWidth: '110px'
                }}>
                  <div style={{ fontSize: '0.65rem', color: 'var(--text-secondary)', marginBottom: '0.2rem' }}>STEP 4</div>
                  <div style={{ fontSize: '0.8rem', fontWeight: '600', color: 'var(--text-primary)' }}>GRU Model</div>
                  <div style={{ fontSize: '0.65rem', color: 'var(--text-secondary)', marginTop: '0.2rem' }}>FedPer</div>
                </div>

                {/* Arrow */}
                <div style={{ fontSize: '1.25rem', color: 'var(--accent)' }}>→</div>

                {/* Step 5: Prediction */}
                <div style={{
                  padding: '0.75rem 1rem',
                  background: 'var(--accent)',
                  border: '2px solid var(--accent)',
                  borderRadius: '10px',
                  textAlign: 'center',
                  minWidth: '110px'
                }}>
                  <div style={{ fontSize: '0.65rem', color: 'rgba(255,255,255,0.8)', marginBottom: '0.2rem' }}>STEP 5</div>
                  <div style={{ fontSize: '0.8rem', fontWeight: '600', color: 'white' }}>Prediction</div>
                  <div style={{ fontSize: '0.65rem', color: 'rgba(255,255,255,0.8)', marginTop: '0.2rem' }}>4 zones</div>
                </div>
              </div>
            </div>
          </div>
        </div>

        {/* 통계 */}
        <div className="grid grid-4">
          <div className="stat-card">
            <div className="stat-label">Visual Messages</div>
            <div className="stat-value-small">{stats.visualMsgCount}</div>
          </div>
          <div className="stat-card">
            <div className="stat-label">Audio Messages</div>
            <div className="stat-value-small">{stats.audioMsgCount}</div>
          </div>
          <div className="stat-card">
            <div className="stat-label">Synced Timesteps</div>
            <div className="stat-value-small" style={{ color: 'var(--accent)' }}>{stats.syncedCount}</div>
          </div>
          <div className="stat-card">
            <div className="stat-label">Sync Latency</div>
            <div className="stat-value-small">{stats.latencyMs}ms</div>
          </div>
        </div>
      </div>
    </div>
  );
};

export default FR2Page;
