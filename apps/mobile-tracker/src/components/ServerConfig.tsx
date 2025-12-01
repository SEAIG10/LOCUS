import { useState } from 'react';
import { ConnectionStatus } from '../types';
import '../styles/ServerConfig.css';

interface ServerConfigProps {
  onConnect: (url: string) => void;
  onDisconnect: () => void;
  status: ConnectionStatus;
}

export const ServerConfig: React.FC<ServerConfigProps> = ({
  onConnect,
  onDisconnect,
  status,
}) => {
  const [url, setUrl] = useState('ws://localhost:8080');

  const handleToggle = () => {
    if (status === 'connected') {
      onDisconnect();
    } else {
      onConnect(url);
    }
  };

  const isConnected = status === 'connected';
  const isConnecting = status === 'connecting';

  return (
    <section className="config-section">
      <h2>서버 설정</h2>
      <div className="input-group">
        <label htmlFor="serverUrl">WebSocket URL</label>
        <input
          type="text"
          id="serverUrl"
          value={url}
          onChange={(e) => setUrl(e.target.value)}
          disabled={isConnected}
          placeholder="ws://your-server:8080"
        />
      </div>
      <button
        onClick={handleToggle}
        disabled={isConnecting}
        className={`btn ${isConnected ? 'btn-stop' : 'btn-connect'}`}
      >
        {isConnecting ? '연결 중...' : isConnected ? '연결 끊기' : '연결하기'}
      </button>
    </section>
  );
};
