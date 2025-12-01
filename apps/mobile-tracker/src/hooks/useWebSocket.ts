import { useCallback, useEffect, useRef, useState } from 'react';
import { ConnectionStatus, WebSocketMessage } from '../types';

interface UseWebSocketOptions {
  onMessage?: (message: WebSocketMessage) => void;
  onOpen?: () => void;
  onClose?: () => void;
  onError?: (error: Event) => void;
}

export const useWebSocket = (options: UseWebSocketOptions = {}) => {
  const [status, setStatus] = useState<ConnectionStatus>('disconnected');
  const [url, setUrl] = useState<string>('');
  const wsRef = useRef<WebSocket | null>(null);

  const connect = useCallback((serverUrl: string) => {
    if (wsRef.current?.readyState === WebSocket.OPEN) {
      console.warn('WebSocket is already connected');
      return;
    }

    setUrl(serverUrl);
    setStatus('connecting');

    try {
      const ws = new WebSocket(serverUrl);
      wsRef.current = ws;

      ws.onopen = () => {
        setStatus('connected');
        options.onOpen?.();
      };

      ws.onmessage = (event) => {
        try {
          const message = JSON.parse(event.data) as WebSocketMessage;
          options.onMessage?.(message);
        } catch (error) {
          console.error('Failed to parse WebSocket message:', error);
        }
      };

      ws.onerror = (error) => {
        setStatus('error');
        options.onError?.(error);
      };

      ws.onclose = () => {
        setStatus('disconnected');
        options.onClose?.();
      };
    } catch (error) {
      console.error('Failed to create WebSocket:', error);
      setStatus('error');
    }
  }, [options]);

  const disconnect = useCallback(() => {
    if (wsRef.current) {
      wsRef.current.close();
      wsRef.current = null;
    }
    setStatus('disconnected');
  }, []);

  const send = useCallback((message: WebSocketMessage) => {
    if (wsRef.current?.readyState === WebSocket.OPEN) {
      wsRef.current.send(JSON.stringify(message));
      return true;
    }
    console.warn('WebSocket is not connected');
    return false;
  }, []);

  useEffect(() => {
    return () => {
      if (wsRef.current) {
        wsRef.current.close();
      }
    };
  }, []);

  return {
    status,
    url,
    connect,
    disconnect,
    send,
    isConnected: status === 'connected',
  };
};
