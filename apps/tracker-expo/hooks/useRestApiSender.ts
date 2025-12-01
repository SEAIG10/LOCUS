import { useState, useRef, useCallback } from 'react';
import axios from 'axios';

interface UseRestSenderProps {
  serverUrl: string; // 예: https://.../api/log/record
  interval?: number; // 전송 주기 (ms), 기본값 200ms (초당 5회)
}

export const useRestApiSender = ({ serverUrl, interval = 200 }: UseRestSenderProps) => {
  const [isSending, setIsSending] = useState(false);
  const [error, setError] = useState<string | null>(null);
  
  // 마지막 전송 시간을 기록 (Throttling 용)
  const lastSentTime = useRef<number>(0);
  
  // 기존엔 서버가 ID를 줬지만, 이제 직접 보내야 하므로 로컬에서 랜덤 ID 생성
  const clientId = useRef(Math.floor(Math.random() * 10000) + 1).current;

  const sendPosition = useCallback(async (position: any, accuracy: number) => {
    const now = Date.now();
    
    // 설정된 주기(interval)보다 빨리 호출되면 전송 스킵
    if (now - lastSentTime.current < interval) {
      return;
    }

    try {
      // isSending은 UI 깜빡임 방지를 위해 에러 상황에서만 false로 처리하거나, 
      // 너무 잦은 갱신을 피하기 위해 로직 단순화
      lastSentTime.current = now;

      // 백엔드(LogRecord) 타입에 맞춘 데이터 포맷
      const payload = {
        clientId: clientId,
        receivedAt: new Date().toISOString(),
        timestamp: now,
        accuracy: accuracy,
        position3D: {
          x: position.x,
          y: position.y,
          z: position.z
        }
      };

      // 백엔드로 직접 POST 요청 (Fire and Forget)
      // 응답을 기다리지만 UI를 블로킹하지 않음
      await axios.post(serverUrl, payload);
      
      setError(null);
      setIsSending(true); // 성공적으로 보내고 있음
      
      // UI 상태 깜빡임 방지를 위해 잠시 후 false 처리 (선택사항)
      setTimeout(() => setIsSending(false), interval / 2);

    } catch (err: any) {
      console.error('API 전송 실패:', err);
      // 네트워크 에러 등
      setError(err.message || '전송 실패');
      setIsSending(false);
    }
  }, [serverUrl, interval, clientId]);

  return {
    isSending,
    error,
    sendPosition,
    clientId // UI에 표시할 내 ID
  };
};