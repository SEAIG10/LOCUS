import { prisma } from '../../config/db';
import { FastifyInstance } from 'fastify';
import { LocationSource } from '@prisma/client';
import { eventBus, EVENTS } from '../../lib/eventBus'; // ✅ EventBus 임포트

const BATCH_SIZE = 50;
const FLUSH_INTERVAL = 5000;

// 메모리 버퍼용 인터페이스
interface PendingLog {
  deviceId: number;
  x: number;
  y: number;
  z: number;
  recordedAt: Date;
  accuracy: number;
  source: LocationSource;
}

let logBuffer: PendingLog[] = [];

/**
 * 1. [저장] 위치 데이터 수신 및 버퍼링 (POST용)
 * Mobile App -> HTTP POST -> 여기 도착
 */
export const bufferLocationLog = async (server: FastifyInstance, data: any) => {
  const record: PendingLog = {
    deviceId: data.clientId ? Number(data.clientId) : 1,
    x: data.position3D?.x || 0,
    y: data.position3D?.y || 0,
    z: data.position3D?.z || 0,
    recordedAt: new Date(data.timestamp || Date.now()),
    accuracy: data.accuracy || 0,
    source: 'MOBILE',
  };

  // ✅ [수정됨] 직접 io.emit 하지 않고, EventBus에 "위치 업데이트 됨" 알림
  eventBus.emit(EVENTS.NEW_ROBOT_LOCATION, record);

  // 메모리 버퍼에 추가 (DB 일괄 저장용)
  logBuffer.push(record);

  if (logBuffer.length >= BATCH_SIZE) {
    await flushLogsToDB();
  }
};

/**
 * 2. [조회] 가장 최신 위치 데이터 1개 반환 (GET Polling Fallback용)
 */
export const getLatestLocation = async () => {
  // 1순위: 버퍼 확인
  if (logBuffer.length > 0) {
    return logBuffer[logBuffer.length - 1];
  }

  // 2순위: DB 확인
  const latestFromDB = await prisma.robotLocation.findFirst({
    orderBy: { recordedAt: 'desc' },
    select: { x: true, y: true, z: true, recordedAt: true, id: true }
  });

  return latestFromDB;
};

/**
 * 3. [내부] 버퍼 -> DB 일괄 저장 (Flush)
 */
const flushLogsToDB = async () => {
  if (logBuffer.length === 0) return;

  const chunk = [...logBuffer];
  logBuffer = []; 

  try {
    console.log(`💾 [Batch] 위치 로그 ${chunk.length}개 DB 저장...`);
    
    await prisma.robotLocation.createMany({
      data: chunk.map(log => ({
        deviceId: log.deviceId,
        x: log.x,
        y: log.y,
        z: log.z,
        recordedAt: log.recordedAt,
        source: log.source,
        rawPayloadJson: { accuracy: log.accuracy } 
      })),
      skipDuplicates: true,
    });
  } catch (error) {
    console.error('❌ [Batch] 로그 저장 실패:', error);
  }
};

// 주기적 저장 실행
setInterval(() => {
  if (logBuffer.length > 0) flushLogsToDB();
}, FLUSH_INTERVAL);