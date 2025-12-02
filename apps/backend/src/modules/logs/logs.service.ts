import { prisma } from '../../config/db';
import { FastifyInstance } from 'fastify';
import { LocationSource } from '@prisma/client';
import { eventBus, EVENTS } from '../../lib/eventBus';
import { publishRobotLocation } from '../mqtt/mqtt.service';

const BATCH_SIZE = 50;
const FLUSH_INTERVAL = 5000;

// =========================================================
// 1. [Util] 기하학 계산 (Point in Polygon)
// =========================================================
interface Point { x: number; z: number; }

function isPointInPolygon(p: Point, polygon: Point[]): boolean {
  let isInside = false;
  let minX = polygon[0].x, maxX = polygon[0].x;
  let minZ = polygon[0].z, maxZ = polygon[0].z;

  // Bounding Box Check
  for (const point of polygon) {
    minX = Math.min(point.x, minX);
    maxX = Math.max(point.x, maxX);
    minZ = Math.min(point.z, minZ);
    maxZ = Math.max(point.z, maxZ);
  }
  if (p.x < minX || p.x > maxX || p.z < minZ || p.z > maxZ) return false;

  // Ray Casting Algorithm
  let j = polygon.length - 1;
  for (let i = 0; i < polygon.length; i++) {
    if ((polygon[i].z > p.z) !== (polygon[j].z > p.z) &&
        p.x < ((polygon[j].x - polygon[i].x) * (p.z - polygon[i].z)) / (polygon[j].z - polygon[i].z) + polygon[i].x) {
      isInside = !isInside;
    }
    j = i;
  }
  return isInside;
}

// =========================================================
// 2. [Cache] 라벨 & 보정 데이터 캐싱
// =========================================================
interface CachedLabel {
  id: number;
  homeId: number;
  name: string;
  points: Point[];
}

interface CachedCalibration {
  homeId: number;
  scale: number;
  sensorDirectionDeg: number; // 프론트의 dataRotateDeg (Dir)
  rotationDeg: number;        // 프론트의 modelRotationY (Map)
  offsetX: number;
  offsetZ: number;
}

let cachedLabels: CachedLabel[] = [];
let cachedCalibs: CachedCalibration[] = [];
let lastCacheUpdate = 0;

async function refreshLabelCache() {
  const now = Date.now();
  // 1분(60000ms) 쿨타임
  if (cachedLabels.length > 0 && now - lastCacheUpdate < 60000) return;

  try {
    // 1. 라벨 가져오기
    const labels = await prisma.roomLabel.findMany({
      include: { points: { orderBy: { orderIndex: 'asc' } } }
    });
    cachedLabels = labels.map(l => ({
      id: l.id,
      homeId: l.homeId,
      name: l.name,
      points: l.points.map(p => ({ x: p.x, z: p.z }))
    }));

    // 2. 보정값 가져오기
    const calibs = await prisma.mapCalibration.findMany();
    cachedCalibs = calibs.map(c => ({
      homeId: c.homeId,
      scale: c.scale,
      sensorDirectionDeg: c.sensorDirectionDeg, // 중요: 회전값 1
      rotationDeg: c.rotationDeg,               // 중요: 회전값 2
      offsetX: c.offsetX,
      offsetZ: c.offsetZ
    }));

    lastCacheUpdate = now;
    console.log(`🗺️  [Cache] Refreshed: ${cachedLabels.length} Labels, ${cachedCalibs.length} Calibrations.`);
  } catch (e) {
    console.error('Cache Refresh Error:', e);
  }
}

// 🔥 [Math] 좌표 변환 함수 (프론트엔드 로직 복제)
function applyCalibration(rawX: number, rawZ: number, calib: CachedCalibration): Point {
  // 1. Scale
  const scaledX = rawX * calib.scale;
  const scaledZ = rawZ * calib.scale;

  // 2. Rotation (Sensor Direction)
  // 프론트엔드 코드: const radData = (mapConfig.dataRotateDeg * Math.PI) / 180;
  const rad = (calib.sensorDirectionDeg * Math.PI) / 180;
  
  const rotatedX = scaledX * Math.cos(rad) - scaledZ * Math.sin(rad);
  const rotatedZ = scaledX * Math.sin(rad) + scaledZ * Math.cos(rad);

  // 3. Offset ( + Map Rotation은 Three.js 뷰포트용이라 좌표 계산엔 보통 Offset만 관여)
  // 프론트엔드: return [dataX + mapConfig.offsetX, ... , dataZ + mapConfig.offsetZ];
  return {
    x: rotatedX + calib.offsetX,
    z: rotatedZ + calib.offsetZ
  };
}

// =========================================================
// 3. [ID Resolution] 기기 ID 고정 (Device 1, Home 6)
// =========================================================
async function getFixedDeviceAndHomeId() {
  if (global.fixedDeviceInfo) return global.fixedDeviceInfo;
  const FIXED_DEVICE_ID = 1;
  const FIXED_HOME_ID = 6;

  // DB 확인
  const device = await prisma.device.findUnique({ where: { id: FIXED_DEVICE_ID } });
  
  if (device) {
    global.fixedDeviceInfo = { deviceId: device.id, homeId: device.homeId };
    return global.fixedDeviceInfo;
  } else {
    // 없으면 기본값 리턴 (나중에 에러 날 수 있지만 일단 진행)
    return { deviceId: FIXED_DEVICE_ID, homeId: FIXED_HOME_ID };
  }
}
declare global { var fixedDeviceInfo: { deviceId: number, homeId: number } | undefined; }

// =========================================================
// 4. 메인 로직 (Buffer -> Flush)
// =========================================================
interface PendingLog {
  deviceId: number;
  x: number;
  y: number;
  z: number;
  recordedAt: Date;
  accuracy: number;
  source: LocationSource;
  labelId?: number | null;
}

let logBuffer: PendingLog[] = [];

export const bufferLocationLog = async (server: FastifyInstance, data: any) => {
  if (cachedLabels.length === 0) await refreshLabelCache();

  // 1. 기기/홈 ID 고정 (1, 6)
  const { deviceId, homeId } = await getFixedDeviceAndHomeId();

  // 2. Raw 좌표
  const rawX = data.position3D?.x || 0;
  const rawZ = data.position3D?.z || 0;

  // 3. 🔥 좌표 보정 (DB값 적용)
  const calib = cachedCalibs.find(c => c.homeId === homeId);
  let targetX = rawX;
  let targetZ = rawZ;

  if (calib) {
    const p = applyCalibration(rawX, rawZ, calib);
    targetX = p.x;
    targetZ = p.z;

    // 🕵️‍♀️ [디버깅 로그] - 숫자가 제대로 바뀌는지 확인하세요!
    // console.log(`[DEBUG] Raw(${rawX.toFixed(2)}, ${rawZ.toFixed(2)}) -> Calib(${targetX.toFixed(2)}, ${targetZ.toFixed(2)})`);
  } else {
    // console.warn(`⚠️ [Warning] Home ${homeId} has no calibration! Using Raw Coords.`);
  }

  // 4. 위치 판별 (보정된 좌표 사용)
  let foundLabelId: number | null = null;
  let foundLabelName: string | null = null;

  const targetLabels = cachedLabels.filter(l => l.homeId === homeId);
  
  for (const label of targetLabels) {
    if (label.points.length >= 3 && isPointInPolygon({ x: targetX, z: targetZ }, label.points)) {
      foundLabelId = label.id;
      foundLabelName = label.name;
      // console.log(`✅ MATCH! Robot is in [ ${label.name} ]`);
      break; 
    }
  }

  // 5. MQTT 전송
  publishRobotLocation(homeId, {
    x: targetX, // 보정된 좌표 전송 (AI 분석용)
    z: targetZ,
    zone: foundLabelName, 
    timestamp: new Date().toISOString()
  });

  // 6. DB 저장용 레코드 생성
  const record: PendingLog = {
    deviceId: deviceId, 
    x: rawX, // DB에는 원본 좌표 저장 (나중에 설정 바뀌어도 원본 유지)
    y: data.position3D?.y || 0,
    z: rawZ, 
    recordedAt: new Date(data.timestamp || Date.now()),
    accuracy: data.accuracy || 0,
    source: 'MOBILE',
    labelId: foundLabelId // 이제는 NULL이 아닐 것임!
  };

  eventBus.emit(EVENTS.NEW_ROBOT_LOCATION, {
    ...record,
    zoneName: foundLabelName,
    homeId: homeId 
  });

  logBuffer.push(record);
  if (logBuffer.length >= BATCH_SIZE) {
    await flushLogsToDB();
  }
};

export const getLatestLocation = async () => {
  if (logBuffer.length > 0) return logBuffer[logBuffer.length - 1];
  return await prisma.robotLocation.findFirst({
    orderBy: { recordedAt: 'desc' },
    select: { x: true, y: true, z: true, recordedAt: true, id: true }
  });
};

const flushLogsToDB = async () => {
  if (logBuffer.length === 0) return;
  const chunk = [...logBuffer];
  logBuffer = []; 

  try {
    // console.log(`💾 [Batch] Saving ${chunk.length} logs (Device: ${chunk[0].deviceId})...`);
    await prisma.robotLocation.createMany({
      data: chunk.map(log => ({
        deviceId: log.deviceId,
        x: log.x, y: log.y, z: log.z,
        recordedAt: log.recordedAt,
        source: log.source,
        labelId: log.labelId,
        rawPayloadJson: { accuracy: log.accuracy } 
      })),
      skipDuplicates: true,
    });
  } catch (error) {
    console.error('❌ [Batch] Save Error:', error);
  }
};

setInterval(() => {
  if (logBuffer.length > 0) flushLogsToDB();
}, FLUSH_INTERVAL);