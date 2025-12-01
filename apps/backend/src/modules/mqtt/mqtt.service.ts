/**
 * src/modules/mqtt/mqtt.service.ts
 * 역할: MQTT 메시지 수신 -> DB 저장 -> EventBus 알림
 */

import mqtt from 'mqtt';
import { prisma } from '../../config/db';
import { eventBus, EVENTS } from '../../lib/eventBus'; // ✅ EventBus 임포트

// MQTT 클라이언트 설정
const MQTT_BROKER_URL = process.env.MQTT_BROKER_URL || 'mqtt://localhost:1883';

let client: mqtt.MqttClient | null = null;

export function connectMQTT(): void {
  if (client) {
    console.log('[MQTT] Already connected');
    return;
  }

  client = mqtt.connect(MQTT_BROKER_URL);

  client.on('connect', () => {
    console.log('[MQTT] Connected to broker:', MQTT_BROKER_URL);

    // 토픽 구독
    client!.subscribe('home/+/prediction/pollution');
    client!.subscribe('home/+/cleaning/#');
    client!.subscribe('edge/+/status');
  });

  client.on('message', handleMessage);

  client.on('error', (err) => console.error('[MQTT] Connection error:', err));
  client.on('close', () => console.log('[MQTT] Connection closed'));
}

async function handleMessage(topic: string, message: Buffer): Promise<void> {
  try {
    const data = JSON.parse(message.toString());
    
    // 1. 오염도 예측
    if (topic.includes('/prediction/pollution')) {
      await handlePollutionPrediction(topic, data);
    }
    // 2. 청소 결과
    else if (topic.includes('/cleaning/result')) {
      await handleCleaningResult(topic, data);
    }
    // 3. 청소 상태 (로그만)
    else if (topic.includes('/cleaning/status')) {
      console.log(`[MQTT] Cleaning status: ${topic}`, data);
    }
    // 4. 디바이스 상태 (로그만)
    else if (topic.includes('/status')) {
      console.log(`[MQTT] Device status: ${topic}`, data);
    }
  } catch (error) {
    console.error('[MQTT] Error processing message:', error);
  }
}

/**
 * 오염도 예측 처리
 */
async function handlePollutionPrediction(topic: string, data: any): Promise<void> {
  try {
    const homeId = extractHomeIdFromTopic(topic);
    if (!homeId) return;

    const predictions = data.predictions || {};
    
    // 디바이스 조회
    const device = await prisma.device.findFirst({
      where: { homeId: parseInt(homeId) },
    });
    if (!device) return; // 디바이스 없으면 무시

    const savedPredictions = [];

    // 구역별 저장
    for (const [zoneName, probability] of Object.entries(predictions)) {
      const label = await prisma.roomLabel.findFirst({
        where: { homeId: parseInt(homeId), name: zoneName },
      });
      if (!label) continue;

      const saved = await prisma.pollutionPrediction.create({
        data: {
          homeId: parseInt(homeId),
          deviceId: device.id,
          labelId: label.id,
          probability: probability as number,
          modelVersion: 'gru-v1',
          predictionTime: new Date(),
        },
        include: { label: true } // 프론트 전송용 include
      });
      savedPredictions.push(saved);
    }

    // ✅ [핵심] 저장 완료 후 내부 이벤트 발생
    if (savedPredictions.length > 0) {
      eventBus.emit(EVENTS.NEW_POLLUTION_PREDICTION, {
        homeId: parseInt(homeId),
        data: savedPredictions
      });
      console.log(`📢 [EventBus] Emitted pollution prediction for home ${homeId}`);
    }

  } catch (error) {
    console.error('[MQTT] Error saving pollution prediction:', error);
  }
}

/**
 * 청소 결과 처리
 * 🔥 수정됨: 청소 완료 시 오염도 0% 데이터 생성 로직 추가
 */
async function handleCleaningResult(topic: string, data: any): Promise<void> {
  try {
    const homeId = extractHomeIdFromTopic(topic);
    if (!homeId) return;

    const zoneName = data.zone;
    
    // 라벨 찾기
    const label = await prisma.roomLabel.findFirst({
      where: { homeId: parseInt(homeId), name: zoneName },
    });

    if (!label) {
      console.warn(`[MQTT] Zone '${zoneName}' not found for cleaning result.`);
      return;
    }

    // 1. [SensorEvent] 청소 완료 이벤트 저장 (타임라인용)
    const savedEvent = await prisma.sensorEvent.create({
      data: {
        homeId: parseInt(homeId),
        eventType: 'SYSTEM',
        subType: 'CLEANING_COMPLETED',
        eventTime: new Date(data.timestamp),
        labelId: label.id,
        payloadJson: {
          zone: zoneName,
          duration_seconds: data.duration_seconds,
        },
      },
      include: { label: true } 
    });

    // 타임라인 업데이트 알림
    eventBus.emit(EVENTS.NEW_SENSOR_EVENT, {
      homeId: parseInt(homeId),
      data: savedEvent
    });
    console.log(`📢 [EventBus] Emitted cleaning event for home ${homeId}`);

    // 2. 🔥 [PollutionPrediction] 오염도 0%로 리셋 (구조도용)
    // 디바이스 찾기 (데이터 무결성 위해)
    const device = await prisma.device.findFirst({
      where: { homeId: parseInt(homeId) }
    });

    if (device) {
      // 0% 데이터 생성
      const cleanPrediction = await prisma.pollutionPrediction.create({
        data: {
          homeId: parseInt(homeId),
          deviceId: device.id,
          labelId: label.id,
          probability: 0, // ✅ 0% 강제 설정
          modelVersion: 'cleaning-reset',
          predictionTime: new Date(), // 현재 시간
        },
        include: { label: true }
      });

      // 오염도 업데이트 알림 (프론트엔드가 받아서 빨간색 없앰)
      eventBus.emit(EVENTS.NEW_POLLUTION_PREDICTION, {
        homeId: parseInt(homeId),
        data: [cleanPrediction]
      });
      console.log(`✨ [Auto-Reset] Pollution level for '${zoneName}' set to 0%`);
    }

  } catch (error) {
    console.error('[MQTT] Error saving cleaning result:', error);
  }
}

// 헬퍼 함수
function extractHomeIdFromTopic(topic: string): string | null {
  const match = topic.match(/home\/([^\/]+)\//);
  return match ? match[1] : null;
}

// (필요 시) 구역 업데이트 알림 - 나가는 메시지이므로 client.publish 유지
export async function notifyZoneUpdate(homeId: string): Promise<void> {
  if (!client) return;
  // ... (기존 로직 유지, 필요시 구현) ...
}

export function disconnectMQTT(): void {
  if (client) {
    client.end();
    client = null;
  }
}