/**
 * src/modules/mqtt/mqtt.service.ts
 * 역할: MQTT 메시지 수신 -> DB 저장 -> EventBus 알림
 * + [NEW] 로봇 위치 정보 송신 (Backend -> AI Context)
 */

import mqtt from 'mqtt';
import { prisma } from '../../config/db';
import { eventBus, EVENTS } from '../../lib/eventBus';

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

/**
 * 🚀 [추가됨] 로봇 위치+구역 정보를 MQTT로 전송 (To: sensor_context.py)
 * Topic: home/{homeId}/robot/location
 */
export function publishRobotLocation(homeId: number, payload: any) {
  if (!client || !client.connected) {
    // 연결 안 됐으면 패스 (로그 너무 많이 찍히면 주석 처리)
    return;
  }

  const topic = `home/${homeId}/robot/location`;
  const message = JSON.stringify(payload);

  // QoS 0: 빠르고 가볍게 전송 (위치는 가끔 유실돼도 괜찮음)
  client.publish(topic, message, { qos: 0 }, (err) => {
    if (err) console.error('[MQTT] Publish error:', err);
  });
}

// -------------------------------------------------------------
// 기존 로직들 (handleMessage, PollutionPrediction 등) 그대로 유지
// -------------------------------------------------------------

async function handleMessage(topic: string, message: Buffer): Promise<void> {
  try {
    const data = JSON.parse(message.toString());
    
    if (topic.includes('/prediction/pollution')) {
      await handlePollutionPrediction(topic, data);
    }
    else if (topic.includes('/cleaning/result')) {
      await handleCleaningResult(topic, data);
    }
    else if (topic.includes('/cleaning/status')) {
      console.log(`[MQTT] Cleaning status: ${topic}`, data);
    }
    else if (topic.includes('/status')) {
      console.log(`[MQTT] Device status: ${topic}`, data);
    }
  } catch (error) {
    console.error('[MQTT] Error processing message:', error);
  }
}

async function handlePollutionPrediction(topic: string, data: any): Promise<void> {
  try {
    const homeId = extractHomeIdFromTopic(topic);
    if (!homeId) return;

    const predictions = data.predictions || {};
    const device = await prisma.device.findFirst({
      where: { homeId: parseInt(homeId) },
    });
    if (!device) return;

    const savedPredictions = [];

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
        include: { label: true }
      });
      savedPredictions.push(saved);
    }

    if (savedPredictions.length > 0) {
      eventBus.emit(EVENTS.NEW_POLLUTION_PREDICTION, {
        homeId: parseInt(homeId),
        data: savedPredictions
      });
    }

  } catch (error) {
    console.error('[MQTT] Error saving pollution prediction:', error);
  }
}

async function handleCleaningResult(topic: string, data: any): Promise<void> {
  try {
    const homeId = extractHomeIdFromTopic(topic);
    if (!homeId) return;

    const zoneName = data.zone;
    const label = await prisma.roomLabel.findFirst({
      where: { homeId: parseInt(homeId), name: zoneName },
    });

    if (!label) {
      console.warn(`[MQTT] Zone '${zoneName}' not found for cleaning result.`);
      return;
    }

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

    eventBus.emit(EVENTS.NEW_SENSOR_EVENT, {
      homeId: parseInt(homeId),
      data: savedEvent
    });

    // 오염도 리셋 로직
    const device = await prisma.device.findFirst({
      where: { homeId: parseInt(homeId) }
    });

    if (device) {
      const cleanPrediction = await prisma.pollutionPrediction.create({
        data: {
          homeId: parseInt(homeId),
          deviceId: device.id,
          labelId: label.id,
          probability: 0,
          modelVersion: 'cleaning-reset',
          predictionTime: new Date(),
        },
        include: { label: true }
      });

      eventBus.emit(EVENTS.NEW_POLLUTION_PREDICTION, {
        homeId: parseInt(homeId),
        data: [cleanPrediction]
      });
    }

  } catch (error) {
    console.error('[MQTT] Error saving cleaning result:', error);
  }
}

function extractHomeIdFromTopic(topic: string): string | null {
  const match = topic.match(/home\/([^\/]+)\//);
  return match ? match[1] : null;
}

export function disconnectMQTT(): void {
  if (client) {
    client.end();
    client = null;
  }
}