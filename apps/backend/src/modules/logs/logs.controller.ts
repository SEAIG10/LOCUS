import { FastifyRequest, FastifyReply } from 'fastify';
import { prisma } from '../../config/db';
import { bufferLocationLog, getLatestLocation } from './logs.service';

/**
 * [POST] 위치 데이터 수신 (트래커 -> 백엔드)
 */
export async function createLogHandler(
  request: FastifyRequest,
  reply: FastifyReply
) {
  try {
    const body = request.body as any;
    bufferLocationLog(request.server, body);
    return reply.code(200).send({ status: 'ok', buffered: true });
  } catch (error) {
    console.error('Log Create Error:', error);
    return reply.code(500).send({ message: 'Internal Server Error' });
  }
}

/**
 * [GET] 최신 위치 데이터 1개 조회 (프론트엔드 폴링용)
 */
export async function getLatestLogHandler(
  request: FastifyRequest,
  reply: FastifyReply
) {
  try {
    const latest = await getLatestLocation();
    if (!latest) return reply.code(200).send({});
    return reply.code(200).send(latest);
  } catch (error) {
    console.error('Log Get Error:', error);
    return reply.code(500).send({ message: 'Internal Server Error' });
  }
}

/**
 * [GET] 타임라인용 이벤트 조회 (청소 기록 등)
 * 조건: 오늘 하루 (00:00 ~ 23:59) 데이터만 조회
 */
export async function getEventsHandler(req: FastifyRequest, reply: FastifyReply) {
  try {
    const { homeId } = req.query as { homeId: string };
    
    // 오늘 날짜 범위 설정
    const now = new Date();
    const startOfDay = new Date(now.setHours(0, 0, 0, 0));
    const endOfDay = new Date(now.setHours(23, 59, 59, 999));

    const events = await prisma.sensorEvent.findMany({
      where: {
        homeId: parseInt(homeId),
        eventTime: {
          gte: startOfDay,
          lte: endOfDay
        }
      },
      orderBy: { eventTime: 'desc' },
      include: { label: true }
    });

    // BigInt 처리
    const safeEvents = events.map(e => ({
      ...e,
      id: e.id.toString(),
      robotLocationId: e.robotLocationId?.toString()
    }));

    return reply.send(safeEvents);
  } catch (error) {
    console.error(error);
    return reply.code(500).send({ message: 'Error fetching events' });
  }
}

/**
 * ✅ [수정됨] 오염 예측 현황 조회 (새로고침 시 유지용)
 * 로직: 집안의 모든 구역을 돌면서, 각 구역의 "가장 최신" 예측값을 하나씩 가져와서 합칩니다.
 */
export async function getPredictionsHandler(req: FastifyRequest, reply: FastifyReply) {
  try {
    const { homeId } = req.query as { homeId: string };
    const hId = parseInt(homeId);

    if (isNaN(hId)) {
      return reply.code(400).send({ message: 'Invalid homeId' });
    }

    // 1. 해당 집의 모든 라벨(구역) 조회
    const labels = await prisma.roomLabel.findMany({
      where: { homeId: hId },
      select: { id: true, name: true }
    });

    // 2. 각 라벨별로 가장 최신 예측값 조회
    // (Promise.all로 병렬 처리하여 속도 최적화)
    const promises = labels.map(async (label) => {
      const latestPrediction = await prisma.pollutionPrediction.findFirst({
        where: { 
          homeId: hId,
          labelId: label.id 
        },
        orderBy: { predictionTime: 'desc' }, // 최신순 정렬
        take: 1, // 딱 1개만
        include: { label: true }
      });
      return latestPrediction;
    });

    const predictions = await Promise.all(promises);

    // 3. 데이터가 있는 것만 필터링하고 ID 포맷 변환
    const cleanResults = predictions
      .filter(p => p !== null)
      .map(p => ({
        ...p,
        id: p!.id.toString(), // BigInt -> String
        labelId: Number(p!.labelId) // 프론트 비교용 Number 변환
      }));

    return reply.send(cleanResults);

  } catch (error) {
    console.error('Predictions Fetch Error:', error);
    return reply.code(500).send({ message: 'Error fetching predictions' });
  }
}