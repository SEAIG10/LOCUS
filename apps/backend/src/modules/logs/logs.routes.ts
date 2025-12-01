import { FastifyInstance } from 'fastify';
import { 
  createLogHandler, 
  getLatestLogHandler, 
  getEventsHandler,      // ✅ 추가
  getPredictionsHandler  // ✅ 추가
} from './logs.controller';

export default async function logRoutes(fastify: FastifyInstance) {
  
  // 1. [POST] 트래커가 데이터를 쏘는 곳 (Mobile App)
  fastify.post('/record', createLogHandler);

  // 2. [GET] 최신 위치 조회 (Polling Fallback)
  fastify.get('/latest', getLatestLogHandler);

  // 3. [GET] 타임라인 이벤트 조회 (청소 기록 등) ✅
  fastify.get('/events', getEventsHandler);

  // 4. [GET] 오염 예측 현황 조회 ✅
  fastify.get('/predictions', getPredictionsHandler);

}