import Fastify, { FastifyInstance } from 'fastify';
import cors from '@fastify/cors';
import websocket from '@fastify/websocket';
import jwt from '@fastify/jwt';
import multipart from '@fastify/multipart';
import fastifyStatic from '@fastify/static';
import path from 'path';

import { env } from './config/env';
import { healthRoutes } from './modules/health/health.routes';
import { usersRoutes } from './modules/users/users.routes';
import { authRoutes } from './modules/auth/auth.routes';
import { homesRoutes } from './modules/homes/homes.routes';
import { devicesRoutes } from './modules/devices/devices.routes';
import { labelsRoutes } from './modules/labels/labels.routes';
import logRoutes from './modules/logs/logs.routes';

(BigInt.prototype as any).toJSON = function () {
  return this.toString();
};

export function buildApp(): FastifyInstance {
  const app = Fastify({
    logger: true,
  });

  // 🔥 [수정 1] DELETE 메소드 명시적 허용 (삭제 문제 해결)
  app.register(cors, {
    origin: true,
    credentials: true,
    methods: ['GET', 'POST', 'PUT', 'DELETE', 'OPTIONS', 'PATCH'] 
  });

  app.register(websocket);
  
  app.register(jwt, {
    secret: env.JWT_SECRET || 'super_secret_key',
  });

  app.register(multipart, {
    attachFieldsToBody: true,
    limits: { fileSize: 5 * 1024 * 1024 }
  });

  // 🔥 [수정 2] 이미지 경로를 '프로젝트 루트/uploads'로 고정 (이미지 엑박 해결)
  // process.cwd()는 현재 실행 중인 프로젝트 폴더 위치를 가리킵니다.
  app.register(fastifyStatic, {
    root: path.join(process.cwd(), 'uploads'), 
    prefix: '/uploads/', 
    decorateReply: false 
  });

  app.register(healthRoutes);

  app.register(async (api) => {
    api.register(authRoutes, { prefix: '/auth' });
    api.register(usersRoutes, { prefix: '/users' });
    api.register(homesRoutes, { prefix: '/homes' });
    api.register(devicesRoutes, { prefix: '/devices' });
    api.register(labelsRoutes, { prefix: '/homes' });
    app.register(logRoutes, { prefix: '/log' });
  }, { prefix: '/api' });

  return app;
}