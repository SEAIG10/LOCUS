// src/modules/health/health.routes.ts
import { FastifyInstance } from 'fastify';

export async function healthRoutes(fastify: FastifyInstance) {
  fastify.get('/health', async () => {
    return { status: 'ok', service: 'LocusBackend' };
  });
}
