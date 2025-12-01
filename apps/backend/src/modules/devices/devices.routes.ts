import { FastifyInstance } from 'fastify';

export async function devicesRoutes(app: FastifyInstance) {
  app.get('/', async () => {
    return { message: 'Devices API working' };
  });
}