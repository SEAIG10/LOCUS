import { FastifyInstance } from 'fastify';
import { loginHandler, registerHandler } from './auth.controller';

export async function authRoutes(app: FastifyInstance) {
  app.post('/register', registerHandler);
  app.post('/login', loginHandler);
}