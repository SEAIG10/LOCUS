// src/modules/users/users.routes.ts
import { FastifyInstance } from 'fastify';
import { UsersController } from './users.controller';

export async function usersRoutes(fastify: FastifyInstance) {
  const controller = new UsersController();

  // prefix는 app.ts에서 '/api'로 붙일 거니까
  // 여기서는 '/users'까지만 쓴다.
  fastify.get('/users', controller.listUsers);
  fastify.post('/users', controller.createUser);
}
