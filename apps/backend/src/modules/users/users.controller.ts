import { FastifyReply, FastifyRequest } from 'fastify';
import { z } from 'zod';
import { UsersService } from './users.service';

const createUserSchema = z.object({
  email: z.string().email(),
  password: z.string().min(6),
  name: z.string().optional(),
});

export class UsersController {
  private service = new UsersService();

  listUsers = async (_req: FastifyRequest, reply: FastifyReply) => {
    const users = await this.service.listUsers();
    return reply.send(users);
  };

  createUser = async (req: FastifyRequest, reply: FastifyReply) => {
    const parsed = createUserSchema.parse(req.body);
    const user = await this.service.createUser(parsed);
    return reply.code(201).send(user);
  };
}
