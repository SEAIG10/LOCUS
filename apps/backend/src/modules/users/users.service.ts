// src/modules/users/users.service.ts
import { prisma } from '../../config/db';

export class UsersService {
  async listUsers() {
    return prisma.user.findMany();
  }

  async createUser(input: { email: string; password: string; name?: string }) {
    return prisma.user.create({
      data: {
        email: input.email,
        password: input.password,
        name: input.name,
      },
    });
  }
}
