// src/config/db.ts
import { PrismaClient } from '@prisma/client';

export const prisma = new PrismaClient();

export async function connectDB() {
  await prisma.$connect();
  console.log('[DB] Connected to PostgreSQL');
}
