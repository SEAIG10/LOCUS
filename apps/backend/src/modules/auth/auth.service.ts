import { prisma } from '../../config/db'; // config/db.ts에 prisma 인스턴스가 있다고 가정
import bcrypt from 'bcryptjs';
import { FastifyInstance } from 'fastify';

// 회원가입
export const registerUser = async (email: string, password: string, name: string) => {
  const existingUser = await prisma.user.findUnique({ where: { email } });
  if (existingUser) {
    throw new Error('Email already exists');
  }

  const hashedPassword = await bcrypt.hash(password, 10);

  const user = await prisma.user.create({
    data: {
      email,
      name,
      passwordHash: hashedPassword,
    },
  });
  
  return user;
};

// 로그인
export const loginUser = async (email: string, password: string, app: FastifyInstance) => {
  const user = await prisma.user.findUnique({ where: { email } });
  
  if (!user || !user.passwordHash) {
    throw new Error('Invalid credentials');
  }

  const isMatch = await bcrypt.compare(password, user.passwordHash);
  if (!isMatch) {
    throw new Error('Invalid credentials');
  }

  // Fastify JWT 플러그인 사용
  const token = app.jwt.sign({ 
    id: user.id.toString(), 
    email: user.email, 
    name: user.name 
  });

  return { user, token };
};