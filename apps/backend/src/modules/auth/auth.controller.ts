import { FastifyReply, FastifyRequest } from 'fastify';
import * as authService from './auth.service';

// 회원가입 핸들러
export async function registerHandler(
  request: FastifyRequest<{ Body: { email: string; password: string; name: string } }>,
  reply: FastifyReply
) {
  try {
    const { email, password, name } = request.body;
    const user = await authService.registerUser(email, password, name);
    
    return reply.code(201).send({ 
      message: 'User created', 
      userId: user.id.toString() 
    });
  } catch (e: any) {
    return reply.code(400).send({ message: e.message });
  }
}

// 로그인 핸들러
export async function loginHandler(
  request: FastifyRequest<{ Body: { email: string; password: string } }>,
  reply: FastifyReply
) {
  try {
    const { email, password } = request.body;
    
    // this context를 사용하거나 app 인스턴스를 넘겨줘야 함. 
    // 여기서는 간단히 request.server(app 인스턴스)를 서비스에 넘깁니다.
    const { user, token } = await authService.loginUser(email, password, request.server);

    return reply.code(200).send({
      message: 'Login successful',
      token,
      user: {
        id: user.id.toString(),
        email: user.email,
        name: user.name,
      },
    });
  } catch (e: any) {
    return reply.code(401).send({ message: e.message });
  }
}