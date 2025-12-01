import { FastifyInstance } from 'fastify';
// 🔥 deleteHomeHandler 추가 import
import { createHomeHandler, getMyHomesHandler, getHomeDetailHandler, deleteHomeHandler } from './homes.controller';

export async function homesRoutes(app: FastifyInstance) {
  // 로그인 체크
  app.addHook('onRequest', async (request) => {
    await request.jwtVerify(); 
  });

  app.post('/', createHomeHandler);       
  app.get('/', getMyHomesHandler);        
  app.get('/:id', getHomeDetailHandler);
  
  // 🔥 이 줄이 빠져 있어서 삭제가 안 됐던 것입니다!
  app.delete('/:id', deleteHomeHandler);
}