import { prisma } from '../../config/db';
import fs from 'fs';
import path from 'path';

/**
 * 집 생성 (이미지 및 3D 모델 파일 업로드 지원)
 */
export const createHome = async (
  userId: string, 
  name: string, 
  addressLine?: string, 
  imageFile?: any, 
  modelFile?: any // 🔥 [추가] 모델 파일(glb 등)을 받는 인자 추가
) => {
  
  let imageUrl: string | null = null;
  let modelUrl: string | null = null;

  // 업로드 디렉토리 경로 설정
  const uploadDir = path.join(process.cwd(), 'uploads');

  // 이미지나 모델 파일 중 하나라도 있으면 폴더 확인 및 생성
  if ((imageFile || modelFile) && !fs.existsSync(uploadDir)) {
    fs.mkdirSync(uploadDir, { recursive: true });
  }

  // 1. 이미지 파일 처리
  if (imageFile) {
    const filename = `${Date.now()}_img_${imageFile.filename}`;
    const savePath = path.join(uploadDir, filename);

    // 스트림 대신 버퍼로 저장 (파일 깨짐 방지)
    const buffer = await imageFile.toBuffer();
    await fs.promises.writeFile(savePath, buffer);
    
    imageUrl = `/uploads/${filename}`;
  }

  // 2. 🔥 [추가] 모델 파일 처리 (.glb 등)
  if (modelFile) {
    // 파일명 중복 방지를 위해 타임스탬프 추가
    const filename = `${Date.now()}_model_${modelFile.filename}`;
    const savePath = path.join(uploadDir, filename);

    // 버퍼로 변환하여 저장
    const buffer = await modelFile.toBuffer();
    await fs.promises.writeFile(savePath, buffer);
    
    modelUrl = `/uploads/${filename}`;
  }

  // DB에 집 정보 생성
  const home = await prisma.home.create({
    data: {
      name,
      addressLine,
      imageUrl,
      modelUrl, // 🔥 [추가] DB에 모델 URL 저장
      ownerId: parseInt(userId),
    },
  });

  // 생성자를 OWNER로 멤버 등록
  await prisma.homeMember.create({
    data: {
      homeId: home.id,
      userId: parseInt(userId),
      role: 'OWNER',
    },
  });

  return home;
};

/**
 * 유저가 속한 집 목록 조회
 */
export const getUserHomes = async (userId: string) => {
  const memberships = await prisma.homeMember.findMany({
    where: { userId: parseInt(userId), isActive: true },
    include: {
      home: {
        include: {
          _count: { select: { devices: true } },
        },
      },
    },
  });

  return memberships.map((m) => ({
    id: m.home.id.toString(),
    name: m.home.name,
    addressLine: m.home.addressLine,
    role: m.role,
    deviceCount: m.home._count.devices,
    imageUrl: m.home.imageUrl,
    modelUrl: m.home.modelUrl, // 🔥 [추가] 목록에서도 모델 유무 확인 가능하도록 추가
  }));
};

/**
 * 집 상세 정보 조회
 */
export const getHomeDetail = async (homeId: string, userId: string) => {
    const membership = await prisma.homeMember.findUnique({
        where: {
            homeId_userId: {
                homeId: parseInt(homeId),
                userId: parseInt(userId)
            }
        }
    });

    if (!membership) throw new Error("접근 권한이 없습니다.");

    const home = await prisma.home.findUnique({
        where: { id: parseInt(homeId) },
        include: {
            devices: true,
            roomLabels: true
        }
    });

    // home 객체 안에 modelUrl이 이미 포함되어 있으므로 그대로 리턴
    return home;
}