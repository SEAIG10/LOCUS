import { prisma } from '../../config/db';
import { notifyZoneUpdate } from '../mqtt/mqtt.service';

export const createLabel = async (homeId: string, name: string, points: any[]) => {
  // 라벨 메타데이터 생성
  const label = await prisma.roomLabel.create({
    data: {
      homeId: parseInt(homeId),
      name,
      labelType: 'ROOM',
    }
  });

  // 폴리곤 좌표 저장
  if (points && points.length > 0) {
    await Promise.all(points.map((p, index) =>
      prisma.roomLabelPolygonPoint.create({
        data: {
          labelId: label.id,
          orderIndex: index,
          x: parseFloat(p.x),
          z: parseFloat(p.z)
        }
      })
    ));
  }

  // MQTT로 엣지 디바이스에 구역 변경 알림
  await notifyZoneUpdate(homeId);

  return label;
};

export const getHomeLabels = async (homeId: string) => {
  const labels = await prisma.roomLabel.findMany({
    where: { homeId: parseInt(homeId) },
    include: {
      points: {
        orderBy: { orderIndex: 'asc' }
      }
    }
  });

  return labels.map(label => ({
    ...label,
    id: label.id.toString(),
    homeId: label.homeId.toString(),
  }));
};

export const deleteLabel = async (labelId: string, homeId: string) => {
  const result = await prisma.roomLabel.delete({
    where: { id: parseInt(labelId) }
  });

  // MQTT로 엣지 디바이스에 구역 변경 알림
  await notifyZoneUpdate(homeId);

  return result;
};