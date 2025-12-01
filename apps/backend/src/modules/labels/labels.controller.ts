import { FastifyReply, FastifyRequest } from 'fastify';
import * as labelsService from './labels.service';

export async function getLabelsHandler(
  request: FastifyRequest<{ Params: { homeId: string } }>,
  reply: FastifyReply
) {
  try {
    const { homeId } = request.params;
    const labels = await labelsService.getHomeLabels(homeId);
    return reply.code(200).send(labels);
  } catch (e: any) {
    console.error("[Get Labels Error]", e);
    return reply.code(500).send({ message: e.message });
  }
}

export async function createLabelHandler(
  request: FastifyRequest<{ Params: { homeId: string }, Body: { name: string, points: any[] } }>,
  reply: FastifyReply
) {
  try {
    const { homeId } = request.params;
    const { name, points } = request.body;

    console.log(`[Create Label] Home: ${homeId}, Name: ${name}, Points: ${points?.length}`);

    const label = await labelsService.createLabel(homeId, name, points);
    return reply.code(201).send(label);
  } catch (e: any) {
    console.error("[Create Label Error]:", e);
    return reply.code(500).send({ message: e.message });
  }
}

export async function deleteLabelHandler(
  request: FastifyRequest<{ Params: { homeId: string; labelId: string } }>,
  reply: FastifyReply
) {
  try {
    const { homeId, labelId } = request.params;
    await labelsService.deleteLabel(labelId, homeId);
    return reply.code(200).send({ message: "Deleted successfully" });
  } catch (e: any) {
    console.error("[Delete Label Error]", e);
    return reply.code(500).send({ message: e.message });
  }
}