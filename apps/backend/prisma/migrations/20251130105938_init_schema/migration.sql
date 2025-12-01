-- AlterTable
ALTER TABLE "homes" ADD COLUMN     "image_url" TEXT,
ADD COLUMN     "model_url" TEXT;

-- AlterTable
ALTER TABLE "room_labels" ALTER COLUMN "robot_map_id" DROP NOT NULL;

-- AlterTable
ALTER TABLE "sensor_events" ADD COLUMN     "snapshot_x" DOUBLE PRECISION,
ADD COLUMN     "snapshot_y" DOUBLE PRECISION,
ADD COLUMN     "snapshot_z" DOUBLE PRECISION;
