/*
  Warnings:

  - You are about to drop the `User` table. If the table is not empty, all the data it contains will be lost.

*/
-- CreateEnum
CREATE TYPE "UserStatus" AS ENUM ('ACTIVE', 'INACTIVE', 'DELETED');

-- CreateEnum
CREATE TYPE "ProviderType" AS ENUM ('LOCAL', 'GOOGLE', 'APPLE');

-- CreateEnum
CREATE TYPE "HomeStatus" AS ENUM ('ACTIVE', 'DISABLED');

-- CreateEnum
CREATE TYPE "MemberRole" AS ENUM ('OWNER', 'MEMBER', 'VIEWER');

-- CreateEnum
CREATE TYPE "DeviceType" AS ENUM ('ROBOT_VACUUM', 'AIR_PURIFIER', 'OTHER');

-- CreateEnum
CREATE TYPE "DeviceStatus" AS ENUM ('ONLINE', 'OFFLINE', 'ERROR');

-- CreateEnum
CREATE TYPE "LabelType" AS ENUM ('ROOM', 'ZONE');

-- CreateEnum
CREATE TYPE "LocationSource" AS ENUM ('ROBOT', 'MOBILE', 'SIMULATOR');

-- CreateEnum
CREATE TYPE "EventType" AS ENUM ('AUDIO', 'VISION', 'SYSTEM', 'USER_ACTION');

-- CreateEnum
CREATE TYPE "Severity" AS ENUM ('INFO', 'WARNING', 'CRITICAL');

-- CreateEnum
CREATE TYPE "PollutionType" AS ENUM ('CLEANING_NEEDED');

-- DropTable
DROP TABLE "User";

-- CreateTable
CREATE TABLE "users" (
    "id" SERIAL NOT NULL,
    "email" TEXT NOT NULL,
    "password_hash" TEXT,
    "name" TEXT NOT NULL,
    "status" "UserStatus" NOT NULL DEFAULT 'ACTIVE',
    "last_login_at" TIMESTAMP(3),
    "created_at" TIMESTAMP(3) NOT NULL DEFAULT CURRENT_TIMESTAMP,
    "updated_at" TIMESTAMP(3) NOT NULL,

    CONSTRAINT "users_pkey" PRIMARY KEY ("id")
);

-- CreateTable
CREATE TABLE "auth_providers" (
    "id" SERIAL NOT NULL,
    "user_id" INTEGER NOT NULL,
    "provider" "ProviderType" NOT NULL,
    "provider_user_id" TEXT NOT NULL,
    "created_at" TIMESTAMP(3) NOT NULL DEFAULT CURRENT_TIMESTAMP,

    CONSTRAINT "auth_providers_pkey" PRIMARY KEY ("id")
);

-- CreateTable
CREATE TABLE "homes" (
    "id" SERIAL NOT NULL,
    "owner_id" INTEGER NOT NULL,
    "name" TEXT NOT NULL,
    "address_line" TEXT,
    "status" "HomeStatus" NOT NULL DEFAULT 'ACTIVE',
    "created_at" TIMESTAMP(3) NOT NULL DEFAULT CURRENT_TIMESTAMP,
    "updated_at" TIMESTAMP(3) NOT NULL,

    CONSTRAINT "homes_pkey" PRIMARY KEY ("id")
);

-- CreateTable
CREATE TABLE "home_members" (
    "id" SERIAL NOT NULL,
    "home_id" INTEGER NOT NULL,
    "user_id" INTEGER NOT NULL,
    "role" "MemberRole" NOT NULL DEFAULT 'MEMBER',
    "is_active" BOOLEAN NOT NULL DEFAULT true,
    "created_at" TIMESTAMP(3) NOT NULL DEFAULT CURRENT_TIMESTAMP,

    CONSTRAINT "home_members_pkey" PRIMARY KEY ("id")
);

-- CreateTable
CREATE TABLE "devices" (
    "id" SERIAL NOT NULL,
    "home_id" INTEGER NOT NULL,
    "name" TEXT NOT NULL,
    "device_type" "DeviceType" NOT NULL DEFAULT 'ROBOT_VACUUM',
    "model_name" TEXT,
    "serial_number" TEXT,
    "status" "DeviceStatus" NOT NULL DEFAULT 'OFFLINE',
    "last_online_at" TIMESTAMP(3),
    "created_at" TIMESTAMP(3) NOT NULL DEFAULT CURRENT_TIMESTAMP,
    "updated_at" TIMESTAMP(3) NOT NULL,

    CONSTRAINT "devices_pkey" PRIMARY KEY ("id")
);

-- CreateTable
CREATE TABLE "robot_maps" (
    "id" SERIAL NOT NULL,
    "device_id" INTEGER NOT NULL,
    "version" INTEGER NOT NULL,
    "map_name" TEXT,
    "map_json" JSONB,
    "width" DOUBLE PRECISION,
    "height" DOUBLE PRECISION,
    "is_active" BOOLEAN NOT NULL DEFAULT true,
    "created_at" TIMESTAMP(3) NOT NULL DEFAULT CURRENT_TIMESTAMP,

    CONSTRAINT "robot_maps_pkey" PRIMARY KEY ("id")
);

-- CreateTable
CREATE TABLE "map_calibrations" (
    "id" SERIAL NOT NULL,
    "robot_map_id" INTEGER NOT NULL,
    "home_id" INTEGER NOT NULL,
    "rotation_deg" DOUBLE PRECISION NOT NULL DEFAULT 0,
    "sensor_direction_deg" DOUBLE PRECISION NOT NULL DEFAULT 0,
    "scale" DOUBLE PRECISION NOT NULL DEFAULT 1,
    "offset_x" DOUBLE PRECISION NOT NULL DEFAULT 0,
    "offset_z" DOUBLE PRECISION NOT NULL DEFAULT 0,
    "created_by" INTEGER,
    "created_at" TIMESTAMP(3) NOT NULL DEFAULT CURRENT_TIMESTAMP,
    "updated_at" TIMESTAMP(3) NOT NULL,

    CONSTRAINT "map_calibrations_pkey" PRIMARY KEY ("id")
);

-- CreateTable
CREATE TABLE "room_labels" (
    "id" SERIAL NOT NULL,
    "home_id" INTEGER NOT NULL,
    "robot_map_id" INTEGER NOT NULL,
    "name" TEXT NOT NULL,
    "color_hex" TEXT,
    "center_x" DOUBLE PRECISION NOT NULL DEFAULT 0,
    "center_y" DOUBLE PRECISION NOT NULL DEFAULT 0,
    "center_z" DOUBLE PRECISION NOT NULL DEFAULT 0,
    "label_type" "LabelType" NOT NULL DEFAULT 'ROOM',
    "created_by" INTEGER,
    "created_at" TIMESTAMP(3) NOT NULL DEFAULT CURRENT_TIMESTAMP,
    "updated_at" TIMESTAMP(3) NOT NULL,

    CONSTRAINT "room_labels_pkey" PRIMARY KEY ("id")
);

-- CreateTable
CREATE TABLE "room_label_polygon_points" (
    "id" SERIAL NOT NULL,
    "label_id" INTEGER NOT NULL,
    "order_index" INTEGER NOT NULL,
    "x" DOUBLE PRECISION NOT NULL,
    "z" DOUBLE PRECISION NOT NULL,

    CONSTRAINT "room_label_polygon_points_pkey" PRIMARY KEY ("id")
);

-- CreateTable
CREATE TABLE "robot_locations" (
    "id" BIGSERIAL NOT NULL,
    "device_id" INTEGER NOT NULL,
    "robot_map_id" INTEGER,
    "recorded_at" TIMESTAMP(3) NOT NULL,
    "x" DOUBLE PRECISION NOT NULL,
    "y" DOUBLE PRECISION NOT NULL,
    "z" DOUBLE PRECISION NOT NULL,
    "heading_deg" DOUBLE PRECISION,
    "raw_lat" DOUBLE PRECISION,
    "raw_lng" DOUBLE PRECISION,
    "source" "LocationSource" NOT NULL DEFAULT 'ROBOT',
    "raw_payload_json" JSONB,

    CONSTRAINT "robot_locations_pkey" PRIMARY KEY ("id")
);

-- CreateTable
CREATE TABLE "sensor_events" (
    "id" BIGSERIAL NOT NULL,
    "home_id" INTEGER NOT NULL,
    "device_id" INTEGER,
    "event_time" TIMESTAMP(3) NOT NULL,
    "event_type" "EventType" NOT NULL,
    "sub_type" TEXT,
    "severity" "Severity" NOT NULL DEFAULT 'INFO',
    "label_id" INTEGER,
    "robot_location_id" BIGINT,
    "payload_json" JSONB,

    CONSTRAINT "sensor_events_pkey" PRIMARY KEY ("id")
);

-- CreateTable
CREATE TABLE "pollution_predictions" (
    "id" BIGSERIAL NOT NULL,
    "home_id" INTEGER NOT NULL,
    "device_id" INTEGER NOT NULL,
    "robot_map_id" INTEGER,
    "label_id" INTEGER,
    "prediction_time" TIMESTAMP(3) NOT NULL,
    "time_window_start" TIMESTAMP(3),
    "time_window_end" TIMESTAMP(3),
    "pollution_type" "PollutionType" NOT NULL DEFAULT 'CLEANING_NEEDED',
    "probability" DOUBLE PRECISION NOT NULL,
    "model_version" TEXT NOT NULL,
    "feature_summary_json" JSONB,

    CONSTRAINT "pollution_predictions_pkey" PRIMARY KEY ("id")
);

-- CreateIndex
CREATE UNIQUE INDEX "users_email_key" ON "users"("email");

-- CreateIndex
CREATE UNIQUE INDEX "auth_providers_provider_provider_user_id_key" ON "auth_providers"("provider", "provider_user_id");

-- CreateIndex
CREATE UNIQUE INDEX "home_members_home_id_user_id_key" ON "home_members"("home_id", "user_id");

-- CreateIndex
CREATE UNIQUE INDEX "robot_maps_device_id_version_key" ON "robot_maps"("device_id", "version");

-- CreateIndex
CREATE UNIQUE INDEX "room_labels_home_id_name_key" ON "room_labels"("home_id", "name");

-- CreateIndex
CREATE UNIQUE INDEX "room_label_polygon_points_label_id_order_index_key" ON "room_label_polygon_points"("label_id", "order_index");

-- CreateIndex
CREATE INDEX "robot_locations_device_id_recorded_at_idx" ON "robot_locations"("device_id", "recorded_at");

-- CreateIndex
CREATE INDEX "robot_locations_robot_map_id_recorded_at_idx" ON "robot_locations"("robot_map_id", "recorded_at");

-- CreateIndex
CREATE INDEX "sensor_events_home_id_event_time_idx" ON "sensor_events"("home_id", "event_time");

-- CreateIndex
CREATE INDEX "sensor_events_device_id_event_time_idx" ON "sensor_events"("device_id", "event_time");

-- CreateIndex
CREATE INDEX "sensor_events_label_id_event_time_idx" ON "sensor_events"("label_id", "event_time");

-- CreateIndex
CREATE INDEX "pollution_predictions_home_id_prediction_time_idx" ON "pollution_predictions"("home_id", "prediction_time");

-- AddForeignKey
ALTER TABLE "auth_providers" ADD CONSTRAINT "auth_providers_user_id_fkey" FOREIGN KEY ("user_id") REFERENCES "users"("id") ON DELETE CASCADE ON UPDATE CASCADE;

-- AddForeignKey
ALTER TABLE "homes" ADD CONSTRAINT "homes_owner_id_fkey" FOREIGN KEY ("owner_id") REFERENCES "users"("id") ON DELETE RESTRICT ON UPDATE CASCADE;

-- AddForeignKey
ALTER TABLE "home_members" ADD CONSTRAINT "home_members_home_id_fkey" FOREIGN KEY ("home_id") REFERENCES "homes"("id") ON DELETE CASCADE ON UPDATE CASCADE;

-- AddForeignKey
ALTER TABLE "home_members" ADD CONSTRAINT "home_members_user_id_fkey" FOREIGN KEY ("user_id") REFERENCES "users"("id") ON DELETE CASCADE ON UPDATE CASCADE;

-- AddForeignKey
ALTER TABLE "devices" ADD CONSTRAINT "devices_home_id_fkey" FOREIGN KEY ("home_id") REFERENCES "homes"("id") ON DELETE CASCADE ON UPDATE CASCADE;

-- AddForeignKey
ALTER TABLE "robot_maps" ADD CONSTRAINT "robot_maps_device_id_fkey" FOREIGN KEY ("device_id") REFERENCES "devices"("id") ON DELETE CASCADE ON UPDATE CASCADE;

-- AddForeignKey
ALTER TABLE "map_calibrations" ADD CONSTRAINT "map_calibrations_robot_map_id_fkey" FOREIGN KEY ("robot_map_id") REFERENCES "robot_maps"("id") ON DELETE CASCADE ON UPDATE CASCADE;

-- AddForeignKey
ALTER TABLE "map_calibrations" ADD CONSTRAINT "map_calibrations_home_id_fkey" FOREIGN KEY ("home_id") REFERENCES "homes"("id") ON DELETE CASCADE ON UPDATE CASCADE;

-- AddForeignKey
ALTER TABLE "map_calibrations" ADD CONSTRAINT "map_calibrations_created_by_fkey" FOREIGN KEY ("created_by") REFERENCES "users"("id") ON DELETE SET NULL ON UPDATE CASCADE;

-- AddForeignKey
ALTER TABLE "room_labels" ADD CONSTRAINT "room_labels_home_id_fkey" FOREIGN KEY ("home_id") REFERENCES "homes"("id") ON DELETE CASCADE ON UPDATE CASCADE;

-- AddForeignKey
ALTER TABLE "room_labels" ADD CONSTRAINT "room_labels_robot_map_id_fkey" FOREIGN KEY ("robot_map_id") REFERENCES "robot_maps"("id") ON DELETE CASCADE ON UPDATE CASCADE;

-- AddForeignKey
ALTER TABLE "room_labels" ADD CONSTRAINT "room_labels_created_by_fkey" FOREIGN KEY ("created_by") REFERENCES "users"("id") ON DELETE SET NULL ON UPDATE CASCADE;

-- AddForeignKey
ALTER TABLE "room_label_polygon_points" ADD CONSTRAINT "room_label_polygon_points_label_id_fkey" FOREIGN KEY ("label_id") REFERENCES "room_labels"("id") ON DELETE CASCADE ON UPDATE CASCADE;

-- AddForeignKey
ALTER TABLE "robot_locations" ADD CONSTRAINT "robot_locations_device_id_fkey" FOREIGN KEY ("device_id") REFERENCES "devices"("id") ON DELETE CASCADE ON UPDATE CASCADE;

-- AddForeignKey
ALTER TABLE "robot_locations" ADD CONSTRAINT "robot_locations_robot_map_id_fkey" FOREIGN KEY ("robot_map_id") REFERENCES "robot_maps"("id") ON DELETE SET NULL ON UPDATE CASCADE;

-- AddForeignKey
ALTER TABLE "sensor_events" ADD CONSTRAINT "sensor_events_home_id_fkey" FOREIGN KEY ("home_id") REFERENCES "homes"("id") ON DELETE CASCADE ON UPDATE CASCADE;

-- AddForeignKey
ALTER TABLE "sensor_events" ADD CONSTRAINT "sensor_events_device_id_fkey" FOREIGN KEY ("device_id") REFERENCES "devices"("id") ON DELETE SET NULL ON UPDATE CASCADE;

-- AddForeignKey
ALTER TABLE "sensor_events" ADD CONSTRAINT "sensor_events_label_id_fkey" FOREIGN KEY ("label_id") REFERENCES "room_labels"("id") ON DELETE SET NULL ON UPDATE CASCADE;

-- AddForeignKey
ALTER TABLE "sensor_events" ADD CONSTRAINT "sensor_events_robot_location_id_fkey" FOREIGN KEY ("robot_location_id") REFERENCES "robot_locations"("id") ON DELETE SET NULL ON UPDATE CASCADE;

-- AddForeignKey
ALTER TABLE "pollution_predictions" ADD CONSTRAINT "pollution_predictions_home_id_fkey" FOREIGN KEY ("home_id") REFERENCES "homes"("id") ON DELETE CASCADE ON UPDATE CASCADE;

-- AddForeignKey
ALTER TABLE "pollution_predictions" ADD CONSTRAINT "pollution_predictions_device_id_fkey" FOREIGN KEY ("device_id") REFERENCES "devices"("id") ON DELETE CASCADE ON UPDATE CASCADE;

-- AddForeignKey
ALTER TABLE "pollution_predictions" ADD CONSTRAINT "pollution_predictions_robot_map_id_fkey" FOREIGN KEY ("robot_map_id") REFERENCES "robot_maps"("id") ON DELETE SET NULL ON UPDATE CASCADE;

-- AddForeignKey
ALTER TABLE "pollution_predictions" ADD CONSTRAINT "pollution_predictions_label_id_fkey" FOREIGN KEY ("label_id") REFERENCES "room_labels"("id") ON DELETE SET NULL ON UPDATE CASCADE;
