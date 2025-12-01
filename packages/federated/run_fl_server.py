"""Entry point for the MQTT-based FedPer aggregation server."""

import argparse

from config import (
    CLIENTS_PER_ROUND,
    MQTT_BROKER_HOST,
    MQTT_BROKER_PORT,
    MQTT_KEEPALIVE,
    MQTT_TOPIC_NAMESPACE,
    PRETRAINED_MODEL_PATH,
)
from server import MQTTFLServer


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="LOCUS FedPer MQTT server")
    parser.add_argument(
        "--broker-host", default=MQTT_BROKER_HOST, help="MQTT broker hostname"
    )
    parser.add_argument(
        "--broker-port", default=MQTT_BROKER_PORT, type=int, help="MQTT broker port"
    )
    parser.add_argument(
        "--topic-namespace",
        default=MQTT_TOPIC_NAMESPACE,
        help="Base topic namespace used for FL coordination",
    )
    parser.add_argument(
        "--keepalive",
        default=MQTT_KEEPALIVE,
        type=int,
        help="MQTT keepalive interval in seconds",
    )
    parser.add_argument(
        "--clients-per-round",
        default=CLIENTS_PER_ROUND,
        type=int,
        help="Number of client updates to wait for before aggregating",
    )
    parser.add_argument(
        "--server-id",
        default="locus_fl_server",
        help="Unique MQTT client id for the server",
    )
    parser.add_argument(
        "--model-path",
        default=str(PRETRAINED_MODEL_PATH),
        help="Path to the pretrained Keras GRU model",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    server = MQTTFLServer(
        broker_host=args.broker_host,
        broker_port=args.broker_port,
        topic_namespace=args.topic_namespace,
        clients_per_round=args.clients_per_round,
        keepalive=args.keepalive,
        server_id=args.server_id,
        model_path=args.model_path,
    )
    server.serve_forever()


if __name__ == "__main__":
    main()
