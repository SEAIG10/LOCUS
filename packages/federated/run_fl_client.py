"""Entry point for the FedPer client."""

import argparse

from config import (
    CLIENT_ID,
    LOCAL_EPOCHS,
    LR,
    MQTT_BROKER_HOST,
    MQTT_BROKER_PORT,
    MQTT_KEEPALIVE,
    MQTT_TOPIC_NAMESPACE,
    PRETRAINED_MODEL_PATH,
)
from client import MQTTFLClient


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="LOCUS FedPer client")
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
        help="MQTT keepalive heartbeat in seconds",
    )
    parser.add_argument("--client-id", default=CLIENT_ID, help="Unique client id")
    parser.add_argument(
        "--model-path",
        default=str(PRETRAINED_MODEL_PATH),
        help="Path to the pretrained Keras GRU model",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    client = MQTTFLClient(
        client_id=args.client_id,
        broker_host=args.broker_host,
        broker_port=args.broker_port,
        topic_namespace=args.topic_namespace,
        local_epochs=LOCAL_EPOCHS,
        learning_rate=LR,
        keepalive=args.keepalive,
        model_path=args.model_path,
    )
    client.run()


if __name__ == "__main__":
    main()
