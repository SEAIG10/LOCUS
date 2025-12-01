from __future__ import annotations

import argparse
import threading

from packages.gateway.audio_bridge_win import WindowsAudioBridge
from packages.gateway.policy_bridge import PolicyBridge
from packages.gateway.visual_bridge_win import WindowsVisualBridge


def run_policy(backend_url: str, device_id: str) -> None:
    PolicyBridge(backend_url=backend_url, device_id=device_id).run()


def run_audio() -> None:
    WindowsAudioBridge().run()


def run_visual() -> None:
    WindowsVisualBridge().run()


def main() -> None:
    parser = argparse.ArgumentParser(description="LOCUS IoT Gateway")
    parser.add_argument("--policy", action="store_true", help="Enable FR3→FR5 policy bridge")
    parser.add_argument("--audio", action="store_true", help="Enable Windows audio bridge")
    parser.add_argument("--visual", action="store_true", help="Enable Windows visual bridge")
    parser.add_argument("--backend-url", default="http://localhost:4000")
    parser.add_argument("--device-id", default="robot_001")
    args = parser.parse_args()

    threads: list[threading.Thread] = []
    if args.policy:
        threads.append(
            threading.Thread(
                target=run_policy,
                kwargs={"backend_url": args.backend_url, "device_id": args.device_id},
                daemon=True,
            )
        )
    if args.audio:
        threads.append(threading.Thread(target=run_audio, daemon=True))
    if args.visual:
        threads.append(threading.Thread(target=run_visual, daemon=True))

    if not threads:
        parser.error("At least one bridge (--policy/--audio/--visual) must be enabled.")

    for thread in threads:
        thread.start()

    print("[IoT Gateway] Bridges running (Ctrl+C to stop)...")
    try:
        for thread in threads:
            thread.join()
    except KeyboardInterrupt:
        print("\n[IoT Gateway] Shutting down.")


if __name__ == "__main__":
    main()
