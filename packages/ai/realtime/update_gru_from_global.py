# LOCUS/packages/ai/realtime/update_gru_from_global.py
from __future__ import annotations

from pathlib import Path

from packages.federated.checkpoint_utils import load_latest_global_weights
from packages.federated.config import FEDPER_MODEL_PATH, CONTEXT_DIM, GRU_NUM_ZONES, GRU_HIDDEN_DIM
from packages.ai.models.fedper_gru import FedPerGRUModel


def main() -> None:
    """
    최신 FL 글로벌 체크포인트(round_*.npy)를 읽어서
    FedPerGRUModel의 Base(GRU) weight만 업데이트합니다.

    - Head(개인화)는 그대로 유지
    - FedPer 모델이 없으면 새로 만들고 Base만 글로벌로 세팅
    """

    # 1) Flower 서버 글로벌 weights 로드
    global_weights = load_latest_global_weights()

    if len(global_weights) < 3:
        raise ValueError(
            "Global weights list is too short. "
            "[가정: GRU kernel/recurrent/bias 3개 + Dense kernel/bias 2개 이상이어야 함]"
        )

    # [가정] weights[:3] = GRU (kernel, recurrent_kernel, bias)
    gru_weights = global_weights[:3]

    # 2) FedPer 모델 로컬 파일 경로
    model_path = Path(FEDPER_MODEL_PATH).expanduser().resolve()
    model_path.parent.mkdir(parents=True, exist_ok=True)

    # 3) 기존 FedPer 모델 있으면 로드, 없으면 새로 생성
    if model_path.exists():
        print(f"[update_gru_from_global] Loading existing FedPer model: {model_path}")
        fedper = FedPerGRUModel.load(model_path)
    else:
        print(f"[update_gru_from_global] No FedPer model found, creating new one: {model_path}")
        fedper = FedPerGRUModel(
            num_zones=GRU_NUM_ZONES,
            context_dim=CONTEXT_DIM,
            hidden_dim=GRU_HIDDEN_DIM,
        )

    # 4) Base(GRU) weight만 글로벌로 업데이트
    fedper.set_base_weights_from_list(gru_weights)

    # 5) FedPer 모델 저장 (Base+Head 통합)
    fedper.save(model_path)
    print(f"[update_gru_from_global] FedPer model updated and saved to: {model_path}")


if __name__ == "__main__":
    main()
