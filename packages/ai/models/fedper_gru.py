# LOCUS/packages/ai/models/fedper_gru.py
from __future__ import annotations

from pathlib import Path
from typing import List, Optional

import numpy as np
import tensorflow as tf

from packages.federated.config import (
    SEQUENCE_LENGTH,
    CONTEXT_DIM,
    GRU_HIDDEN_DIM,
    GRU_NUM_ZONES,
    GRU_MODEL_PATH,
)


class FedPerGRUModel:
    """
    FedPer 스타일 GRU 모델

    - Base: GRU 레이어 (전역 공유, Flower로 학습)
    - Head: Dense 레이어 (집마다 다른 개인화 헤드, 온디바이스로 학습)
    """

    def __init__(
        self,
        num_zones: int = GRU_NUM_ZONES,
        context_dim: int = CONTEXT_DIM,
        hidden_dim: int = GRU_HIDDEN_DIM,
    ) -> None:
        self.num_zones = num_zones
        self.context_dim = context_dim
        self.hidden_dim = hidden_dim

        self.base_model: Optional[tf.keras.Model] = None
        self.head_model: Optional[tf.keras.Model] = None
        self.model: Optional[tf.keras.Model] = None

        self._build_model()

    # ------------------------------------------------------------------ build
    def _build_model(self) -> None:
        """Base + Head + Full model을 한 번에 구성."""

        # Input: (seq_len, context_dim)
        inp = tf.keras.Input(
            shape=(SEQUENCE_LENGTH, self.context_dim),
            name="context_input",
        )

        # Base: GRU (전역 공유)
        # 중요: Flower 분류 모델의 GRU 레이어 이름과 동일하게 "gru"로 맞춘다.
        gru_layer = tf.keras.layers.GRU(
            self.hidden_dim,
            return_sequences=False,
            name="gru",
        )
        h = gru_layer(inp)  # (batch, hidden_dim)

        # Head: 개인화 회귀/스코어 헤드 (zone별 오염도 등)
        # activation은 나중에 필요에 따라 변경 가능
        head_out = tf.keras.layers.Dense(
            self.num_zones,
            activation="linear",
            name="head",
        )(h)

        # Base 모델: Input → GRU 출력
        self.base_model = tf.keras.Model(
            inputs=inp,
            outputs=h,
            name="fedper_base",
        )

        # Head 모델: GRU 출력 → Head 출력
        head_inp = tf.keras.Input(shape=(self.hidden_dim,), name="head_input")
        head_body = self._build_head_layers(head_inp)
        self.head_model = tf.keras.Model(
            inputs=head_inp,
            outputs=head_body,
            name="fedper_head",
        )

        # 전체 모델: Input → GRU → Head
        out = self.head_model(self.base_model(inp))
        self.model = tf.keras.Model(
            inputs=inp,
            outputs=out,
            name="fedper_full",
        )

    def _build_head_layers(self, x: tf.Tensor) -> tf.Tensor:
        """
        Head용 레이어 블록 정의.

        현재는 단일 Dense만 사용하지만,
        필요하면 중간에 Dense/Dropout 등을 넣어 확장 가능.
        """
        x = tf.keras.layers.Dense(
            self.num_zones,
            activation="linear",
            name="head_dense",
        )(x)
        return x

    # --------------------------------------------------------------- compile/save
    def compile_model(
        self,
        learning_rate: float = 5e-4,
        loss: str = "mse",
        metrics: Optional[list] = None,
    ) -> None:
        """
        전체 모델 컴파일. OnDeviceTrainer가 Head만 학습하도록 하기 위해,
        Base는 외부에서 trainable=False로 설정해줘야 한다.
        """
        if metrics is None:
            metrics = ["mae"]

        if self.model is None:
            raise RuntimeError("Model is not built")

        opt = tf.keras.optimizers.Adam(learning_rate=learning_rate)
        self.model.compile(optimizer=opt, loss=loss, metrics=metrics)

    def save(self, path: str | Path) -> None:
        """전체 FedPer 모델 저장."""
        if self.model is None:
            raise RuntimeError("Model is not built")

        path = Path(path).expanduser().resolve()
        path.parent.mkdir(parents=True, exist_ok=True)
        self.model.save(path)

    @classmethod
    def load(cls, path: str | Path) -> "FedPerGRUModel":
        """
        저장된 FedPerGRUModel 전체를 로드.

        주의: SavedModel/keras 형식이 동일하다고 가정.
        """
        path = Path(path).expanduser().resolve()
        full_model = tf.keras.models.load_model(path)

        # 새 인스턴스를 만들고, 내부 모델들을 full_model로 치환
        inst = cls(
            num_zones=full_model.output_shape[-1],
            context_dim=full_model.input_shape[-1],
            hidden_dim=full_model.get_layer("gru").units,
        )
        inst.model = full_model

        # base/head는 layer 이름을 통해 다시 구성
        inp = full_model.get_layer("context_input").input
        gru_out = full_model.get_layer("gru").output
        inst.base_model = tf.keras.Model(inputs=inp, outputs=gru_out, name="fedper_base")

        head_inp = tf.keras.Input(shape=(inst.hidden_dim,), name="head_input")
        head_out = full_model.get_layer("head_dense")(head_inp)
        inst.head_model = tf.keras.Model(inputs=head_inp, outputs=head_out, name="fedper_head")

        return inst

    # ---------------------------------------------------- global base sync (핵심)
    def load_base_from_global_classifier(
        self,
        global_model_path: str | Path = GRU_MODEL_PATH,
    ) -> None:
        """
        Flower로 학습된 전역 GRU 분류 모델(.keras)에서
        GRU(Base) 레이어 가중치만 가져와 base에 심는다.

        전역 모델 구조는 대략:
            Input → GRU(name="gru") → Dense(name="dense")
        라고 가정한다.
        """
        global_model_path = Path(global_model_path).expanduser().resolve()
        if not global_model_path.exists():
            raise FileNotFoundError(f"Global classifier model not found: {global_model_path}")

        global_model = tf.keras.models.load_model(global_model_path)

        # 전역 모델에서 GRU 레이어 찾기
        try:
            global_gru = global_model.get_layer("gru")
        except ValueError as e:  # 레이어 이름이 다를 때
            raise RuntimeError(
                "Global model does not have a 'gru' layer. "
                "Ensure the layer name matches."
            ) from e

        local_gru = self.model.get_layer("gru")  # 우리 모델의 GRU

        # 가중치 복사
        local_gru.set_weights(global_gru.get_weights())
        print(
            f"[FedPerGRUModel] Loaded base GRU weights from global classifier: "
            f"{global_model_path}"
        )

    def set_base_weights_from_list(self, weights: List[np.ndarray]) -> None:
        """
        전체 weight 리스트에서 GRU에 해당하는 부분만 골라 설정하고 싶을 때 사용.
        예: Flower 체크포인트 round_*.npy에서 특정 위치의 weight를 GRU로 쓰고 싶을 때.
        """
        if self.model is None:
            raise RuntimeError("Model is not built")

        gru_layer = self.model.get_layer("gru")

        # 간단한 버전: weights 안에 GRU용 weight 세트를 그대로 넣었다고 가정
        # (kernel, recurrent_kernel, bias) 3개
        if len(weights) != 3:
            raise ValueError(
                "Expected list of 3 arrays for GRU weights "
                "(kernel, recurrent_kernel, bias)."
            )

        gru_layer.set_weights(weights)
        print("[FedPerGRUModel] Base GRU weights updated from provided list.")
