/**
 * useExpoARTracking.ts
 * Expo Module (ExpoArkit) 기반 ARKit 추적 훅
 */

import { useEffect, useState, useCallback } from "react";
import { Platform } from "react-native";
import { requireNativeModule, EventEmitter } from "expo-modules-core";

export interface ARPosition {
  x: number;
  y: number;
  z: number;
}

export interface ARTrackingData {
  position: ARPosition;
  trackingState:
    | "normal"
    | "not_available"
    | "initializing"
    | "excessive_motion"
    | "insufficient_features"
    | "relocalizing"
    | "limited"
    | string;
  accuracy: number;
  yaw: number;
  timestamp: number;
}

export interface UseExpoARTrackingOptions {
  autoStart?: boolean;
  onError?: (error: string) => void;
}

// ⚠️ Swift 쪽 Name("ExpoArkit") 과 정확히 일치해야 함
const MODULE_NAME = "ExpoArkit";

type ExpoArkitModuleType = {
  startSession: () => Promise<boolean>;
  stopSession: () => Promise<boolean>;
  resetOrigin: () => Promise<boolean>;
};

// 네이티브 모듈 & 이벤트 emitter 안전 로드
let NativeModule: ExpoArkitModuleType | null = null;
let arEmitter: EventEmitter<any> | null = null;

try {
  NativeModule = requireNativeModule<ExpoArkitModuleType>(MODULE_NAME);
  arEmitter = new EventEmitter<any>(NativeModule as any);
} catch (e) {
  console.warn(`[ExpoArkit] native module 로드 실패:`, e);
  NativeModule = null;
  arEmitter = null;
}

export function useExpoARTracking(options: UseExpoARTrackingOptions = {}) {
  const { autoStart = true, onError } = options;

  const [trackingData, setTrackingData] = useState<ARTrackingData | null>(null);
  const [isTracking, setIsTracking] = useState(false);
  const [error, setError] = useState<string | null>(null);

  // 공통 에러 처리 헬퍼
  const reportError = useCallback(
    (msg: string) => {
      setError(msg);
      onError?.(msg);
    },
    [onError]
  );

  // ARKit 세션 시작
  const startTracking = useCallback(async () => {
    if (Platform.OS !== "ios") {
      reportError("ARKit은 iOS에서만 사용 가능합니다");
      return;
    }

    if (!NativeModule || typeof NativeModule.startSession !== 'function') {
        const msg = 'ARKit native module이 없거나 startSession이 구현되지 않았습니다';
        reportError(msg);
        return;
    }

    try {
      await NativeModule.startSession();
      setIsTracking(true);
      setError(null);
      console.log("✅ ARKit 추적 시작 (Expo)");
    } catch (err: any) {
      const msg = `ARKit 시작 실패: ${String(err?.message ?? err)}`;
      reportError(msg);
      console.error(msg);
    }
  }, [reportError]);

  // ARKit 세션 중지
  const stopTracking = useCallback(async () => {
    if (!NativeModule) {
      console.warn("[ExpoArkit] stopSession 호출: 모듈이 로드되지 않음");
      setIsTracking(false);
      return;
    }

    try {
      await NativeModule.stopSession();
      setIsTracking(false);
      console.log("🛑 ARKit 추적 중지 (Expo)");
    } catch (err: any) {
      console.error("ARKit 중지 실패:", err?.message ?? err);
    }
  }, []);

  // 원점 재설정
  const resetOrigin = useCallback(async () => {
    if (!NativeModule) {
      console.warn("[ExpoArkit] resetOrigin 호출: 모듈이 로드되지 않음");
      return;
    }

    try {
      await NativeModule.resetOrigin();
      console.log("📍 ARKit 원점 재설정 (Expo)");
    } catch (err: any) {
      console.error("원점 재설정 실패:", err?.message ?? err);
    }
  }, []);

  useEffect(() => {
    // 네이티브 모듈/Emitter가 없으면 그냥 에러만 띄우고 끝
    if (!NativeModule || !arEmitter) {
      reportError("ExpoArkit 네이티브 모듈 또는 이벤트 emitter가 초기화되지 않았습니다");
      return;
    }

    // ARFrame 업데이트 리스너
    const frameSubscription = arEmitter.addListener(
      "onARFrame",
      (data: ARTrackingData) => {
        setTrackingData(data);
      }
    );

    // 추적 상태 변경 리스너
    const stateSubscription = arEmitter.addListener(
      "onTrackingStateChanged",
      (event: { type: string; message?: string }) => {
        console.log("[ExpoArkit] 상태 변경:", event.type, event.message);

        if (event.type === "error") {
          const msg = String(event.message ?? "알 수 없는 오류");
          reportError(msg);
        }
      }
    );

    // 자동 시작
    if (autoStart) {
      startTracking();
    }

    return () => {
      frameSubscription.remove();
      stateSubscription.remove();

      if (isTracking) {
        // cleanup 시 에러 나도 앱 안 죽게 try/catch
        stopTracking().catch((err) =>
          console.error("cleanup 중 stopTracking 실패:", err)
        );
      }
    };
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [autoStart, startTracking, stopTracking, reportError]);

  return {
    // 현재 위치 데이터
    position: trackingData?.position ?? null,
    trackingState: trackingData?.trackingState ?? "not_available",
    accuracy: trackingData?.accuracy ?? 0,
    yaw: trackingData?.yaw ?? 0,
    timestamp: trackingData?.timestamp ?? 0,

    // 상태
    isTracking,
    error,

    // 제어 함수
    startTracking,
    stopTracking,
    resetOrigin,
  };
}
