// ios/ExpoArkitModule.swift

import ExpoModulesCore
import ARKit

public class ExpoArkitModule: Module {
  fileprivate var arSession: ARSession?
  fileprivate var originTransform: simd_float4x4?
  fileprivate var isTracking = false

  private let sessionDelegate = ExpoArkitSessionDelegate()

  public func definition() -> ModuleDefinition {
    // JS에서 requireNativeModule('ExpoArkit') 로 부를 이름
    Name("ExpoArkit")

    // JS로 쏠 이벤트 이름
    Events("onARFrame", "onTrackingStateChanged")

    OnCreate {
      self.sessionDelegate.owner = self
    }

    // 세션 시작
    AsyncFunction("startSession") { (promise: Promise) in
      DispatchQueue.main.async {
        self.startARSession()
        promise.resolve(true)
      }
    }

    // 세션 중지
    AsyncFunction("stopSession") { (promise: Promise) in
      DispatchQueue.main.async {
        self.arSession?.pause()
        self.isTracking = false
        promise.resolve(true)
      }
    }

    // 원점 재설정
    AsyncFunction("resetOrigin") { (promise: Promise) in
      DispatchQueue.main.async {
        if let frame = self.arSession?.currentFrame {
          self.originTransform = frame.camera.transform
          self.sendEvent("onTrackingStateChanged", [
            "type": "origin_reset",
            "message": "원점이 재설정되었습니다"
          ])
          promise.resolve(true)
        } else {
          promise.reject("NO_FRAME", "현재 프레임을 가져올 수 없습니다")
        }
      }
    }

    // 현재 추적 여부 조회 (선택)
    Function("isTracking") {
      return self.isTracking
    }
  }

  // MARK: - 내부 로직

  private func startARSession() {
    let session = ARSession()
    session.delegate = sessionDelegate
    sessionDelegate.owner = self

    let configuration = ARWorldTrackingConfiguration()

    if ARWorldTrackingConfiguration.supportsSceneReconstruction(.mesh) {
      configuration.sceneReconstruction = .mesh
    }
    configuration.planeDetection = [.horizontal, .vertical]
    configuration.worldAlignment = .gravity

    session.run(configuration)

    self.arSession = session
    self.isTracking = true

    print("✅ ARKit 세션 시작 (Expo)")
  }

  fileprivate func handleFrame(_ frame: ARFrame) {
    guard isTracking else { return }

    let position = calculateRelativePosition(from: frame.camera.transform)
    let (trackingState, accuracy) = getTrackingQuality(state: frame.camera.trackingState)

    let eulerAngles = frame.camera.eulerAngles
    let yawDegrees = Double(eulerAngles.y) * 180.0 / .pi

    sendEvent("onARFrame", [
      "position": position,
      "trackingState": trackingState,
      "accuracy": accuracy,
      "yaw": yawDegrees,
      "timestamp": Date().timeIntervalSince1970 * 1000
    ])
  }

  fileprivate func handleError(_ error: Error) {
    print("❌ ARKit 오류: \(error.localizedDescription)")
    sendEvent("onTrackingStateChanged", [
      "type": "error",
      "message": error.localizedDescription
    ])
  }

  fileprivate func calculateRelativePosition(from transform: simd_float4x4) -> [String: Double] {
    if originTransform == nil {
      originTransform = transform
    }

    let position = simd_make_float3(transform.columns.3)
    let origin = simd_make_float3(originTransform!.columns.3)
    let relative = position - origin

    return [
      "x": Double(relative.x),
      "y": Double(relative.y),
      "z": Double(relative.z)
    ]
  }

  fileprivate func getTrackingQuality(state: ARCamera.TrackingState) -> (String, Double) {
    switch state {
    case .normal:
      return ("normal", 0.05)
    case .notAvailable:
      return ("not_available", 10.0)
    case .limited(let reason):
      switch reason {
      case .initializing:
        return ("initializing", 0.5)
      case .excessiveMotion:
        return ("excessive_motion", 0.3)
      case .insufficientFeatures:
        return ("insufficient_features", 0.5)
      case .relocalizing:
        return ("relocalizing", 0.2)
      @unknown default:
        return ("limited", 0.5)
      }
    }
  }
}

// ARSessionDelegate

private class ExpoArkitSessionDelegate: NSObject, ARSessionDelegate {
  weak var owner: ExpoArkitModule?

  func session(_ session: ARSession, didUpdate frame: ARFrame) {
    owner?.handleFrame(frame)
  }

  func session(_ session: ARSession, didFailWithError error: Error) {
    owner?.handleError(error)
  }
}
