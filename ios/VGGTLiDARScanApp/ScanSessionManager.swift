import ARKit
import Foundation
import SwiftUI

@MainActor
final class ScanSessionManager: NSObject, ObservableObject {
    let session = ARSession()

    @Published private(set) var isSupported = false
    @Published private(set) var isRecording = false
    @Published private(set) var capturedFrameCount = 0
    @Published private(set) var statusText = "Checking LiDAR"
    @Published private(set) var lastPackageURL: URL?
    @Published private(set) var resultURL: URL?
    @Published private(set) var isUploading = false
    @Published private(set) var lastErrorText: String?
    @Published private(set) var lastScaleText: String?

    private let exporter = ScanPackageExporter()
    private let captureQueue = DispatchQueue(label: "edu.ssu.vggt-lidar.capture")
    private var lastCaptureTimestamp: TimeInterval = 0
    private let captureInterval: TimeInterval = 0.35

    override init() {
        super.init()
        session.delegate = self
    }

    func startSession() {
        guard ARWorldTrackingConfiguration.isSupported else {
            isSupported = false
            statusText = "ARKit unavailable"
            return
        }

        let supportsDepth = ARWorldTrackingConfiguration.supportsFrameSemantics(.sceneDepth)
        let supportsSmoothedDepth = ARWorldTrackingConfiguration.supportsFrameSemantics(.smoothedSceneDepth)
        isSupported = supportsDepth || supportsSmoothedDepth
        guard isSupported else {
            statusText = "LiDAR depth unavailable"
            return
        }

        let configuration = ARWorldTrackingConfiguration()
        if supportsSmoothedDepth {
            configuration.frameSemantics.insert(.smoothedSceneDepth)
        } else if supportsDepth {
            configuration.frameSemantics.insert(.sceneDepth)
        }
        if ARWorldTrackingConfiguration.supportsSceneReconstruction(.mesh) {
            configuration.sceneReconstruction = .mesh
        }
        configuration.environmentTexturing = .automatic
        session.run(configuration, options: [.resetTracking, .removeExistingAnchors])
        statusText = "Ready"
    }

    func startScan() {
        guard isSupported else { return }
        do {
            try exporter.beginScan(lidarSupported: isSupported)
            capturedFrameCount = 0
            lastPackageURL = nil
            resultURL = nil
            lastErrorText = nil
            lastScaleText = nil
            lastCaptureTimestamp = 0
            isRecording = true
            statusText = "Scanning"
        } catch {
            statusText = "Export setup failed"
            lastErrorText = error.localizedDescription
        }
    }

    func stopScan() {
        guard isRecording else { return }
        isRecording = false
        statusText = "Packaging"
        let exporter = exporter
        captureQueue.async { [weak self, exporter] in
            guard let self else { return }
            do {
                let url = try exporter.finishScan()
                DispatchQueue.main.async {
                    self.lastPackageURL = url
                    self.statusText = "Package ready"
                    self.lastErrorText = nil
                }
            } catch {
                DispatchQueue.main.async {
                    self.statusText = "Package failed"
                    self.lastErrorText = error.localizedDescription
                }
            }
        }
    }

    func uploadLatestPackage(
        backendBaseURL: String,
        runVGGT: Bool,
        preserveColor: Bool,
        extractObject: Bool,
        reconstructMesh: Bool
    ) async {
        guard let lastPackageURL else { return }
        guard let baseURL = URL(string: backendBaseURL) else {
            statusText = "Bad backend URL"
            return
        }

        isUploading = true
        statusText = "Uploading"
        lastErrorText = nil
        do {
            let client = BackendClient(baseURL: baseURL)
            let result = try await client.reconstruct(
                packageURL: lastPackageURL,
                runVGGT: runVGGT,
                preserveColor: preserveColor,
                extractObject: extractObject,
                reconstructMesh: reconstructMesh
            )
            resultURL = result.outputURL
            if let metrics = result.metrics, metrics.finalOutputType == "mesh", metrics.meshFaces > 0 {
                statusText = "Mesh ready"
            } else if result.metrics?.finalOutputSource == "lidar_metric" {
                statusText = "Metric points ready"
            } else if runVGGT, let metrics = result.metrics, metrics.vggtPoints > 0 {
                statusText = "VGGT ready"
            } else {
                statusText = "Result ready"
            }
            if let warnings = result.metrics?.warnings, !warnings.isEmpty {
                lastErrorText = warnings.joined(separator: "\n")
            }
            lastScaleText = result.metrics?.scaleSummary
        } catch {
            statusText = "Backend failed"
            lastErrorText = error.localizedDescription
        }
        isUploading = false
    }
}

extension ScanSessionManager: ARSessionDelegate {
    nonisolated func session(_ session: ARSession, didUpdate frame: ARFrame) {
        Task { @MainActor in
            guard isRecording else { return }
            guard frame.timestamp - lastCaptureTimestamp >= captureInterval else { return }
            lastCaptureTimestamp = frame.timestamp
            let nextIndex = capturedFrameCount + 1
            capturedFrameCount = nextIndex

            captureQueue.async { [weak self, exporter] in
                do {
                    try exporter.append(frame: frame, index: nextIndex)
                } catch {
                    DispatchQueue.main.async {
                        self?.statusText = "Frame skipped"
                        self?.lastErrorText = error.localizedDescription
                    }
                }
            }
        }
    }

    nonisolated func session(_ session: ARSession, cameraDidChangeTrackingState camera: ARCamera) {
        Task { @MainActor in
            switch camera.trackingState {
            case .normal where isRecording:
                statusText = "Scanning"
            case .normal:
                statusText = "Ready"
            case .notAvailable:
                statusText = "Tracking unavailable"
            case .limited(let reason):
                statusText = "Limited: \(reason.shortLabel)"
            }
        }
    }
}

private extension BackendMetrics {
    var scaleSummary: String? {
        guard let extent = objectExtentM ?? lidarExtentM, extent.count == 3 else { return nil }
        let size = extent.map { String(format: "%.2f", $0) }.joined(separator: " x ")
        if let cameraPathM {
            return "Scale \(size) m · path \(String(format: "%.2f", cameraPathM)) m"
        }
        return "Scale \(size) m"
    }
}

private extension ARCamera.TrackingState.Reason {
    var shortLabel: String {
        switch self {
        case .initializing: return "initializing"
        case .excessiveMotion: return "slow down"
        case .insufficientFeatures: return "need texture"
        case .relocalizing: return "relocalizing"
        @unknown default: return "unknown"
        }
    }
}
