import ARKit
import Foundation
import SwiftUI

@MainActor
final class ScanSessionManager: NSObject, ObservableObject {
    let session = ARSession()

    @Published private(set) var isSupported = false
    @Published private(set) var isRecording = false
    @Published private(set) var isCapturingPhoto = false
    @Published private(set) var capturedFrameCount = 0
    @Published private(set) var statusText = "Checking LiDAR"
    @Published private(set) var lastPackageURL: URL?
    @Published private(set) var resultURL: URL?
    @Published private(set) var resultJobID: String?
    @Published private(set) var resultMetrics: BackendMetrics?
    @Published private(set) var downloadedAssets: [BackendAssetKind: URL] = [:]
    @Published private(set) var downloadingAssets: Set<BackendAssetKind> = []
    @Published private(set) var backendCapabilities: BackendCapabilities?
    @Published private(set) var isCheckingBackend = false
    @Published private(set) var isUploading = false
    @Published private(set) var lastErrorText: String?
    @Published private(set) var lastScaleText: String?

    private let exporter = ScanPackageExporter()
    private let captureQueue = DispatchQueue(label: "edu.ssu.vggt-lidar.capture")
    private var lastCaptureTimestamp: TimeInterval = 0
    private let captureInterval: TimeInterval = 0.35
    private var latestFrame: ARFrame?

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
        guard isSupported, !isCapturingPhoto else { return }
        do {
            try exporter.beginScan(lidarSupported: isSupported)
            capturedFrameCount = 0
            resetOutputState()
            lastCaptureTimestamp = 0
            isRecording = true
            statusText = "Scanning"
        } catch {
            statusText = "Export setup failed"
            lastErrorText = error.localizedDescription
        }
    }

    func capturePhoto() {
        guard isSupported, !isRecording, !isCapturingPhoto else { return }
        guard let frame = latestFrame else {
            statusText = "Camera warming up"
            lastErrorText = "Wait for the camera preview to start, then capture again."
            return
        }

        do {
            try exporter.beginScan(lidarSupported: isSupported)
            capturedFrameCount = 0
            resetOutputState()
            isCapturingPhoto = true
            statusText = "Capturing photo"
        } catch {
            statusText = "Export setup failed"
            lastErrorText = error.localizedDescription
            return
        }

        let exporter = exporter
        captureQueue.async { [weak self, exporter] in
            guard let self else { return }
            do {
                try exporter.append(frame: frame, index: 1)
                let url = try exporter.finishScan()
                DispatchQueue.main.async {
                    self.capturedFrameCount = 1
                    self.lastPackageURL = url
                    self.isCapturingPhoto = false
                    self.statusText = "Photo ready"
                    self.lastErrorText = nil
                }
            } catch {
                DispatchQueue.main.async {
                    self.isCapturingPhoto = false
                    self.statusText = "Capture failed"
                    self.lastErrorText = error.localizedDescription
                }
            }
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

    private func resetOutputState() {
        lastPackageURL = nil
        resultURL = nil
        resultJobID = nil
        resultMetrics = nil
        downloadedAssets = [:]
        downloadingAssets = []
        lastErrorText = nil
        lastScaleText = nil
    }

    func refreshCapabilities(backendBaseURL: String) async {
        guard let baseURL = URL(string: backendBaseURL) else {
            backendCapabilities = nil
            lastErrorText = "Invalid backend URL."
            return
        }
        isCheckingBackend = true
        defer { isCheckingBackend = false }
        do {
            backendCapabilities = try await BackendClient(baseURL: baseURL).capabilities()
            if lastErrorText == "Backend unavailable." || lastErrorText == "Invalid backend URL." {
                lastErrorText = nil
            }
        } catch {
            backendCapabilities = nil
            lastErrorText = "Backend unavailable."
        }
    }

    var pipelineCapability: PipelineCapability {
        backendCapabilities?.capability ?? .unavailable
    }

    func capability(for pipeline: ScanPipeline) -> PipelineCapability {
        _ = pipeline
        return pipelineCapability
    }

    func uploadLatestPackage(backendBaseURL: String) async {
        guard let lastPackageURL else { return }
        guard let baseURL = URL(string: backendBaseURL) else {
            statusText = "Bad backend URL"
            return
        }

        isUploading = true
        statusText = "Processing"
        lastErrorText = nil
        lastScaleText = nil
        resultURL = nil
        resultJobID = nil
        resultMetrics = nil
        downloadedAssets = [:]
        downloadingAssets = []
        do {
            let client = BackendClient(baseURL: baseURL)
            let result = try await client.reconstruct(packageURL: lastPackageURL)
            resultURL = result.outputURL
            resultJobID = result.jobID
            resultMetrics = result.metrics
            if let metrics = result.metrics, metrics.finalOutputType == "mesh", metrics.meshFaces > 0 {
                statusText = "Metric mesh ready"
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

    func assetURL(for kind: BackendAssetKind) -> URL? {
        if kind == .result {
            return resultURL
        }
        return downloadedAssets[kind]
    }

    func assetIsAvailable(_ kind: BackendAssetKind) -> Bool {
        kind == .result ? resultURL != nil : resultMetrics?.supports(kind) == true
    }

    func downloadAsset(_ kind: BackendAssetKind, backendBaseURL: String) async {
        guard kind != .result,
              downloadedAssets[kind] == nil,
              let jobID = resultJobID,
              let baseURL = URL(string: backendBaseURL)
        else { return }
        downloadingAssets.insert(kind)
        defer { downloadingAssets.remove(kind) }
        do {
            downloadedAssets[kind] = try await BackendClient(baseURL: baseURL).downloadAsset(jobID: jobID, kind: kind)
        } catch {
            lastErrorText = error.localizedDescription
        }
    }
}

extension ScanSessionManager: ARSessionDelegate {
    nonisolated func session(_ session: ARSession, didUpdate frame: ARFrame) {
        Task { @MainActor in
            latestFrame = frame
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
            guard !isCapturingPhoto, !isUploading else { return }
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

extension BackendMetrics {
    var scaleSummary: String? {
        guard let extent = objectExtentM ?? sceneExtentM, extent.count == 3 else { return nil }
        let size = extent.map { String(format: "%.2f", $0) }.joined(separator: " x ")
        var details = ["Scale \(size) m"]
        if let alignmentRmseM {
            details.append("ICP \(String(format: "%.1f", alignmentRmseM * 1_000)) mm")
        }
        if let printMeshWatertight {
            details.append(printMeshWatertight ? "STL watertight" : "STL repair needed")
        }
        return details.joined(separator: " · ")
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
