import Foundation
import SwiftUI

enum ScanPipeline: String, CaseIterable, Identifiable {
    case metric
    case vggt
    case aiMesh = "ai_mesh"

    var id: String { rawValue }

    var title: String {
        switch self {
        case .metric: return "LiDAR"
        case .vggt: return "VGGT"
        case .aiMesh: return "AI Mesh"
        }
    }

    var systemImage: String {
        switch self {
        case .metric: return "ruler"
        case .vggt: return "circle.grid.3x3.fill"
        case .aiMesh: return "wand.and.stars"
        }
    }

    var tint: Color {
        switch self {
        case .metric: return .green
        case .vggt: return .purple
        case .aiMesh: return .pink
        }
    }

    var timeout: TimeInterval {
        self == .aiMesh ? 2_400 : 900
    }
}

struct ReconstructionOptions {
    let pipeline: ScanPipeline
    let preserveColor: Bool
    let extractObject: Bool
    let reconstructMesh: Bool

    var effectiveObject: Bool {
        pipeline == .aiMesh || extractObject
    }

    var effectiveMesh: Bool {
        switch pipeline {
        case .metric: return reconstructMesh
        case .vggt: return false
        case .aiMesh: return true
        }
    }
}

enum BackendAssetKind: String, CaseIterable, Identifiable {
    case result
    case preview
    case print

    var id: String { rawValue }

    var title: String {
        switch self {
        case .result: return "PLY"
        case .preview: return "PBR GLB"
        case .print: return "Print STL"
        }
    }

    var filename: String {
        switch self {
        case .result: return "scan_final.ply"
        case .preview: return "scan_object_preview.glb"
        case .print: return "scan_object_print.stl"
        }
    }

    var systemImage: String {
        switch self {
        case .result: return "cube.transparent"
        case .preview: return "paintpalette"
        case .print: return "printer"
        }
    }
}

struct BackendReconstructionResult {
    let outputURL: URL
    let jobID: String?
    let metrics: BackendMetrics?
}

struct BackendCapabilities: Decodable {
    let pipelines: [String: PipelineCapability]

    func capability(for pipeline: ScanPipeline) -> PipelineCapability {
        pipelines[pipeline.rawValue] ?? .unavailable
    }
}

struct PipelineCapability: Decodable {
    let state: PipelineState
    let reason: String?
    let options: [String]
    let requiredOptions: [String]?

    static let unavailable = PipelineCapability(
        state: .unavailable,
        reason: "Backend unavailable.",
        options: [],
        requiredOptions: nil
    )

    var isAvailable: Bool {
        state == .available
    }
}

enum PipelineState: String, Decodable {
    case available
    case loading
    case unavailable
}

struct BackendMetrics: Decodable {
    let frameCount: Int
    let selectedKeyframes: Int
    let lidarPoints: Int
    let lidarRawPoints: Int?
    let lidarRemovedPoints: Int?
    let vggtPoints: Int
    let meshVertices: Int
    let meshFaces: Int
    let meshMethod: String?
    let finalOutputType: String
    let finalOutputSource: String?
    let aiMeshRequested: Bool?
    let aiMeshUsed: Bool?
    let objectMaskBackend: String?
    let cameraPathM: Double?
    let cameraExtentM: [Double]?
    let lidarBoundsMinM: [Double]?
    let lidarBoundsMaxM: [Double]?
    let lidarExtentM: [Double]?
    let objectBoundsMinM: [Double]?
    let objectBoundsMaxM: [Double]?
    let objectExtentM: [Double]?
    let warnings: [String]
    let meshOutput: String?
    let metricMeshOutput: String?
    let aiMeshOutput: String?
    let previewGlbOutput: String?
    let printStlOutput: String?
    let printMeshWatertight: Bool?
    let alignmentRmseM: Double?
    let alignmentScale: Double?

    func supports(_ kind: BackendAssetKind) -> Bool {
        switch kind {
        case .result: return true
        case .preview: return previewGlbOutput != nil
        case .print: return printStlOutput != nil
        }
    }
}
