import Foundation
import SwiftUI

enum ScanPipeline: String, CaseIterable, Identifiable {
    case reconviagen = "reconviagen_lidar_scale"

    var id: String { rawValue }

    var title: String { "ReconViaGen" }

    var systemImage: String { "wand.and.stars" }

    var tint: Color { .pink }

    var timeout: TimeInterval { 2_400 }
}

enum BackendAssetKind: String, CaseIterable, Identifiable {
    case result
    case preview
    case raw
    case rawPly = "raw-ply"
    case rawStl = "raw-stl"
    case print
    case lidar

    var id: String { rawValue }

    var title: String {
        switch self {
        case .result: return "PLY"
        case .preview: return "PBR GLB"
        case .raw: return "Raw GLB"
        case .rawPly: return "Raw PLY"
        case .rawStl: return "Raw STL"
        case .print: return "Print STL"
        case .lidar: return "LiDAR Ref"
        }
    }

    var filename: String {
        switch self {
        case .result: return "reconviagen_metric.ply"
        case .preview: return "reconviagen_metric.glb"
        case .raw: return "reconviagen_raw.glb"
        case .rawPly: return "reconviagen_raw.ply"
        case .rawStl: return "reconviagen_raw.stl"
        case .print: return "reconviagen_metric_print_mm.stl"
        case .lidar: return "lidar_reference.ply"
        }
    }

    var systemImage: String {
        switch self {
        case .result: return "cube.transparent"
        case .preview: return "paintpalette"
        case .raw: return "cube"
        case .rawPly: return "cube.transparent"
        case .rawStl: return "cube.transparent.fill"
        case .print: return "printer"
        case .lidar: return "ruler"
        }
    }
}

struct BackendReconstructionResult {
    let outputURL: URL
    let jobID: String?
    let metrics: BackendMetrics?
}

struct BackendReconstructionOptions {
    var enableSAM3ObjectMasking: Bool
    var enableLiDARScaleAlignment: Bool
    var sam3TextPrompt: String
}

struct BackendCapabilities: Decodable {
    let pipeline: String?
    let state: PipelineState
    let reason: String?
    let features: BackendFeatures?

    var capability: PipelineCapability {
        PipelineCapability(state: state, reason: reason)
    }
}

struct BackendFeatures: Decodable {
    let sam3ObjectMasking: PipelineCapability?
    let lidarScaleAlignment: PipelineCapability?
}

struct PipelineCapability: Decodable {
    let state: PipelineState
    let reason: String?

    static let unavailable = PipelineCapability(
        state: .unavailable,
        reason: "Backend unavailable."
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
    let inputViews: Int?
    let lidarPoints: Int
    let scenePoints: Int?
    let meshVertices: Int
    let meshFaces: Int
    let finalOutputType: String
    let finalOutputSource: String?
    let objectBoundsMinM: [Double]?
    let objectBoundsMaxM: [Double]?
    let objectExtentM: [Double]?
    let sceneBoundsMinM: [Double]?
    let sceneBoundsMaxM: [Double]?
    let sceneExtentM: [Double]?
    let warnings: [String]
    let previewGlbOutput: String?
    let printStlOutput: String?
    let lidarReferenceOutput: String?
    let printMeshWatertight: Bool?
    let alignmentRmseM: Double?
    let alignmentScale: Double?
    let sam3ObjectMaskingEnabled: Bool?
    let sam3MaskingUsed: Bool?
    let maskSource: String?
    let lidarScaleAlignmentEnabled: Bool?
    let rawMeshOutput: String?
    let rawPlyOutput: String?
    let rawStlOutput: String?
    let alignedMeshOutput: String?
    let rawObjectExtentM: [Double]?
    let alignedObjectExtentM: [Double]?
    let lidarObjectExtentM: [Double]?

    func supports(_ kind: BackendAssetKind) -> Bool {
        switch kind {
        case .result: return true
        case .preview: return previewGlbOutput != nil
        case .raw: return rawMeshOutput != nil
        case .rawPly: return rawPlyOutput != nil
        case .rawStl: return rawStlOutput != nil
        case .print: return printStlOutput != nil
        case .lidar: return lidarReferenceOutput != nil
        }
    }
}
