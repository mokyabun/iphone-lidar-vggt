import Foundation
import simd

struct ScanMetadata: Codable {
    let appVersion: String
    let packageVersion: Int
    let deviceModel: String
    let osVersion: String
    let lidarSupported: Bool
    let scanMode: String

    enum CodingKeys: String, CodingKey {
        case appVersion = "app_version"
        case packageVersion = "package_version"
        case deviceModel = "device_model"
        case osVersion = "os_version"
        case lidarSupported = "lidar_supported"
        case scanMode = "scan_mode"
    }
}

struct ScanFrameRecord: Codable {
    let frameId: String
    let timestamp: TimeInterval
    let imagePath: String
    let depthPath: String
    let confidencePath: String?
    let imageWidth: Int
    let imageHeight: Int
    let depthWidth: Int
    let depthHeight: Int
    let intrinsicsDepth: [[Float]]
    let cameraToWorld: [[Float]]
    let orientation: String

    enum CodingKeys: String, CodingKey {
        case frameId = "frame_id"
        case timestamp
        case imagePath = "image_path"
        case depthPath = "depth_path"
        case confidencePath = "confidence_path"
        case imageWidth = "image_width"
        case imageHeight = "image_height"
        case depthWidth = "depth_width"
        case depthHeight = "depth_height"
        case intrinsicsDepth = "intrinsics_depth"
        case cameraToWorld = "camera_to_world"
        case orientation
    }
}

extension simd_float3x3 {
    func rowMajorScaled(imageWidth: Int, imageHeight: Int, depthWidth: Int, depthHeight: Int) -> [[Float]] {
        let scaleX = Float(depthWidth) / Float(imageWidth)
        let scaleY = Float(depthHeight) / Float(imageHeight)
        return [
            [columns.0.x * scaleX, columns.1.x * scaleX, columns.2.x * scaleX],
            [columns.0.y * scaleY, columns.1.y * scaleY, columns.2.y * scaleY],
            [columns.0.z, columns.1.z, columns.2.z]
        ]
    }
}

extension simd_float4x4 {
    var rowMajor: [[Float]] {
        [
            [columns.0.x, columns.1.x, columns.2.x, columns.3.x],
            [columns.0.y, columns.1.y, columns.2.y, columns.3.y],
            [columns.0.z, columns.1.z, columns.2.z, columns.3.z],
            [columns.0.w, columns.1.w, columns.2.w, columns.3.w]
        ]
    }
}

