import ARKit
import CoreImage
import Foundation
import UIKit

final class ScanPackageExporter {
    private let encoder: JSONEncoder = {
        let encoder = JSONEncoder()
        encoder.outputFormatting = [.sortedKeys]
        return encoder
    }()

    private var rootURL: URL?
    private var framesFileHandle: FileHandle?
    private let ciContext = CIContext()

    func beginScan(lidarSupported: Bool) throws {
        closeFramesFile()
        let base = FileManager.default.temporaryDirectory
            .appendingPathComponent("ScanPackage-\(UUID().uuidString)", isDirectory: true)
        try FileManager.default.createDirectory(at: base.appendingPathComponent("images", isDirectory: true), withIntermediateDirectories: true)
        try FileManager.default.createDirectory(at: base.appendingPathComponent("depth", isDirectory: true), withIntermediateDirectories: true)
        try FileManager.default.createDirectory(at: base.appendingPathComponent("confidence", isDirectory: true), withIntermediateDirectories: true)

        let metadata = ScanMetadata(
            appVersion: "0.1.0",
            packageVersion: 1,
            deviceModel: UIDevice.current.model,
            osVersion: UIDevice.current.systemVersion,
            lidarSupported: lidarSupported,
            scanMode: "object"
        )
        let metadataData = try encoder.encode(metadata)
        try metadataData.write(to: base.appendingPathComponent("metadata.json"))

        let framesURL = base.appendingPathComponent("frames.jsonl")
        FileManager.default.createFile(atPath: framesURL.path, contents: nil)
        framesFileHandle = try FileHandle(forWritingTo: framesURL)
        rootURL = base
    }

    func append(frame: ARFrame, index: Int) throws {
        guard let rootURL, let framesFileHandle else { return }
        guard let depthData = frame.smoothedSceneDepth ?? frame.sceneDepth else { return }

        let frameId = String(format: "frame_%06d", index)
        let imagePath = "images/\(frameId).jpg"
        let depthPath = "depth/\(frameId).float32"
        let confidencePath = depthData.confidenceMap == nil ? nil : "confidence/\(frameId).uint8"

        let imageSize = try writeJPEG(from: frame.capturedImage, to: rootURL.appendingPathComponent(imagePath))
        let depthSize = try writeFloat32Depth(depthData.depthMap, to: rootURL.appendingPathComponent(depthPath))
        if let confidenceMap = depthData.confidenceMap, let confidencePath {
            try writeUInt8(confidenceMap, to: rootURL.appendingPathComponent(confidencePath))
        }

        let record = ScanFrameRecord(
            frameId: frameId,
            timestamp: frame.timestamp,
            imagePath: imagePath,
            depthPath: depthPath,
            confidencePath: confidencePath,
            imageWidth: imageSize.width,
            imageHeight: imageSize.height,
            depthWidth: depthSize.width,
            depthHeight: depthSize.height,
            intrinsicsDepth: frame.camera.intrinsics.rowMajorScaled(
                imageWidth: imageSize.width,
                imageHeight: imageSize.height,
                depthWidth: depthSize.width,
                depthHeight: depthSize.height
            ),
            cameraToWorld: frame.camera.transform.rowMajor,
            orientation: UIDevice.current.orientation.captureLabel
        )

        var line = try encoder.encode(record)
        line.append(0x0a)
        framesFileHandle.write(line)
    }

    func finishScan() throws -> URL {
        closeFramesFile()
        guard let rootURL else {
            throw CocoaError(.fileNoSuchFile)
        }
        let zipURL = rootURL.deletingLastPathComponent().appendingPathComponent("ScanPackage-\(UUID().uuidString).zip")
        try ZipWriter.zipDirectory(rootURL, to: zipURL)
        return zipURL
    }

    private func closeFramesFile() {
        try? framesFileHandle?.close()
        framesFileHandle = nil
    }

    private func writeJPEG(from pixelBuffer: CVPixelBuffer, to url: URL) throws -> (width: Int, height: Int) {
        let width = CVPixelBufferGetWidth(pixelBuffer)
        let height = CVPixelBufferGetHeight(pixelBuffer)
        let ciImage = CIImage(cvPixelBuffer: pixelBuffer)
        guard let cgImage = ciContext.createCGImage(ciImage, from: CGRect(x: 0, y: 0, width: width, height: height)) else {
            throw CocoaError(.coderInvalidValue)
        }
        let image = UIImage(cgImage: cgImage)
        guard let data = image.jpegData(compressionQuality: 0.92) else {
            throw CocoaError(.coderInvalidValue)
        }
        try data.write(to: url, options: .atomic)
        return (width, height)
    }

    private func writeFloat32Depth(_ pixelBuffer: CVPixelBuffer, to url: URL) throws -> (width: Int, height: Int) {
        CVPixelBufferLockBaseAddress(pixelBuffer, .readOnly)
        defer { CVPixelBufferUnlockBaseAddress(pixelBuffer, .readOnly) }

        let width = CVPixelBufferGetWidth(pixelBuffer)
        let height = CVPixelBufferGetHeight(pixelBuffer)
        let bytesPerRow = CVPixelBufferGetBytesPerRow(pixelBuffer)
        guard let baseAddress = CVPixelBufferGetBaseAddress(pixelBuffer) else {
            throw CocoaError(.fileReadUnknown)
        }

        var data = Data(capacity: width * height * MemoryLayout<Float32>.size)
        for row in 0..<height {
            let rowPointer = baseAddress.advanced(by: row * bytesPerRow)
            let rowBuffer = UnsafeRawBufferPointer(start: rowPointer, count: width * MemoryLayout<Float32>.size)
            data.append(contentsOf: rowBuffer)
        }
        try data.write(to: url, options: .atomic)
        return (width, height)
    }

    private func writeUInt8(_ pixelBuffer: CVPixelBuffer, to url: URL) throws {
        CVPixelBufferLockBaseAddress(pixelBuffer, .readOnly)
        defer { CVPixelBufferUnlockBaseAddress(pixelBuffer, .readOnly) }

        let width = CVPixelBufferGetWidth(pixelBuffer)
        let height = CVPixelBufferGetHeight(pixelBuffer)
        let bytesPerRow = CVPixelBufferGetBytesPerRow(pixelBuffer)
        guard let baseAddress = CVPixelBufferGetBaseAddress(pixelBuffer) else {
            throw CocoaError(.fileReadUnknown)
        }

        var data = Data(capacity: width * height)
        for row in 0..<height {
            let rowPointer = baseAddress.advanced(by: row * bytesPerRow).assumingMemoryBound(to: UInt8.self)
            data.append(contentsOf: UnsafeBufferPointer(start: rowPointer, count: width))
        }
        try data.write(to: url, options: .atomic)
    }
}

private extension UIDeviceOrientation {
    var captureLabel: String {
        switch self {
        case .portrait: return "portrait"
        case .portraitUpsideDown: return "portraitUpsideDown"
        case .landscapeLeft: return "landscapeLeft"
        case .landscapeRight: return "landscapeRight"
        default: return "unknown"
        }
    }
}
