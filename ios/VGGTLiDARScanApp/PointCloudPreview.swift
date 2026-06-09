import SceneKit
import SwiftUI

struct PointCloudPreview: View {
    let resultURL: URL

    @State private var points: [PLYPoint] = []
    @State private var loadError: String?

    var body: some View {
        Group {
            if let loadError {
                ContentUnavailableView("Preview failed", systemImage: "exclamationmark.triangle", description: Text(loadError))
            } else if points.isEmpty {
                ProgressView()
                    .task {
                        await loadPoints()
                    }
            } else {
                ScenePointCloudView(points: points)
                    .ignoresSafeArea()
                    .overlay(alignment: .topLeading) {
                        Text("\(points.count) points")
                            .font(.system(.callout, design: .monospaced))
                            .padding(10)
                            .background(.black.opacity(0.6))
                            .foregroundStyle(.white)
                            .clipShape(Capsule())
                            .padding()
                    }
            }
        }
    }

    private func loadPoints() async {
        do {
            points = try PLYParser.parseAsciiPointCloud(url: resultURL, maxPoints: 120_000)
        } catch {
            loadError = error.localizedDescription
        }
    }
}

struct ScenePointCloudView: UIViewRepresentable {
    let points: [PLYPoint]

    func makeUIView(context: Context) -> SCNView {
        let view = SCNView()
        view.allowsCameraControl = true
        view.autoenablesDefaultLighting = true
        view.backgroundColor = .black
        view.scene = makeScene()
        return view
    }

    func updateUIView(_ uiView: SCNView, context: Context) {
        uiView.scene = makeScene()
    }

    private func makeScene() -> SCNScene {
        let scene = SCNScene()
        let geometry = PointCloudGeometry.makeGeometry(points: points)
        let node = SCNNode(geometry: geometry)
        scene.rootNode.addChildNode(node)

        let cameraNode = SCNNode()
        cameraNode.camera = SCNCamera()
        cameraNode.position = SCNVector3(0, 0, 2.5)
        scene.rootNode.addChildNode(cameraNode)
        return scene
    }
}

struct PLYPoint {
    let x: Float
    let y: Float
    let z: Float
    let red: Float
    let green: Float
    let blue: Float
}

enum PLYParser {
    static func parseAsciiPointCloud(url: URL, maxPoints: Int) throws -> [PLYPoint] {
        let text = try String(contentsOf: url, encoding: .utf8)
        let lines = text.split(separator: "\n", omittingEmptySubsequences: false)
        guard let endHeaderIndex = lines.firstIndex(where: { String($0).trimmingCharacters(in: .whitespacesAndNewlines) == "end_header" }) else {
            throw CocoaError(.fileReadCorruptFile)
        }

        let vertexCount = lines[..<endHeaderIndex]
            .first(where: { $0.hasPrefix("element vertex ") })?
            .split(separator: " ")
            .last
            .flatMap { Int($0) } ?? 0

        let step = max(1, vertexCount / maxPoints)
        var parsed: [PLYPoint] = []
        parsed.reserveCapacity(min(vertexCount, maxPoints))

        for (index, line) in lines[(endHeaderIndex + 1)...].enumerated() {
            guard index % step == 0, parsed.count < maxPoints else { continue }
            let values = line.split(separator: " ")
            guard values.count >= 6,
                  let x = Float(values[0]),
                  let y = Float(values[1]),
                  let z = Float(values[2]),
                  let red = Float(values[3]),
                  let green = Float(values[4]),
                  let blue = Float(values[5]) else {
                continue
            }
            parsed.append(PLYPoint(x: x, y: y, z: z, red: red / 255, green: green / 255, blue: blue / 255))
        }
        return parsed
    }
}

enum PointCloudGeometry {
    static func makeGeometry(points: [PLYPoint]) -> SCNGeometry {
        var vertexData = Data(capacity: points.count * 3 * MemoryLayout<Float>.size)
        var colorData = Data(capacity: points.count * 4 * MemoryLayout<Float>.size)
        var indices = [Int32]()
        indices.reserveCapacity(points.count)

        for (index, point) in points.enumerated() {
            vertexData.appendFloat(point.x)
            vertexData.appendFloat(point.y)
            vertexData.appendFloat(point.z)
            colorData.appendFloat(point.red)
            colorData.appendFloat(point.green)
            colorData.appendFloat(point.blue)
            colorData.appendFloat(1)
            indices.append(Int32(index))
        }

        let vertices = SCNGeometrySource(
            data: vertexData,
            semantic: .vertex,
            vectorCount: points.count,
            usesFloatComponents: true,
            componentsPerVector: 3,
            bytesPerComponent: MemoryLayout<Float>.size,
            dataOffset: 0,
            dataStride: MemoryLayout<Float>.size * 3
        )
        let colors = SCNGeometrySource(
            data: colorData,
            semantic: .color,
            vectorCount: points.count,
            usesFloatComponents: true,
            componentsPerVector: 4,
            bytesPerComponent: MemoryLayout<Float>.size,
            dataOffset: 0,
            dataStride: MemoryLayout<Float>.size * 4
        )
        let element = SCNGeometryElement(indices: indices, primitiveType: .point)
        let geometry = SCNGeometry(sources: [vertices, colors], elements: [element])
        let material = SCNMaterial()
        material.lightingModel = .constant
        material.diffuse.contents = UIColor.white
        geometry.materials = [material]
        return geometry
    }
}

private extension Data {
    mutating func appendFloat(_ value: Float) {
        var mutableValue = value
        Swift.withUnsafeBytes(of: &mutableValue) { buffer in
            append(contentsOf: buffer)
        }
    }
}
