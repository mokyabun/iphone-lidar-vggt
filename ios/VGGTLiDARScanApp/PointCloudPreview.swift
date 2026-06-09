import SceneKit
import SwiftUI

struct PointCloudPreview: View {
    let resultURL: URL

    @State private var model: PLYModel?
    @State private var loadError: String?

    var body: some View {
        Group {
            if let loadError {
                ContentUnavailableView("Preview failed", systemImage: "exclamationmark.triangle", description: Text(loadError))
            } else if let model {
                ScenePLYView(model: model)
                    .ignoresSafeArea()
                    .overlay(alignment: .topLeading) {
                        Text(model.faces.isEmpty ? "\(model.vertices.count) points" : "\(model.faces.count) faces")
                            .font(.system(.callout, design: .monospaced))
                            .padding(10)
                            .background(.black.opacity(0.6))
                            .foregroundStyle(.white)
                            .clipShape(Capsule())
                            .padding()
                    }
            } else {
                ProgressView()
                    .task {
                        await loadModel()
                    }
            }
        }
    }

    private func loadModel() async {
        do {
            model = try PLYParser.parseAscii(url: resultURL, maxPoints: 120_000)
        } catch {
            loadError = error.localizedDescription
        }
    }
}

struct ScenePLYView: UIViewRepresentable {
    let model: PLYModel

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
        let geometry = PLYGeometry.makeGeometry(model: model)
        let node = SCNNode(geometry: geometry)
        scene.rootNode.addChildNode(node)

        let cameraNode = SCNNode()
        cameraNode.camera = SCNCamera()
        cameraNode.position = SCNVector3(0, 0, 2.5)
        scene.rootNode.addChildNode(cameraNode)
        return scene
    }
}

struct PLYModel {
    let vertices: [PLYVertex]
    let faces: [PLYFace]
}

struct PLYVertex {
    let x: Float
    let y: Float
    let z: Float
    let red: Float
    let green: Float
    let blue: Float
}

struct PLYFace {
    let a: Int32
    let b: Int32
    let c: Int32
}

enum PLYParser {
    static func parseAscii(url: URL, maxPoints: Int) throws -> PLYModel {
        let text = try String(contentsOf: url, encoding: .utf8)
        let lines = text.split(separator: "\n", omittingEmptySubsequences: false)
        guard let endHeaderIndex = lines.firstIndex(where: { String($0).trimmingCharacters(in: .whitespacesAndNewlines) == "end_header" }) else {
            throw CocoaError(.fileReadCorruptFile)
        }

        let header = lines[..<endHeaderIndex].map(String.init)
        guard header.contains(where: { $0 == "format ascii 1.0" }) else {
            throw CocoaError(.fileReadUnsupportedScheme)
        }

        let vertexCount = elementCount("vertex", in: header)
        let faceCount = elementCount("face", in: header)
        let properties = vertexProperties(from: header)
        guard let xIndex = properties.firstIndex(of: "x"),
              let yIndex = properties.firstIndex(of: "y"),
              let zIndex = properties.firstIndex(of: "z") else {
            throw CocoaError(.fileReadCorruptFile)
        }
        let redIndex = properties.firstIndex(of: "red")
        let greenIndex = properties.firstIndex(of: "green")
        let blueIndex = properties.firstIndex(of: "blue")

        let vertexLines = Array(lines[(endHeaderIndex + 1)...].prefix(vertexCount))
        let faceLines = Array(lines.dropFirst(endHeaderIndex + 1 + vertexCount).prefix(faceCount))

        if faceCount > 0 {
            let vertices = parseVertices(
                vertexLines,
                step: 1,
                limit: vertexCount,
                properties: properties,
                xIndex: xIndex,
                yIndex: yIndex,
                zIndex: zIndex,
                redIndex: redIndex,
                greenIndex: greenIndex,
                blueIndex: blueIndex
            )
            return PLYModel(vertices: vertices, faces: parseFaces(faceLines))
        }

        let step = max(1, vertexCount / maxPoints)
        let vertices = parseVertices(
            vertexLines,
            step: step,
            limit: maxPoints,
            properties: properties,
            xIndex: xIndex,
            yIndex: yIndex,
            zIndex: zIndex,
            redIndex: redIndex,
            greenIndex: greenIndex,
            blueIndex: blueIndex
        )
        return PLYModel(vertices: vertices, faces: [])
    }

    private static func parseVertices(
        _ lines: [Substring],
        step: Int,
        limit: Int,
        properties: [String],
        xIndex: Int,
        yIndex: Int,
        zIndex: Int,
        redIndex: Int?,
        greenIndex: Int?,
        blueIndex: Int?
    ) -> [PLYVertex] {
        var vertices: [PLYVertex] = []
        vertices.reserveCapacity(min(lines.count, limit))
        for (index, line) in lines.enumerated() {
            guard index % step == 0, vertices.count < limit else { continue }
            let values = line.split(separator: " ")
            guard values.count >= properties.count,
                  let x = Float(values[xIndex]),
                  let y = Float(values[yIndex]),
                  let z = Float(values[zIndex]) else {
                continue
            }
            let red = redIndex.flatMap { Float(values[$0]) } ?? 220
            let green = greenIndex.flatMap { Float(values[$0]) } ?? 220
            let blue = blueIndex.flatMap { Float(values[$0]) } ?? 220
            vertices.append(PLYVertex(x: x, y: y, z: z, red: red / 255, green: green / 255, blue: blue / 255))
        }
        return vertices
    }

    private static func parseFaces(_ lines: [Substring]) -> [PLYFace] {
        var faces: [PLYFace] = []
        for line in lines {
            let values = line.split(separator: " ").compactMap { Int32($0) }
            guard let count = values.first, count >= 3, values.count >= Int(count) + 1 else { continue }
            let indices = values.dropFirst().prefix(Int(count))
            guard let first = indices.first else { continue }
            for index in 1..<(indices.count - 1) {
                let b = indices[indices.index(indices.startIndex, offsetBy: index)]
                let c = indices[indices.index(indices.startIndex, offsetBy: index + 1)]
                faces.append(PLYFace(a: first, b: b, c: c))
            }
        }
        return faces
    }

    private static func elementCount(_ name: String, in header: [String]) -> Int {
        header
            .first(where: { $0.hasPrefix("element \(name) ") })?
            .split(separator: " ")
            .last
            .flatMap { Int($0) } ?? 0
    }

    private static func vertexProperties(from header: [String]) -> [String] {
        var properties: [String] = []
        var readingVertex = false
        for line in header {
            if line.hasPrefix("element vertex ") {
                readingVertex = true
                continue
            }
            if line.hasPrefix("element ") {
                readingVertex = false
            }
            if readingVertex, line.hasPrefix("property ") {
                let parts = line.split(separator: " ")
                if let name = parts.last {
                    properties.append(String(name))
                }
            }
        }
        return properties
    }
}

enum PLYGeometry {
    static func makeGeometry(model: PLYModel) -> SCNGeometry {
        var vertexData = Data(capacity: model.vertices.count * 3 * MemoryLayout<Float>.size)
        var colorData = Data(capacity: model.vertices.count * 4 * MemoryLayout<Float>.size)
        for vertex in model.vertices {
            vertexData.appendFloat(vertex.x)
            vertexData.appendFloat(vertex.y)
            vertexData.appendFloat(vertex.z)
            colorData.appendFloat(vertex.red)
            colorData.appendFloat(vertex.green)
            colorData.appendFloat(vertex.blue)
            colorData.appendFloat(1)
        }

        let vertices = SCNGeometrySource(
            data: vertexData,
            semantic: .vertex,
            vectorCount: model.vertices.count,
            usesFloatComponents: true,
            componentsPerVector: 3,
            bytesPerComponent: MemoryLayout<Float>.size,
            dataOffset: 0,
            dataStride: MemoryLayout<Float>.size * 3
        )
        let colors = SCNGeometrySource(
            data: colorData,
            semantic: .color,
            vectorCount: model.vertices.count,
            usesFloatComponents: true,
            componentsPerVector: 4,
            bytesPerComponent: MemoryLayout<Float>.size,
            dataOffset: 0,
            dataStride: MemoryLayout<Float>.size * 4
        )

        let element: SCNGeometryElement
        if model.faces.isEmpty {
            var indices = [Int32]()
            indices.reserveCapacity(model.vertices.count)
            for index in model.vertices.indices {
                indices.append(Int32(index))
            }
            element = SCNGeometryElement(indices: indices, primitiveType: .point)
        } else {
            var indexData = Data(capacity: model.faces.count * 3 * MemoryLayout<Int32>.size)
            for face in model.faces {
                indexData.appendInt32(face.a)
                indexData.appendInt32(face.b)
                indexData.appendInt32(face.c)
            }
            element = SCNGeometryElement(
                data: indexData,
                primitiveType: .triangles,
                primitiveCount: model.faces.count,
                bytesPerIndex: MemoryLayout<Int32>.size
            )
        }

        let geometry = SCNGeometry(sources: [vertices, colors], elements: [element])
        let material = SCNMaterial()
        material.lightingModel = model.faces.isEmpty ? .constant : .physicallyBased
        material.diffuse.contents = UIColor.white
        material.isDoubleSided = true
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

    mutating func appendInt32(_ value: Int32) {
        var mutableValue = value
        Swift.withUnsafeBytes(of: &mutableValue) { buffer in
            append(contentsOf: buffer)
        }
    }
}
