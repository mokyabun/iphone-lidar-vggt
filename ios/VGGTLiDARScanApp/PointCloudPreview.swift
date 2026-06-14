import SceneKit
import simd
import SwiftUI

struct PointCloudPreview: View {
    let resultURL: URL

    @State private var model: PLYModel?
    @State private var loadError: String?
    @State private var displayMode = PLYDisplayMode.color

    var body: some View {
        Group {
            if let loadError {
                ContentUnavailableView("Preview failed", systemImage: "exclamationmark.triangle", description: Text(loadError))
            } else if let model {
                ScenePLYView(model: model, displayMode: displayMode)
                    .ignoresSafeArea()
                    .overlay(alignment: .topLeading) {
                        VStack(alignment: .leading, spacing: 4) {
                            Label(
                                model.faces.isEmpty ? "\(model.vertices.count) points" : "\(model.faces.count) faces",
                                systemImage: model.faces.isEmpty ? "circle.grid.3x3.fill" : "cube.fill"
                            )
                            Text(model.sizeSummary)
                                .foregroundStyle(.white.opacity(0.78))
                        }
                        .font(.system(.callout, design: .monospaced))
                        .padding(10)
                        .background(.black.opacity(0.62))
                        .foregroundStyle(.white)
                        .clipShape(RoundedRectangle(cornerRadius: 8))
                        .padding()
                    }
                    .overlay(alignment: .bottom) {
                        if !model.faces.isEmpty {
                            Picker("Display", selection: $displayMode) {
                                ForEach(PLYDisplayMode.allCases) { mode in
                                    Label(mode.title, systemImage: mode.systemImage)
                                        .tag(mode)
                                }
                            }
                            .pickerStyle(.segmented)
                            .padding(8)
                            .background(.ultraThinMaterial)
                            .clipShape(RoundedRectangle(cornerRadius: 8))
                            .padding()
                        }
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
    let displayMode: PLYDisplayMode

    func makeUIView(context: Context) -> SCNView {
        let view = SCNView()
        view.allowsCameraControl = true
        view.backgroundColor = UIColor(white: 0.07, alpha: 1)
        view.antialiasingMode = .multisampling4X
        configure(view)
        return view
    }

    func updateUIView(_ uiView: SCNView, context: Context) {
        configure(uiView)
    }

    private func configure(_ view: SCNView) {
        let prepared = makeScene()
        view.scene = prepared.scene
        view.pointOfView = prepared.camera
        view.defaultCameraController.target = prepared.target
        view.defaultCameraController.inertiaEnabled = true
    }

    private func makeScene() -> (scene: SCNScene, camera: SCNNode, target: SCNVector3) {
        let scene = SCNScene()
        let bounds = model.bounds
        let center = bounds.center
        let floorY = bounds.minimum.y
        let modelOffset = SIMD3<Float>(-center.x, -floorY, -center.z)

        let geometry = PLYGeometry.makeGeometry(model: model, offset: modelOffset, displayMode: displayMode)
        let modelNode = SCNNode(geometry: geometry)
        scene.rootNode.addChildNode(modelNode)

        let horizontalExtent = max(bounds.extent.x, bounds.extent.z)
        let gridSize = max(horizontalExtent * 3.0, model.gridSpacing * 12.0)
        scene.rootNode.addChildNode(makeFloor(size: gridSize))
        scene.rootNode.addChildNode(makeGrid(size: gridSize, spacing: model.gridSpacing))

        let target = SCNVector3(0, max(bounds.extent.y * 0.45, 0.01), 0)
        let largestExtent = max(bounds.extent.x, bounds.extent.y, bounds.extent.z, 0.04)
        let cameraDistance = largestExtent * 2.35
        let cameraNode = SCNNode()
        let camera = SCNCamera()
        camera.fieldOfView = 48
        camera.zNear = Double(max(largestExtent * 0.015, 0.0005))
        camera.zFar = Double(max(largestExtent * 30, 5))
        camera.wantsHDR = true
        camera.wantsExposureAdaptation = true
        cameraNode.camera = camera
        cameraNode.position = SCNVector3(cameraDistance * 0.85, cameraDistance * 0.62, cameraDistance)
        let lookAt = SCNLookAtConstraint(target: makeTargetNode(target, in: scene))
        lookAt.isGimbalLockEnabled = true
        cameraNode.constraints = [lookAt]
        scene.rootNode.addChildNode(cameraNode)

        addLighting(to: scene, target: target, distance: cameraDistance)
        return (scene, cameraNode, target)
    }

    private func makeTargetNode(_ position: SCNVector3, in scene: SCNScene) -> SCNNode {
        let node = SCNNode()
        node.position = position
        scene.rootNode.addChildNode(node)
        return node
    }

    private func makeFloor(size: Float) -> SCNNode {
        let plane = SCNPlane(width: CGFloat(size), height: CGFloat(size))
        let material = SCNMaterial()
        material.diffuse.contents = UIColor(white: 0.16, alpha: 1)
        material.roughness.contents = 0.92
        material.metalness.contents = 0
        plane.materials = [material]
        let node = SCNNode(geometry: plane)
        node.eulerAngles.x = -.pi / 2
        node.position.y = -0.001
        node.castsShadow = false
        return node
    }

    private func makeGrid(size: Float, spacing: Float) -> SCNNode {
        let half = size / 2
        let lineCount = min(80, max(4, Int(ceil(size / spacing))))
        let actualSpacing = size / Float(lineCount)
        var positions: [SCNVector3] = []
        positions.reserveCapacity((lineCount + 1) * 4)
        for index in 0...lineCount {
            let value = -half + Float(index) * actualSpacing
            positions.append(SCNVector3(value, 0, -half))
            positions.append(SCNVector3(value, 0, half))
            positions.append(SCNVector3(-half, 0, value))
            positions.append(SCNVector3(half, 0, value))
        }
        let source = SCNGeometrySource(vertices: positions)
        let element = SCNGeometryElement(
            data: nil,
            primitiveType: .line,
            primitiveCount: positions.count / 2,
            bytesPerIndex: 0
        )
        let geometry = SCNGeometry(sources: [source], elements: [element])
        let material = SCNMaterial()
        material.lightingModel = .constant
        material.diffuse.contents = UIColor(white: 0.62, alpha: 0.24)
        material.writesToDepthBuffer = false
        geometry.materials = [material]
        let node = SCNNode(geometry: geometry)
        node.position.y = 0.0002
        node.renderingOrder = 2
        return node
    }

    private func addLighting(to scene: SCNScene, target: SCNVector3, distance: Float) {
        let ambientNode = SCNNode()
        ambientNode.light = SCNLight()
        ambientNode.light?.type = .ambient
        ambientNode.light?.intensity = 520
        ambientNode.light?.color = UIColor(white: 0.72, alpha: 1)
        scene.rootNode.addChildNode(ambientNode)

        let keyNode = SCNNode()
        keyNode.light = SCNLight()
        keyNode.light?.type = .directional
        keyNode.light?.intensity = 1_350
        keyNode.light?.castsShadow = true
        keyNode.light?.shadowRadius = 5
        keyNode.position = SCNVector3(distance, distance * 1.7, distance)
        let targetNode = SCNNode()
        targetNode.position = target
        scene.rootNode.addChildNode(targetNode)
        keyNode.constraints = [SCNLookAtConstraint(target: targetNode)]
        scene.rootNode.addChildNode(keyNode)

        let fillNode = SCNNode()
        fillNode.light = SCNLight()
        fillNode.light?.type = .omni
        fillNode.light?.intensity = 560
        fillNode.position = SCNVector3(-distance, distance * 0.8, -distance * 0.4)
        scene.rootNode.addChildNode(fillNode)
    }
}

enum PLYDisplayMode: String, CaseIterable, Identifiable {
    case color
    case shape
    case wire

    var id: String { rawValue }

    var title: String {
        switch self {
        case .color: return "Color"
        case .shape: return "Shape"
        case .wire: return "Wire"
        }
    }

    var systemImage: String {
        switch self {
        case .color: return "paintpalette"
        case .shape: return "cube"
        case .wire: return "triangle"
        }
    }
}

struct PLYBounds {
    let minimum: SIMD3<Float>
    let maximum: SIMD3<Float>

    var center: SIMD3<Float> {
        (minimum + maximum) * 0.5
    }

    var extent: SIMD3<Float> {
        maximum - minimum
    }
}

struct PLYModel {
    let vertices: [PLYVertex]
    let faces: [PLYFace]

    var bounds: PLYBounds {
        guard let first = vertices.first else {
            return PLYBounds(minimum: .zero, maximum: SIMD3<Float>(repeating: 0.1))
        }
        var minimum = first.position
        var maximum = first.position
        for vertex in vertices.dropFirst() {
            minimum = simd_min(minimum, vertex.position)
            maximum = simd_max(maximum, vertex.position)
        }
        return PLYBounds(minimum: minimum, maximum: maximum)
    }

    var gridSpacing: Float {
        let largest = max(bounds.extent.x, bounds.extent.y, bounds.extent.z)
        if largest <= 0.3 { return 0.01 }
        if largest <= 1.5 { return 0.05 }
        return 0.1
    }

    var sizeSummary: String {
        let extent = bounds.extent
        let values = [extent.x, extent.y, extent.z]
        let largest = values.max() ?? 0
        if largest < 1 {
            let size = values.map { String(format: "%.1f", $0 * 100) }.joined(separator: " x ")
            return "\(size) cm · \(Int(gridSpacing * 100)) cm grid"
        }
        let size = values.map { String(format: "%.2f", $0) }.joined(separator: " x ")
        return "\(size) m · \(Int(gridSpacing * 100)) cm grid"
    }
}

struct PLYVertex {
    let x: Float
    let y: Float
    let z: Float
    let red: Float
    let green: Float
    let blue: Float
    let nx: Float?
    let ny: Float?
    let nz: Float?

    var position: SIMD3<Float> {
        SIMD3(x, y, z)
    }

    var normal: SIMD3<Float>? {
        guard let nx, let ny, let nz else { return nil }
        return SIMD3(nx, ny, nz)
    }
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
        guard let endHeaderIndex = lines.firstIndex(where: {
            String($0).trimmingCharacters(in: .whitespacesAndNewlines) == "end_header"
        }) else {
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

        let vertexLines = Array(lines[(endHeaderIndex + 1)...].prefix(vertexCount))
        let faceLines = Array(lines.dropFirst(endHeaderIndex + 1 + vertexCount).prefix(faceCount))
        let redIndex = properties.firstIndex(of: "red")
        let greenIndex = properties.firstIndex(of: "green")
        let blueIndex = properties.firstIndex(of: "blue")
        let nxIndex = properties.firstIndex(of: "nx")
        let nyIndex = properties.firstIndex(of: "ny")
        let nzIndex = properties.firstIndex(of: "nz")

        if faceCount > 0 {
            let vertices = parseVertices(
                vertexLines,
                step: 1,
                limit: vertexCount,
                properties: properties,
                positionIndices: (xIndex, yIndex, zIndex),
                colorIndices: (redIndex, greenIndex, blueIndex),
                normalIndices: (nxIndex, nyIndex, nzIndex)
            )
            return PLYModel(vertices: vertices, faces: parseFaces(faceLines))
        }

        let step = max(1, vertexCount / maxPoints)
        let vertices = parseVertices(
            vertexLines,
            step: step,
            limit: maxPoints,
            properties: properties,
            positionIndices: (xIndex, yIndex, zIndex),
            colorIndices: (redIndex, greenIndex, blueIndex),
            normalIndices: (nxIndex, nyIndex, nzIndex)
        )
        return PLYModel(vertices: vertices, faces: [])
    }

    private static func parseVertices(
        _ lines: [Substring],
        step: Int,
        limit: Int,
        properties: [String],
        positionIndices: (Int, Int, Int),
        colorIndices: (Int?, Int?, Int?),
        normalIndices: (Int?, Int?, Int?)
    ) -> [PLYVertex] {
        var vertices: [PLYVertex] = []
        vertices.reserveCapacity(min(lines.count, limit))
        for (index, line) in lines.enumerated() {
            guard index % step == 0, vertices.count < limit else { continue }
            let values = line.split(whereSeparator: { $0 == " " || $0 == "\t" })
            guard values.count >= properties.count,
                  let x = Float(values[positionIndices.0]),
                  let y = Float(values[positionIndices.1]),
                  let z = Float(values[positionIndices.2]) else {
                continue
            }
            let red = colorIndices.0.flatMap { Float(values[$0]) } ?? 220
            let green = colorIndices.1.flatMap { Float(values[$0]) } ?? 220
            let blue = colorIndices.2.flatMap { Float(values[$0]) } ?? 220
            let nx = normalIndices.0.flatMap { Float(values[$0]) }
            let ny = normalIndices.1.flatMap { Float(values[$0]) }
            let nz = normalIndices.2.flatMap { Float(values[$0]) }
            vertices.append(
                PLYVertex(
                    x: x,
                    y: y,
                    z: z,
                    red: red / 255,
                    green: green / 255,
                    blue: blue / 255,
                    nx: nx,
                    ny: ny,
                    nz: nz
                )
            )
        }
        return vertices
    }

    private static func parseFaces(_ lines: [Substring]) -> [PLYFace] {
        var faces: [PLYFace] = []
        for line in lines {
            let values = line.split(whereSeparator: { $0 == " " || $0 == "\t" }).compactMap { Int32($0) }
            guard let count = values.first, count >= 3, values.count >= Int(count) + 1 else { continue }
            let indices = Array(values.dropFirst().prefix(Int(count)))
            for index in 1..<(indices.count - 1) {
                faces.append(PLYFace(a: indices[0], b: indices[index], c: indices[index + 1]))
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
            if readingVertex, line.hasPrefix("property "), let name = line.split(separator: " ").last {
                properties.append(String(name))
            }
        }
        return properties
    }
}

enum PLYGeometry {
    static func makeGeometry(model: PLYModel, offset: SIMD3<Float>, displayMode: PLYDisplayMode) -> SCNGeometry {
        var vertexData = Data(capacity: model.vertices.count * 3 * MemoryLayout<Float>.size)
        var colorData = Data(capacity: model.vertices.count * 4 * MemoryLayout<Float>.size)
        for vertex in model.vertices {
            let position = vertex.position + offset
            vertexData.appendFloat(position.x)
            vertexData.appendFloat(position.y)
            vertexData.appendFloat(position.z)
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
            element = SCNGeometryElement(indices: model.vertices.indices.map(Int32.init), primitiveType: .point)
            element.pointSize = 2.4
            element.minimumPointScreenSpaceRadius = 1.2
            element.maximumPointScreenSpaceRadius = 4.5
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

        var sources = displayMode == .color || model.faces.isEmpty ? [vertices, colors] : [vertices]
        if !model.faces.isEmpty {
            sources.append(makeNormalSource(model: model))
        }
        let geometry = SCNGeometry(sources: sources, elements: [element])
        let material = SCNMaterial()
        material.lightingModel = model.faces.isEmpty ? .constant : .physicallyBased
        material.diffuse.contents = UIColor.white
        material.roughness.contents = 0.72
        material.metalness.contents = 0
        material.isDoubleSided = true
        material.fillMode = displayMode == .wire ? .lines : .fill
        geometry.materials = [material]
        return geometry
    }

    private static func makeNormalSource(model: PLYModel) -> SCNGeometrySource {
        var normals = model.vertices.map { $0.normal ?? .zero }
        if normals.contains(where: { simd_length_squared($0) < 0.01 }) {
            normals = calculatedNormals(model: model)
        }
        var normalData = Data(capacity: normals.count * 3 * MemoryLayout<Float>.size)
        for value in normals {
            let normal = simd_length_squared(value) > 0 ? simd_normalize(value) : SIMD3<Float>(0, 1, 0)
            normalData.appendFloat(normal.x)
            normalData.appendFloat(normal.y)
            normalData.appendFloat(normal.z)
        }
        return SCNGeometrySource(
            data: normalData,
            semantic: .normal,
            vectorCount: normals.count,
            usesFloatComponents: true,
            componentsPerVector: 3,
            bytesPerComponent: MemoryLayout<Float>.size,
            dataOffset: 0,
            dataStride: MemoryLayout<Float>.size * 3
        )
    }

    private static func calculatedNormals(model: PLYModel) -> [SIMD3<Float>] {
        var normals = Array(repeating: SIMD3<Float>.zero, count: model.vertices.count)
        for face in model.faces {
            let a = Int(face.a)
            let b = Int(face.b)
            let c = Int(face.c)
            guard model.vertices.indices.contains(a),
                  model.vertices.indices.contains(b),
                  model.vertices.indices.contains(c) else { continue }
            let edgeAB = model.vertices[b].position - model.vertices[a].position
            let edgeAC = model.vertices[c].position - model.vertices[a].position
            let normal = simd_cross(edgeAB, edgeAC)
            normals[a] += normal
            normals[b] += normal
            normals[c] += normal
        }
        return normals
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
