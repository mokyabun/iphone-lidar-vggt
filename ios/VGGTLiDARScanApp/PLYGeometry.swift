import Foundation
import SceneKit
import simd

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
