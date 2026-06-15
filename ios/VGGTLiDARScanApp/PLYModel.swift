import Foundation
import simd

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
