import Foundation

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
