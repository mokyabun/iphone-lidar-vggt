import Foundation

struct BackendClient {
    let baseURL: URL

    func reconstruct(
        packageURL: URL,
        runVGGT: Bool,
        preserveColor: Bool,
        extractObject: Bool,
        reconstructMesh: Bool
    ) async throws -> BackendReconstructionResult {
        var components = URLComponents(url: baseURL.appendingPathComponent("reconstruct"), resolvingAgainstBaseURL: false)
        components?.queryItems = [
            URLQueryItem(name: "run_vggt", value: runVGGT ? "true" : "false"),
            URLQueryItem(name: "preserve_color", value: preserveColor ? "true" : "false"),
            URLQueryItem(name: "extract_object", value: extractObject ? "true" : "false"),
            URLQueryItem(name: "reconstruct_mesh", value: reconstructMesh ? "true" : "false")
        ]
        guard let url = components?.url else {
            throw URLError(.badURL)
        }

        let boundary = "Boundary-\(UUID().uuidString)"
        var request = URLRequest(url: url)
        request.httpMethod = "POST"
        request.timeoutInterval = 600
        request.setValue("multipart/form-data; boundary=\(boundary)", forHTTPHeaderField: "Content-Type")

        var body = Data()
        body.appendString("--\(boundary)\r\n")
        body.appendString("Content-Disposition: form-data; name=\"scan_package\"; filename=\"ScanPackage.zip\"\r\n")
        body.appendString("Content-Type: application/zip\r\n\r\n")
        body.append(try Data(contentsOf: packageURL))
        body.appendString("\r\n--\(boundary)--\r\n")

        let configuration = URLSessionConfiguration.default
        configuration.timeoutIntervalForRequest = 600
        configuration.timeoutIntervalForResource = 900
        let session = URLSession(configuration: configuration)
        let (data, response) = try await session.upload(for: request, from: body)
        guard let httpResponse = response as? HTTPURLResponse, (200..<300).contains(httpResponse.statusCode) else {
            let message = String(data: data, encoding: .utf8) ?? "Backend request failed"
            throw BackendError.requestFailed(message)
        }

        let outputURL = FileManager.default.temporaryDirectory
            .appendingPathComponent("scan_final-\(UUID().uuidString).ply")
        try data.write(to: outputURL, options: .atomic)
        let metrics = httpResponse.value(forHTTPHeaderField: "X-Reconstruction-Metrics")
            .flatMap { $0.data(using: .utf8) }
            .flatMap { try? JSONDecoder.scanBackend.decode(BackendMetrics.self, from: $0) }
        return BackendReconstructionResult(outputURL: outputURL, metrics: metrics)
    }
}

struct BackendReconstructionResult {
    let outputURL: URL
    let metrics: BackendMetrics?
}

struct BackendMetrics: Decodable {
    let frameCount: Int
    let selectedKeyframes: Int
    let lidarPoints: Int
    let vggtPoints: Int
    let meshVertices: Int
    let meshFaces: Int
    let meshMethod: String?
    let finalOutputType: String
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
}

enum BackendError: LocalizedError {
    case requestFailed(String)

    var errorDescription: String? {
        switch self {
        case .requestFailed(let message):
            return message
        }
    }
}

private extension Data {
    mutating func appendString(_ string: String) {
        append(Data(string.utf8))
    }
}

private extension JSONDecoder {
    static var scanBackend: JSONDecoder {
        let decoder = JSONDecoder()
        decoder.keyDecodingStrategy = .convertFromSnakeCase
        return decoder
    }
}
