import Foundation

struct BackendClient {
    let baseURL: URL

    func reconstruct(packageURL: URL, runVGGT: Bool) async throws -> URL {
        var components = URLComponents(url: baseURL.appendingPathComponent("reconstruct"), resolvingAgainstBaseURL: false)
        components?.queryItems = [URLQueryItem(name: "run_vggt", value: runVGGT ? "true" : "false")]
        guard let url = components?.url else {
            throw URLError(.badURL)
        }

        let boundary = "Boundary-\(UUID().uuidString)"
        var request = URLRequest(url: url)
        request.httpMethod = "POST"
        request.setValue("multipart/form-data; boundary=\(boundary)", forHTTPHeaderField: "Content-Type")

        var body = Data()
        body.appendString("--\(boundary)\r\n")
        body.appendString("Content-Disposition: form-data; name=\"scan_package\"; filename=\"ScanPackage.zip\"\r\n")
        body.appendString("Content-Type: application/zip\r\n\r\n")
        body.append(try Data(contentsOf: packageURL))
        body.appendString("\r\n--\(boundary)--\r\n")

        let (data, response) = try await URLSession.shared.upload(for: request, from: body)
        guard let httpResponse = response as? HTTPURLResponse, (200..<300).contains(httpResponse.statusCode) else {
            let message = String(data: data, encoding: .utf8) ?? "Backend request failed"
            throw BackendError.requestFailed(message)
        }

        let outputURL = FileManager.default.temporaryDirectory
            .appendingPathComponent("scan_final-\(UUID().uuidString).ply")
        try data.write(to: outputURL, options: .atomic)
        return outputURL
    }
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

