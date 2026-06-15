import Foundation

struct BackendClient {
    let baseURL: URL

    func capabilities() async throws -> BackendCapabilities {
        let url = baseURL.appendingPathComponent("capabilities")
        let (data, response) = try await URLSession.shared.data(from: url)
        try validate(response: response, data: data)
        return try JSONDecoder.scanBackend.decode(BackendCapabilities.self, from: data)
    }

    func reconstruct(packageURL: URL, options: ReconstructionOptions) async throws -> BackendReconstructionResult {
        let submission = try await submitJob(packageURL: packageURL, options: options)
        let status = try await waitForJob(submission.jobID, timeout: options.pipeline.timeout)
        let outputURL = try await downloadAsset(jobID: submission.jobID, kind: .result)
        return BackendReconstructionResult(
            outputURL: outputURL,
            jobID: submission.jobID,
            metrics: status.metrics
        )
    }

    private func submitJob(packageURL: URL, options: ReconstructionOptions) async throws -> BackendJobSubmission {
        var components = URLComponents(url: baseURL.appendingPathComponent("jobs"), resolvingAgainstBaseURL: false)
        components?.queryItems = [
            URLQueryItem(name: "run_vggt", value: options.pipeline == .vggt ? "true" : "false"),
            URLQueryItem(name: "preserve_color", value: options.preserveColor ? "true" : "false"),
            URLQueryItem(name: "extract_object", value: options.effectiveObject ? "true" : "false"),
            URLQueryItem(name: "reconstruct_mesh", value: options.effectiveMesh ? "true" : "false"),
            URLQueryItem(name: "ai_mesh", value: options.pipeline == .aiMesh ? "true" : "false")
        ]
        guard let url = components?.url else {
            throw URLError(.badURL)
        }

        let boundary = "Boundary-\(UUID().uuidString)"
        var request = URLRequest(url: url)
        request.httpMethod = "POST"
        request.timeoutInterval = 180
        request.setValue("multipart/form-data; boundary=\(boundary)", forHTTPHeaderField: "Content-Type")

        var body = Data()
        body.appendString("--\(boundary)\r\n")
        body.appendString("Content-Disposition: form-data; name=\"scan_package\"; filename=\"ScanPackage.zip\"\r\n")
        body.appendString("Content-Type: application/zip\r\n\r\n")
        body.append(try Data(contentsOf: packageURL))
        body.appendString("\r\n--\(boundary)--\r\n")

        let configuration = URLSessionConfiguration.default
        configuration.timeoutIntervalForRequest = 180
        configuration.timeoutIntervalForResource = 300
        let session = URLSession(configuration: configuration)
        let (data, response) = try await session.upload(for: request, from: body)
        try validate(response: response, data: data)
        return try decodeJSON(BackendJobSubmission.self, from: data)
    }

    private func waitForJob(_ jobID: String, timeout: TimeInterval) async throws -> BackendJobStatus {
        let deadline = Date().addingTimeInterval(timeout)
        let url = baseURL
            .appendingPathComponent("jobs")
            .appendingPathComponent(jobID)
        while Date() < deadline {
            let (data, response) = try await URLSession.shared.data(from: url)
            try validate(response: response, data: data)
            let status = try decodeJSON(BackendJobStatus.self, from: data)
            switch status.state {
            case "complete":
                return status
            case "failed":
                throw BackendError.requestFailed(status.error ?? "Reconstruction failed.")
            default:
                try await Task.sleep(for: .seconds(3))
            }
        }
        throw BackendError.requestFailed("Reconstruction timed out while waiting for the backend job.")
    }

    func downloadAsset(jobID: String, kind: BackendAssetKind) async throws -> URL {
        let url = baseURL
            .appendingPathComponent("jobs")
            .appendingPathComponent(jobID)
            .appendingPathComponent(kind.rawValue)
        let (downloadURL, response) = try await URLSession.shared.download(from: url)
        try validate(response: response, data: nil)
        let destination = temporaryURL(filename: "\(jobID)-\(kind.filename)")
        try? FileManager.default.removeItem(at: destination)
        try FileManager.default.moveItem(at: downloadURL, to: destination)
        return destination
    }

    @discardableResult
    private func validate(response: URLResponse, data: Data?) throws -> HTTPURLResponse {
        guard let httpResponse = response as? HTTPURLResponse else {
            throw URLError(.badServerResponse)
        }
        if httpResponse.mimeType == "text/html" {
            throw BackendError.requestFailed(
                "RunPod returned an HTML proxy page instead of the requested result."
            )
        }
        guard (200..<300).contains(httpResponse.statusCode) else {
            let detail = errorDetail(statusCode: httpResponse.statusCode, data: data)
            throw BackendError.requestFailed(detail)
        }
        return httpResponse
    }

    private func decodeJSON<T: Decodable>(_ type: T.Type, from data: Data) throws -> T {
        if looksLikeHTML(data) {
            throw BackendError.requestFailed(
                "RunPod returned an HTML proxy page instead of API data. The request likely exceeded the proxy timeout."
            )
        }
        do {
            return try JSONDecoder.scanBackend.decode(type, from: data)
        } catch {
            throw BackendError.requestFailed("The backend returned an invalid response.")
        }
    }

    private func errorDetail(statusCode: Int, data: Data?) -> String {
        if let data, looksLikeHTML(data) {
            return "RunPod proxy error (HTTP \(statusCode)). The backend job may still be running."
        }
        return data
            .flatMap { try? JSONDecoder().decode(BackendErrorPayload.self, from: $0).detail }
            ?? data.flatMap { String(data: $0, encoding: .utf8) }
            ?? HTTPURLResponse.localizedString(forStatusCode: statusCode)
    }

    private func looksLikeHTML(_ data: Data) -> Bool {
        let prefix = String(decoding: data.prefix(128), as: UTF8.self)
            .trimmingCharacters(in: .whitespacesAndNewlines)
            .lowercased()
        return prefix.hasPrefix("<!doctype html") || prefix.hasPrefix("<html")
    }

    private func temporaryURL(filename: String) -> URL {
        FileManager.default.temporaryDirectory.appendingPathComponent(filename)
    }
}

private struct BackendJobSubmission: Decodable {
    let jobID: String
}

private struct BackendJobStatus: Decodable {
    let state: String
    let error: String?
    let metrics: BackendMetrics?

    enum CodingKeys: String, CodingKey {
        case state = "status"
        case error
        case metrics
    }
}

enum BackendError: LocalizedError {
    case requestFailed(String)

    var errorDescription: String? {
        switch self {
        case .requestFailed(let message): return message
        }
    }
}

private struct BackendErrorPayload: Decodable {
    let detail: String
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
