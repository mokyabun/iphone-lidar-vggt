import SwiftUI

struct BackendSettingsView: View {
    @Binding var backendBaseURL: String
    @ObservedObject var scanner: ScanSessionManager
    let refresh: () async -> Void
    @Environment(\.dismiss) private var dismiss

    var body: some View {
        NavigationStack {
            Form {
                Section("Backend") {
                    TextField("URL", text: $backendBaseURL)
                        .textInputAutocapitalization(.never)
                        .keyboardType(.URL)
                        .autocorrectionDisabled()
                    Button {
                        Task { await refresh() }
                    } label: {
                        Label(
                            scanner.isCheckingBackend ? "Checking" : "Refresh",
                            systemImage: scanner.isCheckingBackend ? "arrow.triangle.2.circlepath" : "arrow.clockwise"
                        )
                    }
                    .disabled(scanner.isCheckingBackend)
                }

                Section("Pipeline") {
                    let pipelineCapability = scanner.pipelineCapability
                    HStack {
                        Label(ScanPipeline.reconviagen.title, systemImage: ScanPipeline.reconviagen.systemImage)
                        Spacer()
                        Label(
                            pipelineCapability.state.title,
                            systemImage: pipelineCapability.state.systemImage
                        )
                        .font(.footnote)
                        .foregroundStyle(pipelineCapability.state.color)
                    }
                    if let reason = pipelineCapability.reason {
                        Text(reason)
                            .font(.footnote)
                            .foregroundStyle(.secondary)
                    }
                }
            }
            .navigationTitle("Backend")
            .navigationBarTitleDisplayMode(.inline)
            .toolbar {
                ToolbarItem(placement: .confirmationAction) {
                    Button("Done") { dismiss() }
                }
            }
        }
    }
}

extension PipelineState {
    var title: String {
        switch self {
        case .available: return "Ready"
        case .loading: return "Loading"
        case .unavailable: return "Unavailable"
        }
    }

    var systemImage: String {
        switch self {
        case .available: return "checkmark.circle.fill"
        case .loading: return "clock.fill"
        case .unavailable: return "xmark.circle.fill"
        }
    }

    var color: Color {
        switch self {
        case .available: return .green
        case .loading: return .orange
        case .unavailable: return .red
        }
    }
}
