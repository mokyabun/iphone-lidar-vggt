import SwiftUI

struct BackendSettingsView: View {
    @Binding var backendBaseURL: String
    @Binding var enableSAM3ObjectMasking: Bool
    @Binding var sam3TextPrompt: String
    @Binding var enableLiDARScaleAlignment: Bool
    @Binding var enableMeshFragmentCleanup: Bool
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

                Section("Options") {
                    Toggle(isOn: $enableSAM3ObjectMasking) {
                        Label("SAM3 object masking", systemImage: "viewfinder")
                    }
                    if enableSAM3ObjectMasking {
                        TextField("Object prompt", text: $sam3TextPrompt, prompt: Text("toy boat"))
                            .textInputAutocapitalization(.never)
                            .autocorrectionDisabled()
                    }
                    if let capability = scanner.backendCapabilities?.features?.sam3ObjectMasking {
                        capabilityRow("SAM3 worker", capability: capability)
                    }

                    Toggle(isOn: $enableLiDARScaleAlignment) {
                        Label("LiDAR scale alignment", systemImage: "ruler")
                    }
                    Toggle(isOn: $enableMeshFragmentCleanup) {
                        Label("Mesh fragment cleanup", systemImage: "sparkles")
                    }
                    if let capability = scanner.backendCapabilities?.features?.lidarScaleAlignment {
                        capabilityRow("Scale layer", capability: capability)
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

    private func capabilityRow(_ title: String, capability: PipelineCapability) -> some View {
        VStack(alignment: .leading, spacing: 4) {
            HStack {
                Text(title)
                Spacer()
                Label(capability.state.title, systemImage: capability.state.systemImage)
                    .font(.footnote)
                    .foregroundStyle(capability.state.color)
            }
            if let reason = capability.reason {
                Text(reason)
                    .font(.footnote)
                    .foregroundStyle(.secondary)
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
