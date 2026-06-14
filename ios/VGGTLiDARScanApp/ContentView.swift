import SwiftUI

struct ContentView: View {
    @StateObject private var scanner = ScanSessionManager()
    @AppStorage("backendBaseURL") private var backendBaseURL = "http://127.0.0.1:8000"
    @AppStorage("scanPipeline") private var pipelineRawValue = ScanPipeline.aiMesh.rawValue
    @AppStorage("preserveColor") private var preserveColor = true
    @AppStorage("extractObject") private var extractObject = true
    @AppStorage("reconstructMesh") private var reconstructMesh = true
    @State private var showResult = false
    @State private var showSettings = false

    private var pipeline: ScanPipeline {
        ScanPipeline(rawValue: pipelineRawValue) ?? .aiMesh
    }

    private var capability: PipelineCapability {
        scanner.capability(for: pipeline)
    }

    private var options: ReconstructionOptions {
        ReconstructionOptions(
            pipeline: pipeline,
            preserveColor: preserveColor,
            extractObject: extractObject,
            reconstructMesh: reconstructMesh
        )
    }

    var body: some View {
        ZStack(alignment: .bottom) {
            ARScanView(session: scanner.session)
                .ignoresSafeArea()

            VStack {
                topBar
                Spacer()
                controlPanel
            }
        }
        .task {
            scanner.startSession()
            await refreshBackend()
        }
        .sheet(isPresented: $showSettings) {
            BackendSettingsView(
                backendBaseURL: $backendBaseURL,
                scanner: scanner,
                refresh: refreshBackend
            )
        }
        .sheet(isPresented: $showResult) {
            if let resultURL = scanner.resultURL {
                ResultPreviewScreen(
                    scanner: scanner,
                    backendBaseURL: backendBaseURL,
                    resultURL: resultURL
                )
            }
        }
    }

    private var topBar: some View {
        HStack(spacing: 8) {
            statusPill(scanner.statusText, systemImage: scanner.isRecording ? "record.circle.fill" : "viewfinder")
            statusPill("\(scanner.capturedFrameCount)", systemImage: "photo.stack")
            Spacer()
            Button {
                showSettings = true
            } label: {
                Image(systemName: "gearshape.fill")
                    .frame(width: 42, height: 42)
            }
            .buttonStyle(.borderedProminent)
            .buttonBorderShape(.circle)
            .tint(.black.opacity(0.58))
            .accessibilityLabel("Backend settings")
        }
        .padding(.horizontal)
        .padding(.top, 8)
    }

    private var controlPanel: some View {
        VStack(spacing: 12) {
            PipelineSelector(
                selection: Binding(
                    get: { pipeline },
                    set: { pipelineRawValue = $0.rawValue }
                ),
                capability: scanner.capability(for:)
            )

            HStack(spacing: 18) {
                optionToggle(
                    "Color",
                    systemImage: "paintpalette.fill",
                    isOn: $preserveColor,
                    supported: capability.options.contains("color")
                )
                optionToggle(
                    "Object",
                    systemImage: "scope",
                    isOn: pipeline == .aiMesh ? .constant(true) : $extractObject,
                    supported: capability.options.contains("object"),
                    locked: pipeline == .aiMesh
                )
                optionToggle(
                    "Mesh",
                    systemImage: "cube.fill",
                    isOn: pipeline == .aiMesh ? .constant(true) : pipeline == .vggt ? .constant(false) : $reconstructMesh,
                    supported: capability.options.contains("mesh"),
                    locked: pipeline != .metric
                )
            }
            .frame(maxWidth: .infinity)

            if let message = statusMessage {
                Label(message.text, systemImage: message.systemImage)
                    .font(.footnote)
                    .foregroundStyle(message.color)
                    .lineLimit(2)
                    .frame(maxWidth: .infinity, alignment: .leading)
            }

            HStack(spacing: 10) {
                Button {
                    scanner.isRecording ? scanner.stopScan() : scanner.startScan()
                } label: {
                    Label(
                        scanner.isRecording ? "Stop" : "Scan",
                        systemImage: scanner.isRecording ? "stop.fill" : "record.circle"
                    )
                    .frame(maxWidth: .infinity, minHeight: 28)
                }
                .buttonStyle(.borderedProminent)
                .tint(scanner.isRecording ? .red : .blue)
                .disabled(!scanner.isSupported || scanner.isUploading)

                if let packageURL = scanner.lastPackageURL {
                    Button {
                        Task {
                            await scanner.uploadLatestPackage(
                                backendBaseURL: backendBaseURL,
                                options: options
                            )
                            showResult = scanner.resultURL != nil
                        }
                    } label: {
                        if scanner.isUploading {
                            ProgressView()
                                .frame(maxWidth: .infinity, minHeight: 28)
                        } else {
                            Label("Process", systemImage: pipeline.systemImage)
                                .frame(maxWidth: .infinity, minHeight: 28)
                        }
                    }
                    .buttonStyle(.borderedProminent)
                    .tint(pipeline.tint)
                    .disabled(scanner.isUploading || scanner.isRecording || !capability.isAvailable)

                    ShareLink(item: packageURL) {
                        Image(systemName: "square.and.arrow.up")
                            .frame(width: 28, height: 28)
                    }
                    .buttonStyle(.bordered)
                    .accessibilityLabel("Export scan package")
                }
            }
        }
        .padding(.horizontal, 14)
        .padding(.top, 14)
        .padding(.bottom, 10)
        .background(.ultraThinMaterial)
    }

    private var statusMessage: (text: String, systemImage: String, color: Color)? {
        if let errorText = scanner.lastErrorText {
            return (errorText, "exclamationmark.triangle.fill", .red)
        }
        if let scaleText = scanner.lastScaleText {
            return (scaleText, "ruler", .secondary)
        }
        if capability.state == .loading {
            return (capability.reason ?? "Pipeline loading.", "clock.fill", .secondary)
        }
        if capability.state == .unavailable {
            return (capability.reason ?? "Pipeline unavailable.", "xmark.circle.fill", .red)
        }
        return nil
    }

    private func optionToggle(
        _ title: String,
        systemImage: String,
        isOn: Binding<Bool>,
        supported: Bool,
        locked: Bool = false
    ) -> some View {
        VStack(spacing: 7) {
            HStack(spacing: 4) {
                Image(systemName: systemImage)
                Text(title)
                    .lineLimit(1)
                if locked {
                    Image(systemName: "lock.fill")
                        .font(.caption2)
                }
            }
            .font(.caption.weight(.semibold))
            .frame(maxWidth: .infinity)

            Toggle("", isOn: isOn)
                .labelsHidden()
                .toggleStyle(.switch)
        }
        .disabled(!supported || locked)
        .opacity(supported ? 1 : 0.35)
        .frame(maxWidth: .infinity, minHeight: 62)
    }

    private func statusPill(_ text: String, systemImage: String) -> some View {
        Label(text, systemImage: systemImage)
            .font(.system(.callout, design: .monospaced).weight(.medium))
            .padding(.horizontal, 10)
            .frame(height: 38)
            .background(.black.opacity(0.58))
            .foregroundStyle(.white)
            .clipShape(Capsule())
    }

    private func refreshBackend() async {
        await scanner.refreshCapabilities(backendBaseURL: backendBaseURL)
        if !scanner.capability(for: pipeline).isAvailable,
           let fallback = ScanPipeline.allCases.first(where: { scanner.capability(for: $0).isAvailable }) {
            pipelineRawValue = fallback.rawValue
        }
    }
}

private struct PipelineSelector: View {
    @Binding var selection: ScanPipeline
    let capability: (ScanPipeline) -> PipelineCapability

    var body: some View {
        HStack(spacing: 4) {
            ForEach(ScanPipeline.allCases) { pipeline in
                let pipelineCapability = capability(pipeline)
                Button {
                    selection = pipeline
                } label: {
                    VStack(spacing: 4) {
                        Image(systemName: pipelineCapability.state == .loading ? "clock.fill" : pipeline.systemImage)
                            .font(.headline)
                        Text(pipeline.title)
                            .font(.caption.weight(.semibold))
                            .lineLimit(1)
                    }
                    .foregroundStyle(selection == pipeline ? Color.white : Color.primary)
                    .frame(maxWidth: .infinity)
                    .frame(height: 52)
                    .background(selection == pipeline ? pipeline.tint : Color.clear)
                    .clipShape(RoundedRectangle(cornerRadius: 7))
                }
                .buttonStyle(.plain)
                .disabled(!pipelineCapability.isAvailable)
                .opacity(pipelineCapability.state == .unavailable ? 0.34 : 1)
                .accessibilityValue(pipelineCapability.state.rawValue)
            }
        }
        .padding(3)
        .background(Color.secondary.opacity(0.13))
        .clipShape(RoundedRectangle(cornerRadius: 8))
    }
}

private struct BackendSettingsView: View {
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

                Section("Pipelines") {
                    ForEach(ScanPipeline.allCases) { pipeline in
                        let pipelineCapability = scanner.capability(for: pipeline)
                        HStack {
                            Label(pipeline.title, systemImage: pipeline.systemImage)
                            Spacer()
                            Label(
                                pipelineCapability.state.title,
                                systemImage: pipelineCapability.state.systemImage
                            )
                            .font(.footnote)
                            .foregroundStyle(pipelineCapability.state.color)
                        }
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

private struct ResultPreviewScreen: View {
    @ObservedObject var scanner: ScanSessionManager
    let backendBaseURL: String
    let resultURL: URL
    @Environment(\.dismiss) private var dismiss

    var body: some View {
        NavigationStack {
            PointCloudPreview(resultURL: resultURL)
                .navigationTitle("Result")
                .navigationBarTitleDisplayMode(.inline)
                .overlay(alignment: .bottom) {
                    if !scanner.downloadingAssets.isEmpty {
                        Label("Downloading", systemImage: "arrow.down.circle")
                            .font(.callout.weight(.medium))
                            .padding(.horizontal, 12)
                            .frame(height: 38)
                            .background(.ultraThinMaterial)
                            .clipShape(Capsule())
                            .padding(.bottom, 70)
                    }
                }
                .toolbar {
                    ToolbarItem(placement: .topBarLeading) {
                        Button {
                            dismiss()
                        } label: {
                            Image(systemName: "xmark")
                        }
                    }
                    ToolbarItem(placement: .topBarTrailing) {
                        assetMenu
                    }
                }
        }
    }

    private var assetMenu: some View {
        Menu {
            ForEach(BackendAssetKind.allCasesAvailable(scanner: scanner)) { kind in
                if let url = scanner.assetURL(for: kind) {
                    ShareLink(item: url) {
                        Label("Export \(kind.title)", systemImage: kind.systemImage)
                    }
                } else {
                    Button {
                        Task {
                            await scanner.downloadAsset(kind, backendBaseURL: backendBaseURL)
                        }
                    } label: {
                        Label("Download \(kind.title)", systemImage: "arrow.down.circle")
                    }
                    .disabled(scanner.downloadingAssets.contains(kind))
                }
            }
        } label: {
            Image(systemName: "square.and.arrow.down")
        }
        .accessibilityLabel("Result files")
    }
}

private extension BackendAssetKind {
    @MainActor
    static func allCasesAvailable(scanner: ScanSessionManager) -> [BackendAssetKind] {
        allCases.filter { scanner.assetIsAvailable($0) }
    }
}

private extension PipelineState {
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

#Preview {
    ContentView()
}
