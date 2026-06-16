import SwiftUI

struct ContentView: View {
    @StateObject private var scanner = ScanSessionManager()
    @AppStorage("backendBaseURL") private var backendBaseURL = "http://127.0.0.1:8000"
    @AppStorage("scanPipeline") private var pipelineRawValue = ScanPipeline.aiMesh.rawValue
    @AppStorage("preserveColor") private var preserveColor = true
    @AppStorage("extractObject") private var extractObject = true
    @AppStorage("reconstructMesh") private var reconstructMesh = true
    @AppStorage("captureMode") private var captureModeRawValue = CaptureMode.video.rawValue
    @State private var showResult = false
    @State private var showSettings = false

    private var captureMode: CaptureMode {
        CaptureMode(rawValue: captureModeRawValue) ?? .video
    }

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
            statusPill(scanner.statusText, systemImage: statusSystemImage)
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

            Picker("Capture mode", selection: Binding(
                get: { captureMode },
                set: { captureModeRawValue = $0.rawValue }
            )) {
                ForEach(CaptureMode.allCases) { mode in
                    Text(mode.title).tag(mode)
                }
            }
            .pickerStyle(.segmented)
            .disabled(scanner.isRecording || scanner.isCapturingPhoto || scanner.isUploading)

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
                captureButton

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
                    .disabled(scanner.isUploading || scanner.isRecording || scanner.isCapturingPhoto || !capability.isAvailable)

                    ShareLink(item: packageURL) {
                        Image(systemName: "square.and.arrow.up")
                            .frame(width: 28, height: 28)
                    }
                    .buttonStyle(.bordered)
                    .accessibilityLabel("Export scan package")
                }
            }

            if scanner.resultURL != nil {
                Button {
                    showResult = true
                } label: {
                    Label("Open Result", systemImage: "list.bullet.rectangle")
                        .frame(maxWidth: .infinity, minHeight: 28)
                }
                .buttonStyle(.bordered)
                .disabled(scanner.isUploading || scanner.isRecording || scanner.isCapturingPhoto)
            }
        }
        .padding(.horizontal, 14)
        .padding(.top, 14)
        .padding(.bottom, 10)
        .background(.ultraThinMaterial)
    }

    @ViewBuilder
    private var captureButton: some View {
        switch captureMode {
        case .photo:
            Button {
                scanner.capturePhoto()
            } label: {
                if scanner.isCapturingPhoto {
                    ProgressView()
                        .frame(maxWidth: .infinity, minHeight: 28)
                } else {
                    Label("Capture", systemImage: "camera.fill")
                        .frame(maxWidth: .infinity, minHeight: 28)
                }
            }
            .buttonStyle(.borderedProminent)
            .tint(.blue)
            .disabled(!scanner.isSupported || scanner.isUploading || scanner.isCapturingPhoto || scanner.isRecording)
        case .video:
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
            .disabled(!scanner.isSupported || scanner.isUploading || scanner.isCapturingPhoto)
        }
    }

    private var statusSystemImage: String {
        if scanner.isRecording {
            return "record.circle.fill"
        }
        if scanner.isCapturingPhoto || captureMode == .photo {
            return "camera.fill"
        }
        return "viewfinder"
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

private enum CaptureMode: String, CaseIterable, Identifiable {
    case photo
    case video

    var id: String { rawValue }

    var title: String {
        switch self {
        case .photo: return "Photo"
        case .video: return "Video"
        }
    }
}

#Preview {
    ContentView()
}
