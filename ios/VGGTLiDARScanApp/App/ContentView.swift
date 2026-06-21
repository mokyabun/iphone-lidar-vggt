import SwiftUI

struct ContentView: View {
    @StateObject private var scanner = ScanSessionManager()
    @AppStorage("backendBaseURL") private var backendBaseURL = "http://127.0.0.1:8000"
    @AppStorage("enableSAM3ObjectMasking") private var enableSAM3ObjectMasking = false
    @AppStorage("sam3TextPrompt") private var sam3TextPrompt = ""
    @AppStorage("enableLiDARScaleAlignment") private var enableLiDARScaleAlignment = true
    @AppStorage("enableMeshFragmentCleanup") private var enableMeshFragmentCleanup = true
    @AppStorage("enableFloorSheetTrim") private var enableFloorSheetTrim = false
    @State private var showResult = false
    @State private var showSettings = false

    private var capability: PipelineCapability {
        scanner.pipelineCapability
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
                enableSAM3ObjectMasking: $enableSAM3ObjectMasking,
                sam3TextPrompt: $sam3TextPrompt,
                enableLiDARScaleAlignment: $enableLiDARScaleAlignment,
                enableMeshFragmentCleanup: $enableMeshFragmentCleanup,
                enableFloorSheetTrim: $enableFloorSheetTrim,
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
            if let message = statusMessage {
                Label(message.text, systemImage: message.systemImage)
                    .font(.footnote)
                    .foregroundStyle(message.color)
                    .lineLimit(2)
                    .frame(maxWidth: .infinity, alignment: .leading)
            }

            modePicker

            HStack(spacing: 10) {
                captureControls

                if let packageURL = scanner.lastPackageURL {
                    Button {
                        Task {
                            await scanner.uploadLatestPackage(
                                backendBaseURL: backendBaseURL,
                                options: BackendReconstructionOptions(
                                    enableSAM3ObjectMasking: enableSAM3ObjectMasking,
                                    enableLiDARScaleAlignment: enableLiDARScaleAlignment,
                                    enableMeshFragmentCleanup: enableMeshFragmentCleanup,
                                    enableFloorSheetTrim: enableFloorSheetTrim,
                                    sam3TextPrompt: sam3TextPrompt
                                )
                            )
                            showResult = scanner.resultURL != nil
                        }
                    } label: {
                        if scanner.isUploading {
                            ProgressView()
                                .frame(maxWidth: .infinity, minHeight: 28)
                        } else {
                            Label("Process", systemImage: ScanPipeline.reconviagen.systemImage)
                                .frame(maxWidth: .infinity, minHeight: 28)
                        }
                    }
                    .buttonStyle(.borderedProminent)
                    .tint(ScanPipeline.reconviagen.tint)
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

    private var modePicker: some View {
        Picker("Capture mode", selection: Binding(
            get: { scanner.captureMode },
            set: { scanner.setMode($0) }
        )) {
            ForEach(CaptureMode.allCases) { mode in
                Text(mode.title).tag(mode)
            }
        }
        .pickerStyle(.segmented)
        .disabled(scanner.isRecording || scanner.isCapturingPhoto || scanner.isUploading)
    }

    @ViewBuilder
    private var captureControls: some View {
        switch scanner.captureMode {
        case .video:
            videoButton
        case .photo:
            photoButtons
        }
    }

    @ViewBuilder
    private var videoButton: some View {
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

    @ViewBuilder
    private var photoButtons: some View {
        Button {
            scanner.captureSinglePhoto()
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
        .disabled(!scanner.isSupported || scanner.isUploading || scanner.isCapturingPhoto)

        Button(role: .destructive) {
            scanner.resetPhotos()
        } label: {
            Label("Reset", systemImage: "arrow.counterclockwise")
                .frame(maxWidth: .infinity, minHeight: 28)
        }
        .buttonStyle(.bordered)
        .disabled(scanner.capturedFrameCount == 0 || scanner.isCapturingPhoto || scanner.isUploading)
    }

    private var statusSystemImage: String {
        if scanner.isRecording {
            return "record.circle.fill"
        }
        if scanner.isCapturingPhoto {
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
    }
}

#Preview {
    ContentView()
}
