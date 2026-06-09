import SwiftUI

struct ContentView: View {
    @StateObject private var scanner = ScanSessionManager()
    @AppStorage("backendBaseURL") private var backendBaseURL = "http://127.0.0.1:8000"
    @AppStorage("runVGGT") private var runVGGT = true
    @AppStorage("preserveColor") private var preserveColor = true
    @AppStorage("extractObject") private var extractObject = true
    @AppStorage("reconstructMesh") private var reconstructMesh = false
    @State private var showResult = false

    var body: some View {
        ZStack(alignment: .bottom) {
            ARScanView(session: scanner.session)
                .ignoresSafeArea()

            VStack(spacing: 12) {
                TextField("Backend URL", text: $backendBaseURL)
                    .textInputAutocapitalization(.never)
                    .keyboardType(.URL)
                    .autocorrectionDisabled()
                    .textFieldStyle(.roundedBorder)

                HStack {
                    statusPill(scanner.statusText)
                    Spacer()
                    statusPill("\(scanner.capturedFrameCount) frames")
                }

                Toggle(isOn: $runVGGT) {
                    Label("VGGT", systemImage: "sparkles")
                }
                .toggleStyle(.switch)
                .tint(.purple)

                Toggle(isOn: $preserveColor) {
                    Label("Color", systemImage: "paintpalette")
                }
                .toggleStyle(.switch)
                .tint(.orange)

                Toggle(isOn: $extractObject) {
                    Label("Object", systemImage: "scope")
                }
                .toggleStyle(.switch)
                .tint(.blue)

                Toggle(isOn: $reconstructMesh) {
                    Label("Mesh", systemImage: "cube")
                }
                .toggleStyle(.switch)
                .tint(.green)

                if let errorText = scanner.lastErrorText {
                    Text(errorText)
                        .font(.footnote)
                        .foregroundStyle(.red)
                        .lineLimit(3)
                        .frame(maxWidth: .infinity, alignment: .leading)
                }

                HStack(spacing: 12) {
                    Button {
                        scanner.isRecording ? scanner.stopScan() : scanner.startScan()
                    } label: {
                        Label(scanner.isRecording ? "Stop" : "Scan", systemImage: scanner.isRecording ? "stop.fill" : "record.circle")
                            .frame(maxWidth: .infinity)
                    }
                    .buttonStyle(.borderedProminent)
                    .tint(scanner.isRecording ? .red : .blue)
                    .disabled(!scanner.isSupported)

                    if let packageURL = scanner.lastPackageURL {
                        Button {
                            Task {
                                await scanner.uploadLatestPackage(
                                    backendBaseURL: backendBaseURL,
                                    runVGGT: runVGGT,
                                    preserveColor: preserveColor,
                                    extractObject: extractObject,
                                    reconstructMesh: reconstructMesh
                                )
                                showResult = scanner.resultURL != nil
                            }
                        } label: {
                            Label("Backend", systemImage: "network")
                                .frame(maxWidth: .infinity)
                        }
                        .buttonStyle(.borderedProminent)
                        .tint(.green)
                        .disabled(scanner.isUploading)

                        ShareLink(item: packageURL) {
                            Label("Export", systemImage: "square.and.arrow.up")
                                .frame(maxWidth: .infinity)
                        }
                        .buttonStyle(.bordered)
                    }
                }
            }
            .padding()
            .background(.ultraThinMaterial)
        }
        .task {
            scanner.startSession()
        }
        .sheet(isPresented: $showResult) {
            if let resultURL = scanner.resultURL {
                NavigationStack {
                    PointCloudPreview(resultURL: resultURL)
                        .navigationTitle("Backend Result")
                        .navigationBarTitleDisplayMode(.inline)
                }
            }
        }
    }

    private func statusPill(_ text: String) -> some View {
        Text(text)
            .font(.system(.callout, design: .monospaced))
            .padding(.horizontal, 10)
            .padding(.vertical, 6)
            .background(.black.opacity(0.55))
            .foregroundStyle(.white)
            .clipShape(Capsule())
    }
}

#Preview {
    ContentView()
}
