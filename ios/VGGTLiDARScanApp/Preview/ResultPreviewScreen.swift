import QuickLook
import SwiftUI

struct ResultPreviewScreen: View {
    @ObservedObject var scanner: ScanSessionManager
    let backendBaseURL: String
    let resultURL: URL
    @Environment(\.dismiss) private var dismiss
    @State private var selectedTab = ResultTab.result

    var body: some View {
        NavigationStack {
            TabView(selection: $selectedTab) {
                PointCloudPreview(resultURL: resultURL)
                    .tabItem {
                        Label("Result", systemImage: "cube.transparent")
                    }
                    .tag(ResultTab.result)

                ResultDetailsView(
                    scanner: scanner,
                    backendBaseURL: backendBaseURL
                )
                .tabItem {
                    Label("Details", systemImage: "list.bullet.rectangle")
                }
                .tag(ResultTab.details)

                if scanner.assetIsAvailable(.preview) {
                    ResultAssetPreview(
                        scanner: scanner,
                        backendBaseURL: backendBaseURL,
                        kind: .preview
                    )
                    .tabItem {
                        Label("Preview", systemImage: "paintpalette")
                    }
                    .tag(ResultTab.preview)
                }
            }
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

private enum ResultTab {
    case result
    case details
    case preview
}

private struct ResultDetailsView: View {
    @ObservedObject var scanner: ScanSessionManager
    let backendBaseURL: String

    var body: some View {
        ScrollView {
            VStack(alignment: .leading, spacing: 18) {
                if let metrics = scanner.resultMetrics {
                    summarySection(metrics)
                    geometrySection(metrics)
                    scaleSection(metrics)
                    if !metrics.warnings.isEmpty {
                        warningsSection(metrics.warnings)
                    }
                } else {
                    ContentUnavailableView(
                        "No backend details",
                        systemImage: "list.bullet.rectangle"
                    )
                }

                assetsSection
            }
            .padding()
        }
        .background(Color(.systemGroupedBackground))
    }

    private func summarySection(_ metrics: BackendMetrics) -> some View {
        resultSection("Summary", systemImage: "checkmark.seal") {
            detailRow("Job", scanner.resultJobID ?? "local", systemImage: "number")
            detailRow("Output", metrics.finalOutputType.capitalized, systemImage: "cube.transparent")
            if let source = metrics.finalOutputSource {
                detailRow("Source", source, systemImage: "arrow.triangle.branch")
            }
            detailRow("Frames", "\(metrics.frameCount)", systemImage: "photo.stack")
            detailRow("Keyframes", "\(metrics.selectedKeyframes)", systemImage: "key.viewfinder")
            if let inputViews = metrics.inputViews {
                detailRow("Input views", "\(inputViews)", systemImage: "photo.on.rectangle")
            }
        }
    }

    private func geometrySection(_ metrics: BackendMetrics) -> some View {
        resultSection("Geometry", systemImage: "cube.fill") {
            detailRow("LiDAR points", formatCount(metrics.lidarPoints), systemImage: "circle.grid.3x3.fill")
            if let scenePoints = metrics.scenePoints {
                detailRow("Scene points", formatCount(scenePoints), systemImage: "ruler")
            }
            detailRow("Mesh vertices", formatCount(metrics.meshVertices), systemImage: "point.3.connected.trianglepath.dotted")
            detailRow("Mesh faces", formatCount(metrics.meshFaces), systemImage: "triangleshape.fill")
        }
    }

    private func scaleSection(_ metrics: BackendMetrics) -> some View {
        resultSection("Scale", systemImage: "ruler") {
            if let extent = metrics.objectExtentM ?? metrics.sceneExtentM {
                detailRow("Size", formatExtent(extent), systemImage: "arrow.up.left.and.arrow.down.right")
            }
            if let rmse = metrics.alignmentRmseM {
                detailRow("Alignment ICP", "\(formatMillimeters(rmse)) mm", systemImage: "target")
            }
            if let scale = metrics.alignmentScale {
                detailRow("Alignment scale", String(format: "%.3f", scale), systemImage: "scale.3d")
            }
            if let watertight = metrics.printMeshWatertight {
                detailRow("Print mesh", watertight ? "Watertight" : "Needs repair", systemImage: "printer")
            }
        }
    }

    private func warningsSection(_ warnings: [String]) -> some View {
        resultSection("Warnings", systemImage: "exclamationmark.triangle.fill") {
            ForEach(warnings, id: \.self) { warning in
                Label(warning, systemImage: "exclamationmark.triangle.fill")
                    .font(.footnote)
                    .foregroundStyle(.orange)
                    .frame(maxWidth: .infinity, alignment: .leading)
            }
        }
    }

    private var assetsSection: some View {
        resultSection("Files", systemImage: "folder") {
            ForEach(BackendAssetKind.allCasesAvailable(scanner: scanner)) { kind in
                assetRow(kind)
            }
        }
    }

    private func assetRow(_ kind: BackendAssetKind) -> some View {
        HStack(spacing: 12) {
            Image(systemName: kind.systemImage)
                .font(.title3)
                .frame(width: 28)
                .foregroundStyle(kind == .preview ? .pink : .blue)

            VStack(alignment: .leading, spacing: 2) {
                Text(kind.title)
                    .font(.subheadline.weight(.semibold))
                Text(kind.filename)
                    .font(.caption)
                    .foregroundStyle(.secondary)
            }

            Spacer()

            if scanner.downloadingAssets.contains(kind) {
                ProgressView()
            } else if let url = scanner.assetURL(for: kind) {
                ShareLink(item: url) {
                    Image(systemName: "square.and.arrow.up")
                }
                .accessibilityLabel("Export \(kind.title)")
            } else {
                Button {
                    Task {
                        await scanner.downloadAsset(kind, backendBaseURL: backendBaseURL)
                    }
                } label: {
                    Image(systemName: "arrow.down.circle")
                }
                .accessibilityLabel("Download \(kind.title)")
            }
        }
        .padding(.vertical, 3)
    }

    private func resultSection<Content: View>(
        _ title: String,
        systemImage: String,
        @ViewBuilder content: () -> Content
    ) -> some View {
        VStack(alignment: .leading, spacing: 10) {
            Label(title, systemImage: systemImage)
                .font(.headline)
            VStack(alignment: .leading, spacing: 9) {
                content()
            }
            .padding(14)
            .frame(maxWidth: .infinity, alignment: .leading)
            .background(Color(.secondarySystemGroupedBackground))
            .clipShape(RoundedRectangle(cornerRadius: 8))
        }
    }

    private func detailRow(_ title: String, _ value: String, systemImage: String) -> some View {
        HStack(alignment: .firstTextBaseline, spacing: 10) {
            Image(systemName: systemImage)
                .foregroundStyle(.secondary)
                .frame(width: 22)
            Text(title)
                .foregroundStyle(.secondary)
            Spacer(minLength: 12)
            Text(value)
                .fontWeight(.semibold)
                .multilineTextAlignment(.trailing)
        }
        .font(.subheadline)
    }

    private func formatCount(_ value: Int) -> String {
        value.formatted(.number)
    }

    private func formatMeters(_ value: Double) -> String {
        "\(String(format: "%.2f", value)) m"
    }

    private func formatMillimeters(_ value: Double) -> String {
        String(format: "%.1f", value * 1_000)
    }

    private func formatExtent(_ values: [Double]) -> String {
        values
            .prefix(3)
            .map { String(format: "%.2f", $0) }
            .joined(separator: " x ") + " m"
    }
}

private struct ResultAssetPreview: View {
    @ObservedObject var scanner: ScanSessionManager
    let backendBaseURL: String
    let kind: BackendAssetKind

    var body: some View {
        Group {
            if let url = scanner.assetURL(for: kind) {
                QuickLookFilePreview(url: url)
            } else if scanner.downloadingAssets.contains(kind) {
                ProgressView("Downloading \(kind.title)")
            } else {
                ContentUnavailableView {
                    Label("Preview not downloaded", systemImage: kind.systemImage)
                } description: {
                    Text(kind.filename)
                } actions: {
                    Button {
                        Task {
                            await scanner.downloadAsset(kind, backendBaseURL: backendBaseURL)
                        }
                    } label: {
                        Label("Download Preview", systemImage: "arrow.down.circle")
                    }
                    .buttonStyle(.borderedProminent)
                }
            }
        }
    }
}

private struct QuickLookFilePreview: UIViewControllerRepresentable {
    let url: URL

    func makeCoordinator() -> Coordinator {
        Coordinator(url: url)
    }

    func makeUIViewController(context: Context) -> QLPreviewController {
        let controller = QLPreviewController()
        controller.dataSource = context.coordinator
        return controller
    }

    func updateUIViewController(_ controller: QLPreviewController, context: Context) {
        context.coordinator.url = url
        controller.reloadData()
    }

    final class Coordinator: NSObject, QLPreviewControllerDataSource {
        var url: URL

        init(url: URL) {
            self.url = url
        }

        func numberOfPreviewItems(in controller: QLPreviewController) -> Int {
            1
        }

        func previewController(
            _ controller: QLPreviewController,
            previewItemAt index: Int
        ) -> QLPreviewItem {
            url as NSURL
        }
    }
}

extension BackendAssetKind {
    @MainActor
    static func allCasesAvailable(scanner: ScanSessionManager) -> [BackendAssetKind] {
        allCases.filter { scanner.assetIsAvailable($0) }
    }
}
