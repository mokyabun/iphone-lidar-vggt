import SwiftUI

struct ResultPreviewScreen: View {
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

extension BackendAssetKind {
    @MainActor
    static func allCasesAvailable(scanner: ScanSessionManager) -> [BackendAssetKind] {
        allCases.filter { scanner.assetIsAvailable($0) }
    }
}
