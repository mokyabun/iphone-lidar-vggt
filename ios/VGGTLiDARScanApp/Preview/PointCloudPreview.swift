import SwiftUI

struct PointCloudPreview: View {
    let resultURL: URL

    @State private var model: PLYModel?
    @State private var loadError: String?
    @State private var displayMode = PLYDisplayMode.color

    var body: some View {
        Group {
            if let loadError {
                ContentUnavailableView("Preview failed", systemImage: "exclamationmark.triangle", description: Text(loadError))
            } else if let model {
                ScenePLYView(model: model, displayMode: displayMode)
                    .ignoresSafeArea()
                    .overlay(alignment: .topLeading) {
                        modelSummary(model)
                    }
                    .overlay(alignment: .bottom) {
                        displayPicker(for: model)
                    }
            } else {
                ProgressView()
                    .task {
                        await loadModel()
                    }
            }
        }
    }

    private func modelSummary(_ model: PLYModel) -> some View {
        VStack(alignment: .leading, spacing: 4) {
            Label(
                model.faces.isEmpty ? "\(model.vertices.count) points" : "\(model.faces.count) faces",
                systemImage: model.faces.isEmpty ? "circle.grid.3x3.fill" : "cube.fill"
            )
            Text(model.sizeSummary)
                .foregroundStyle(.white.opacity(0.78))
        }
        .font(.system(.callout, design: .monospaced))
        .padding(10)
        .background(.black.opacity(0.62))
        .foregroundStyle(.white)
        .clipShape(RoundedRectangle(cornerRadius: 8))
        .padding()
    }

    @ViewBuilder
    private func displayPicker(for model: PLYModel) -> some View {
        if !model.faces.isEmpty {
            Picker("Display", selection: $displayMode) {
                ForEach(PLYDisplayMode.allCases) { mode in
                    Label(mode.title, systemImage: mode.systemImage)
                        .tag(mode)
                }
            }
            .pickerStyle(.segmented)
            .padding(8)
            .background(.ultraThinMaterial)
            .clipShape(RoundedRectangle(cornerRadius: 8))
            .padding()
        }
    }

    private func loadModel() async {
        do {
            model = try PLYParser.parseAscii(url: resultURL, maxPoints: 120_000)
        } catch {
            loadError = error.localizedDescription
        }
    }
}
