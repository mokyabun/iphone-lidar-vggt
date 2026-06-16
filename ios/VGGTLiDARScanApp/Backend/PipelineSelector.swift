import SwiftUI

struct PipelineSelector: View {
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
