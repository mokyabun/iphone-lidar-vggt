import ARKit
import RealityKit
import SwiftUI

struct ARScanView: UIViewRepresentable {
    let session: ARSession

    func makeUIView(context: Context) -> ARView {
        let view = ARView(frame: .zero)
        view.automaticallyConfigureSession = false
        view.session = session
        return view
    }

    func updateUIView(_ uiView: ARView, context: Context) {
        uiView.session = session
    }
}

