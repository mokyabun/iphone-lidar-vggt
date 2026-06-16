import SceneKit
import simd
import SwiftUI

struct ScenePLYView: UIViewRepresentable {
    let model: PLYModel
    let displayMode: PLYDisplayMode

    func makeUIView(context: Context) -> SCNView {
        let view = SCNView()
        view.allowsCameraControl = true
        view.backgroundColor = UIColor(white: 0.07, alpha: 1)
        view.antialiasingMode = .multisampling4X
        configure(view)
        return view
    }

    func updateUIView(_ uiView: SCNView, context: Context) {
        configure(uiView)
    }

    private func configure(_ view: SCNView) {
        let prepared = makeScene()
        view.scene = prepared.scene
        view.pointOfView = prepared.camera
        view.defaultCameraController.target = prepared.target
        view.defaultCameraController.inertiaEnabled = true
    }

    private func makeScene() -> (scene: SCNScene, camera: SCNNode, target: SCNVector3) {
        let scene = SCNScene()
        let bounds = model.bounds
        let center = bounds.center
        let floorY = bounds.minimum.y
        let modelOffset = SIMD3<Float>(-center.x, -floorY, -center.z)

        let geometry = PLYGeometry.makeGeometry(model: model, offset: modelOffset, displayMode: displayMode)
        scene.rootNode.addChildNode(SCNNode(geometry: geometry))

        let horizontalExtent = max(bounds.extent.x, bounds.extent.z)
        let gridSize = max(horizontalExtent * 3.0, model.gridSpacing * 12.0)
        scene.rootNode.addChildNode(makeFloor(size: gridSize))
        scene.rootNode.addChildNode(makeGrid(size: gridSize, spacing: model.gridSpacing))

        let target = SCNVector3(0, max(bounds.extent.y * 0.45, 0.01), 0)
        let largestExtent = max(bounds.extent.x, bounds.extent.y, bounds.extent.z, 0.04)
        let cameraDistance = largestExtent * 2.35
        let cameraNode = makeCamera(distance: cameraDistance, extent: largestExtent, target: target, scene: scene)
        addLighting(to: scene, target: target, distance: cameraDistance)
        return (scene, cameraNode, target)
    }

    private func makeCamera(distance: Float, extent: Float, target: SCNVector3, scene: SCNScene) -> SCNNode {
        let cameraNode = SCNNode()
        let camera = SCNCamera()
        camera.fieldOfView = 48
        camera.zNear = Double(max(extent * 0.015, 0.0005))
        camera.zFar = Double(max(extent * 30, 5))
        camera.wantsHDR = true
        camera.wantsExposureAdaptation = true
        cameraNode.camera = camera
        cameraNode.position = SCNVector3(distance * 0.85, distance * 0.62, distance)

        let lookAt = SCNLookAtConstraint(target: makeTargetNode(target, in: scene))
        lookAt.isGimbalLockEnabled = true
        cameraNode.constraints = [lookAt]
        scene.rootNode.addChildNode(cameraNode)
        return cameraNode
    }

    private func makeTargetNode(_ position: SCNVector3, in scene: SCNScene) -> SCNNode {
        let node = SCNNode()
        node.position = position
        scene.rootNode.addChildNode(node)
        return node
    }

    private func makeFloor(size: Float) -> SCNNode {
        let plane = SCNPlane(width: CGFloat(size), height: CGFloat(size))
        let material = SCNMaterial()
        material.diffuse.contents = UIColor(white: 0.16, alpha: 1)
        material.roughness.contents = 0.92
        material.metalness.contents = 0
        plane.materials = [material]
        let node = SCNNode(geometry: plane)
        node.eulerAngles.x = -.pi / 2
        node.position.y = -0.001
        node.castsShadow = false
        return node
    }

    private func makeGrid(size: Float, spacing: Float) -> SCNNode {
        let half = size / 2
        let lineCount = min(80, max(4, Int(ceil(size / spacing))))
        let actualSpacing = size / Float(lineCount)
        var positions: [SCNVector3] = []
        positions.reserveCapacity((lineCount + 1) * 4)
        for index in 0...lineCount {
            let value = -half + Float(index) * actualSpacing
            positions.append(SCNVector3(value, 0, -half))
            positions.append(SCNVector3(value, 0, half))
            positions.append(SCNVector3(-half, 0, value))
            positions.append(SCNVector3(half, 0, value))
        }
        let source = SCNGeometrySource(vertices: positions)
        let element = SCNGeometryElement(
            data: nil,
            primitiveType: .line,
            primitiveCount: positions.count / 2,
            bytesPerIndex: 0
        )
        let geometry = SCNGeometry(sources: [source], elements: [element])
        let material = SCNMaterial()
        material.lightingModel = .constant
        material.diffuse.contents = UIColor(white: 0.62, alpha: 0.24)
        material.writesToDepthBuffer = false
        geometry.materials = [material]
        let node = SCNNode(geometry: geometry)
        node.position.y = 0.0002
        node.renderingOrder = 2
        return node
    }

    private func addLighting(to scene: SCNScene, target: SCNVector3, distance: Float) {
        let ambientNode = SCNNode()
        ambientNode.light = SCNLight()
        ambientNode.light?.type = .ambient
        ambientNode.light?.intensity = 520
        ambientNode.light?.color = UIColor(white: 0.72, alpha: 1)
        scene.rootNode.addChildNode(ambientNode)

        let keyNode = SCNNode()
        keyNode.light = SCNLight()
        keyNode.light?.type = .directional
        keyNode.light?.intensity = 1_350
        keyNode.light?.castsShadow = true
        keyNode.light?.shadowRadius = 5
        keyNode.position = SCNVector3(distance, distance * 1.7, distance)
        keyNode.constraints = [SCNLookAtConstraint(target: makeTargetNode(target, in: scene))]
        scene.rootNode.addChildNode(keyNode)

        let fillNode = SCNNode()
        fillNode.light = SCNLight()
        fillNode.light?.type = .omni
        fillNode.light?.intensity = 560
        fillNode.position = SCNVector3(-distance, distance * 0.8, -distance * 0.4)
        scene.rootNode.addChildNode(fillNode)
    }
}
