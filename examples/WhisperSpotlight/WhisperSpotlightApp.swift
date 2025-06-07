import SwiftUI

@main
struct WhisperSpotlightApp: App {
    @NSApplicationDelegateAdaptor(AppDelegate.self) var delegate
    var body: some Scene {
        WindowGroup {
            OverlayView()
        }
    }
}

class AppDelegate: NSObject, NSApplicationDelegate {
    private var hotkey: GlobalHotkey?
    func applicationDidFinishLaunching(_ notification: Notification) {
        hotkey = GlobalHotkey(keyCode: kVK_Space, modifiers: optionKey)
        hotkey?.handler = {
            NotificationCenter.default.post(name: .toggleOverlay, object: nil)
        }
    }
}
