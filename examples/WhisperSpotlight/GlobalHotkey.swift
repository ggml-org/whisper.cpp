import Carbon
import Foundation

class GlobalHotkey {
    var handler: (() -> Void)?
    private var ref: EventHotKeyRef?

    init(keyCode: UInt32, modifiers: UInt32) {
        var hotKeyID = EventHotKeyID(signature: OSType(0x1234), id: UInt32(keyCode))
        RegisterEventHotKey(keyCode, modifiers, hotKeyID, GetApplicationEventTarget(), 0, &ref)
        let eventSpec = EventTypeSpec(eventClass: OSType(kEventClassKeyboard), eventKind: UInt32(kEventHotKeyPressed))
        InstallEventHandler(GetApplicationEventTarget(), { _, evt, ctx in
            let hotKeyIDPtr = UnsafeMutablePointer<EventHotKeyID>.allocate(capacity: 1)
            GetEventParameter(evt!, EventParamName(kEventParamDirectObject), EventParamType(typeEventHotKeyID), nil, MemoryLayout<EventHotKeyID>.size, nil, hotKeyIDPtr)
            Unmanaged<GlobalHotkey>.fromOpaque(ctx!).takeUnretainedValue().handler?()
            return noErr
        }, 1, [eventSpec], Unmanaged.passUnretained(self).toOpaque(), nil)
    }

    deinit {
        if let ref { UnregisterEventHotKey(ref) }
    }
}

let optionKey: UInt32 = UInt32(cmdKey) >> 8
