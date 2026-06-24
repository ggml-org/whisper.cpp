/*
  Simple DirectMedia Layer
  Copyright (C) 1997-2025 Sam Lantinga <slouken@libsdl.org>

  This software is provided 'as-is', without any express or implied
  warranty.  In no event will the authors be held liable for any damages
  arising from the use of this software.

  Permission is granted to anyone to use this software for any purpose,
  including commercial applications, and to alter it and redistribute it
  freely, subject to the following restrictions:

  1. The origin of this software must not be misrepresented; you must not
     claim that you wrote the original software. If you use this software
     in a product, an acknowledgment in the product documentation would be
     appreciated but is not required.
  2. Altered source versions must be plainly marked as such, and must not be
     misrepresented as being the original software.
  3. This notice may not be removed or altered from any source distribution.
*/
#include <Windows.h>

extern int SDL_WinRTInitNonXAMLApp(int (*mainFunction)(int, char **));

ref class SDL_WinRTApp sealed : public Windows::ApplicationModel::Core::IFrameworkView
{
  public:
    SDL_WinRTApp();

    // IFrameworkView Methods.
    virtual void Initialize(Windows::ApplicationModel::Core::CoreApplicationView ^ applicationView);
    virtual void SetWindow(Windows::UI::Core::CoreWindow ^ window);
    virtual void Load(Platform::String ^ entryPoint);
    virtual void Run();
    virtual void Uninitialize();

    internal :
        // SDL-specific methods
        void
        PumpEvents();

  protected:
    bool ShouldWaitForAppResumeEvents();

    // Event Handlers.

#if (WINAPI_FAMILY == WINAPI_FAMILY_APP) && (NTDDI_VERSION < NTDDI_WIN10) // for Windows 8/8.1/RT apps... (and not Phone apps)
    void OnSettingsPaneCommandsRequested(
        Windows::UI::ApplicationSettings::SettingsPane ^ p,
        Windows::UI::ApplicationSettings::SettingsPaneCommandsRequestedEventArgs ^ args);
#endif // if (WINAPI_FAMILY == WINAPI_FAMILY_APP) && (NTDDI_VERSION < NTDDI_WIN10)

#if NTDDI_VERSION > NTDDI_WIN8
    void OnOrientationChanged(Windows::Graphics::Display::DisplayInformation ^ sender, Platform::Object ^ args);
#else
    void OnOrientationChanged(Platform::Object ^ sender);
#endif
    void OnWindowSizeChanged(Windows::UI::Core::CoreWindow ^ sender, Windows::UI::Core::WindowSizeChangedEventArgs ^ args);
    void OnLogicalDpiChanged(Platform::Object ^ sender);
    void OnAppActivated(Windows::ApplicationModel::Core::CoreApplicationView ^ applicationView, Windows::ApplicationModel::Activation::IActivatedEventArgs ^ args);
    void OnSuspending(Platform::Object ^ sender, Windows::ApplicationModel::SuspendingEventArgs ^ args);
    void OnResuming(Platform::Object ^ sender, Platform::Object ^ args);
    void OnExiting(Platform::Object ^ sender, Platform::Object ^ args);
    void OnWindowActivated(Windows::UI::Core::CoreWindow ^ sender, Windows::UI::Core::WindowActivatedEventArgs ^ args);
    void OnWindowClosed(Windows::UI::Core::CoreWindow ^ sender, Windows::UI::Core::CoreWindowEventArgs ^ args);
    void OnVisibilityChanged(Windows::UI::Core::CoreWindow ^ sender, Windows::UI::Core::VisibilityChangedEventArgs ^ args);
    void OnPointerPressed(Windows::UI::Core::CoreWindow ^ sender, Windows::UI::Core::PointerEventArgs ^ args);
    void OnPointerReleased(Windows::UI::Core::CoreWindow ^ sender, Windows::UI::Core::PointerEventArgs ^ args);
    void OnPointerWheelChanged(Windows::UI::Core::CoreWindow ^ sender, Windows::UI::Core::PointerEventArgs ^ args);
    void OnPointerMoved(Windows::UI::Core::CoreWindow ^ sender, Windows::UI::Core::PointerEventArgs ^ args);
    void OnPointerEntered(Windows::UI::Core::CoreWindow ^ sender, Windows::UI::Core::PointerEventArgs ^ args);
    void OnPointerExited(Windows::UI::Core::CoreWindow ^ sender, Windows::UI::Core::PointerEventArgs ^ args);
    void OnMouseMoved(Windows::Devices::Input::MouseDevice ^ mouseDevice, Windows::Devices::Input::MouseEventArgs ^ args);
    void OnKeyDown(Windows::UI::Core::CoreWindow ^ sender, Windows::UI::Core::KeyEventArgs ^ args);
    void OnKeyUp(Windows::UI::Core::CoreWindow ^ sender, Windows::UI::Core::KeyEventArgs ^ args);
    void OnCharacterReceived(Windows::UI::Core::CoreWindow ^ sender, Windows::UI::Core::CharacterReceivedEventArgs ^ args);

#if NTDDI_VERSION >= NTDDI_WIN10
    void OnBackButtonPressed(Platform::Object ^ sender, Windows::UI::Core::BackRequestedEventArgs ^ args);
#elif SDL_WINAPI_FAMILY_PHONE
    void OnBackButtonPressed(Platform::Object ^ sender, Windows::Phone::UI::Input::BackPressedEventArgs ^ args);
#endif

#if NTDDI_VERSION >= NTDDI_WIN10
    void OnGamepadAdded(Platform::Object ^ sender, Windows::Gaming::Input::Gamepad ^ gamepad);
#endif

  private:
    bool m_windowClosed;
    bool m_windowVisible;
};

extern SDL_WinRTApp ^ SDL_WinRTGlobalApp;
