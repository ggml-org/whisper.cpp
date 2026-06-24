include(CMakeParseArguments)
include(${SDL2_SOURCE_DIR}/cmake/sdlfind.cmake)
macro(FindLibraryAndSONAME _LIB)
  cmake_parse_arguments(FLAS "" "" "LIBDIRS" ${ARGN})

  string(TOUPPER ${_LIB} _UPPERLNAME)
  string(REGEX REPLACE "\\-" "_" _LNAME "${_UPPERLNAME}")

  find_library(${_LNAME}_LIB ${_LIB} PATHS ${FLAS_LIBDIRS})

  if(${_LNAME}_LIB MATCHES ".*\\${CMAKE_SHARED_LIBRARY_SUFFIX}.*" AND NOT ${_LNAME}_LIB MATCHES ".*\\${CMAKE_STATIC_LIBRARY_SUFFIX}.*")
    set(${_LNAME}_SHARED TRUE)
  else()
    set(${_LNAME}_SHARED FALSE)
  endif()

  if(${_LNAME}_LIB)
    # reduce the library name for shared linking

    get_filename_component(_LIB_REALPATH ${${_LNAME}_LIB} REALPATH)  # resolves symlinks
    get_filename_component(_LIB_DIRECTORY ${_LIB_REALPATH} DIRECTORY)
    get_filename_component(_LIB_JUSTNAME ${_LIB_REALPATH} NAME)

    if(APPLE)
      string(REGEX REPLACE "(\\.[0-9]*)\\.[0-9\\.]*dylib$" "\\1.dylib" _LIB_REGEXD "${_LIB_JUSTNAME}")
    else()
      string(REGEX REPLACE "(\\.[0-9]*)\\.[0-9\\.]*$" "\\1" _LIB_REGEXD "${_LIB_JUSTNAME}")
    endif()

    if(NOT EXISTS "${_LIB_DIRECTORY}/${_LIB_REGEXD}")
      set(_LIB_REGEXD "${_LIB_JUSTNAME}")
    endif()
    set(${_LNAME}_LIBDIR "${_LIB_LIBDIR}")

    SET(_DEBUG_FindSONAME FALSE)
    if(_DEBUG_FindSONAME)
      message_warn("DYNLIB OUTPUTVAR: ${_LIB} ... ${_LNAME}_LIB")
      message_warn("DYNLIB ORIGINAL LIB: ${_LIB} ... ${${_LNAME}_LIB}")
      message_warn("DYNLIB REALPATH LIB: ${_LIB} ... ${_LIB_REALPATH}")
      message_warn("DYNLIB JUSTNAME LIB: ${_LIB} ... ${_LIB_JUSTNAME}")
      message_warn("DYNLIB REGEX'd LIB: ${_LIB} ... ${_LIB_REGEXD}")
    endif()

    message(STATUS "dynamic lib${_LIB} -> ${_LIB_REGEXD}")
    set(${_LNAME}_LIB_SONAME ${_LIB_REGEXD})
  endif()
endmacro()

macro(CheckDLOPEN)
  cmake_push_check_state(RESET)
  check_symbol_exists(dlopen "dlfcn.h" HAVE_DLOPEN_IN_LIBC)
  if(NOT HAVE_DLOPEN_IN_LIBC)
    set(CMAKE_REQUIRED_LIBRARIES dl)
    check_symbol_exists(dlopen "dlfcn.h" HAVE_DLOPEN_IN_LIBDL)
    if(HAVE_DLOPEN_IN_LIBDL)
      list(APPEND EXTRA_LIBS dl)
    endif()
  endif()
  if(HAVE_DLOPEN_IN_LIBC OR HAVE_DLOPEN_IN_LIBDL)
    set(HAVE_DLOPEN TRUE)
  endif()
  cmake_pop_check_state()
endmacro()

macro(CheckO_CLOEXEC)
  check_c_source_compiles("
    #include <fcntl.h>
    int flag = O_CLOEXEC;
    int main(int argc, char **argv) { return 0; }" HAVE_O_CLOEXEC)
endmacro()

# Requires:
# - n/a
macro(CheckOSS)
  if(SDL_OSS)
    check_c_source_compiles("
        #include <sys/soundcard.h>
        int main(int argc, char **argv) { int arg = SNDCTL_DSP_SETFRAGMENT; return 0; }" HAVE_OSS_SYS_SOUNDCARD_H)

    if(HAVE_OSS_SYS_SOUNDCARD_H)
      set(HAVE_OSS TRUE)
      file(GLOB OSS_SOURCES ${SDL2_SOURCE_DIR}/src/audio/dsp/*.c)
      set(SDL_AUDIO_DRIVER_OSS 1)
      list(APPEND SOURCE_FILES ${OSS_SOURCES})
      if(NETBSD)
        list(APPEND EXTRA_LIBS ossaudio)
      endif()
      set(HAVE_SDL_AUDIO TRUE)
    endif()
  endif()
endmacro()

# Requires:
# - n/a
# Optional:
# - SDL_ALSA_SHARED opt
# - HAVE_SDL_LOADSO opt
macro(CheckALSA)
  if(SDL_ALSA)
    sdlFindALSA()
    if(ALSA_FOUND)
      file(GLOB ALSA_SOURCES "${SDL2_SOURCE_DIR}/src/audio/alsa/*.c")
      list(APPEND SOURCE_FILES ${ALSA_SOURCES})
      set(SDL_AUDIO_DRIVER_ALSA 1)
      set(HAVE_ALSA TRUE)
      set(HAVE_ALSA_SHARED FALSE)
      if(SDL_ALSA_SHARED)
        if(HAVE_SDL_LOADSO)
          FindLibraryAndSONAME("asound")
          if(ASOUND_LIB AND ASOUND_SHARED)
            target_include_directories(sdl-build-options SYSTEM INTERFACE $<TARGET_PROPERTY:ALSA::ALSA,INTERFACE_INCLUDE_DIRECTORIES>)
            set(SDL_AUDIO_DRIVER_ALSA_DYNAMIC "\"${ASOUND_LIB_SONAME}\"")
            set(HAVE_ALSA_SHARED TRUE)
          else()
            message(WARNING "Unable to find asound shared object")
          endif()
        else()
          message(WARNING "You must have SDL_LoadObject() support for dynamic ALSA loading")
        endif()
      endif()
      if(NOT HAVE_ALSA_SHARED)
        list(APPEND CMAKE_DEPENDS ALSA::ALSA)
        list(APPEND PKGCONFIG_DEPENDS alsa)
      endif()
      set(HAVE_SDL_AUDIO TRUE)
    else()
      set(HAVE_ALSA FALSE)
      message(WARNING "Unable to found the alsa development library")
    endif()
  endif()
endmacro()

# Requires:
# - PkgCheckModules
# Optional:
# - SDL_PIPEWIRE_SHARED opt
# - HAVE_SDL_LOADSO opt
macro(CheckPipewire)
    if(SDL_PIPEWIRE)
        pkg_check_modules(PKG_PIPEWIRE libpipewire-0.3>=0.3.20)
        if(PKG_PIPEWIRE_FOUND)
            set(HAVE_PIPEWIRE TRUE)
            file(GLOB PIPEWIRE_SOURCES ${SDL2_SOURCE_DIR}/src/audio/pipewire/*.c)
            list(APPEND SOURCE_FILES ${PIPEWIRE_SOURCES})
            set(SDL_AUDIO_DRIVER_PIPEWIRE 1)
            list(APPEND EXTRA_CFLAGS ${PKG_PIPEWIRE_CFLAGS})
            if(SDL_PIPEWIRE_SHARED AND NOT HAVE_SDL_LOADSO)
                message_warn("You must have SDL_LoadObject() support for dynamic Pipewire loading")
            endif()
            FindLibraryAndSONAME("pipewire-0.3" LIBDIRS ${PKG_PIPEWIRE_LIBRARY_DIRS})
            if(SDL_PIPEWIRE_SHARED AND PIPEWIRE_0.3_LIB AND HAVE_SDL_LOADSO)
                set(SDL_AUDIO_DRIVER_PIPEWIRE_DYNAMIC "\"${PIPEWIRE_0.3_LIB_SONAME}\"")
                set(HAVE_PIPEWIRE_SHARED TRUE)
            else()
                list(APPEND EXTRA_LDFLAGS ${PKG_PIPEWIRE_LDFLAGS})
            endif()
            set(HAVE_SDL_AUDIO TRUE)
        endif()
    endif()
endmacro()

# Requires:
# - PkgCheckModules
# Optional:
# - SDL_PULSEAUDIO_SHARED opt
# - HAVE_SDL_LOADSO opt
macro(CheckPulseAudio)
  if(SDL_PULSEAUDIO)
    pkg_check_modules(PKG_PULSEAUDIO libpulse>=0.9.15)
    if(PKG_PULSEAUDIO_FOUND)
      set(HAVE_PULSEAUDIO TRUE)
      file(GLOB PULSEAUDIO_SOURCES ${SDL2_SOURCE_DIR}/src/audio/pulseaudio/*.c)
      list(APPEND SOURCE_FILES ${PULSEAUDIO_SOURCES})
      set(SDL_AUDIO_DRIVER_PULSEAUDIO 1)
      list(APPEND EXTRA_CFLAGS ${PKG_PULSEAUDIO_CFLAGS})
      if(SDL_PULSEAUDIO_SHARED AND NOT HAVE_SDL_LOADSO)
        message_warn("You must have SDL_LoadObject() support for dynamic PulseAudio loading")
      endif()
      FindLibraryAndSONAME("pulse" LIBDIRS ${PKG_PULSEAUDIO_LIBRARY_DIRS})
      if(SDL_PULSEAUDIO_SHARED AND PULSE_LIB AND HAVE_SDL_LOADSO)
        set(SDL_AUDIO_DRIVER_PULSEAUDIO_DYNAMIC "\"${PULSE_LIB_SONAME}\"")
        set(HAVE_PULSEAUDIO_SHARED TRUE)
      else()
        list(APPEND EXTRA_LDFLAGS ${PKG_PULSEAUDIO_LDFLAGS})
      endif()
      set(HAVE_SDL_AUDIO TRUE)
    endif()
  endif()
endmacro()

# Requires:
# - PkgCheckModules
# Optional:
# - SDL_JACK_SHARED opt
# - HAVE_SDL_LOADSO opt
macro(CheckJACK)
  if(SDL_JACK)
    pkg_check_modules(PKG_JACK jack)
    if(PKG_JACK_FOUND)
      set(HAVE_JACK TRUE)
      file(GLOB JACK_SOURCES ${SDL2_SOURCE_DIR}/src/audio/jack/*.c)
      list(APPEND SOURCE_FILES ${JACK_SOURCES})
      set(SDL_AUDIO_DRIVER_JACK 1)
      list(APPEND EXTRA_CFLAGS ${PKG_JACK_CFLAGS})
      if(SDL_JACK_SHARED AND NOT HAVE_SDL_LOADSO)
        message_warn("You must have SDL_LoadObject() support for dynamic JACK audio loading")
      endif()
      FindLibraryAndSONAME("jack" LIBDIRS ${PKG_JACK_LIBRARY_DIRS})
      if(SDL_JACK_SHARED AND JACK_LIB AND HAVE_SDL_LOADSO)
        set(SDL_AUDIO_DRIVER_JACK_DYNAMIC "\"${JACK_LIB_SONAME}\"")
        set(HAVE_JACK_SHARED TRUE)
      else()
        list(APPEND EXTRA_LDFLAGS ${PKG_JACK_LDFLAGS})
      endif()
      set(HAVE_SDL_AUDIO TRUE)
    endif()
  endif()
endmacro()

# Requires:
# - PkgCheckModules
# Optional:
# - SDL_ESD_SHARED opt
# - HAVE_SDL_LOADSO opt
macro(CheckESD)
  if(SDL_ESD)
    pkg_check_modules(PKG_ESD esound)
    if(PKG_ESD_FOUND)
      set(HAVE_ESD TRUE)
      file(GLOB ESD_SOURCES ${SDL2_SOURCE_DIR}/src/audio/esd/*.c)
      list(APPEND SOURCE_FILES ${ESD_SOURCES})
      set(SDL_AUDIO_DRIVER_ESD 1)
      list(APPEND EXTRA_CFLAGS ${PKG_ESD_CFLAGS})
      if(SDL_ESD_SHARED AND NOT HAVE_SDL_LOADSO)
          message_warn("You must have SDL_LoadObject() support for dynamic ESD loading")
      endif()
      FindLibraryAndSONAME(esd LIBDIRS ${PKG_ESD_LIBRARY_DIRS})
      if(SDL_ESD_SHARED AND ESD_LIB AND HAVE_SDL_LOADSO)
          set(SDL_AUDIO_DRIVER_ESD_DYNAMIC "\"${ESD_LIB_SONAME}\"")
          set(HAVE_ESD_SHARED TRUE)
      else()
          list(APPEND EXTRA_LDFLAGS ${PKG_ESD_LDFLAGS})
      endif()
      set(HAVE_SDL_AUDIO TRUE)
    endif()
  endif()
endmacro()

# Requires:
# - n/a
# Optional:
# - SDL_ARTS_SHARED opt
# - HAVE_SDL_LOADSO opt
macro(CheckARTS)
  if(SDL_ARTS)
    find_program(ARTS_CONFIG arts-config)
    if(ARTS_CONFIG)
      execute_process(CMD_ARTSCFLAGS ${ARTS_CONFIG} --cflags
        OUTPUT_VARIABLE ARTS_CFLAGS OUTPUT_STRIP_TRAILING_WHITESPACE)
      list(APPEND EXTRA_CFLAGS ${ARTS_CFLAGS})
      execute_process(CMD_ARTSLIBS ${ARTS_CONFIG} --libs
        OUTPUT_VARIABLE ARTS_LIBS OUTPUT_STRIP_TRAILING_WHITESPACE)
      file(GLOB ARTS_SOURCES ${SDL2_SOURCE_DIR}/src/audio/arts/*.c)
      list(APPEND SOURCE_FILES ${ARTS_SOURCES})
      set(SDL_AUDIO_DRIVER_ARTS 1)
      set(HAVE_ARTS TRUE)
      if(SDL_ARTS_SHARED AND NOT HAVE_SDL_LOADSO)
        message_warn("You must have SDL_LoadObject() support for dynamic ARTS loading")
      endif()
      FindLibraryAndSONAME(artsc)
      if(SDL_ARTS_SHARED AND ARTSC_LIB AND HAVE_SDL_LOADSO)
        # TODO
        set(SDL_AUDIO_DRIVER_ARTS_DYNAMIC "\"${ARTSC_LIB_SONAME}\"")
        set(HAVE_ARTS_SHARED TRUE)
      else()
        list(APPEND EXTRA_LDFLAGS ${ARTS_LIBS})
      endif()
      set(HAVE_SDL_AUDIO TRUE)
    endif()
  endif()
endmacro()

# Requires:
# - n/a
# Optional:
# - SDL_NAS_SHARED opt
# - HAVE_SDL_LOADSO opt
macro(CheckNAS)
  if(SDL_NAS)
    # TODO: set include paths properly, so the NAS headers are found
    check_include_file(audio/audiolib.h HAVE_NAS_H)
    find_library(D_NAS_LIB audio)
    if(HAVE_NAS_H AND D_NAS_LIB)
      set(HAVE_NAS TRUE)
      file(GLOB NAS_SOURCES ${SDL2_SOURCE_DIR}/src/audio/nas/*.c)
      list(APPEND SOURCE_FILES ${NAS_SOURCES})
      set(SDL_AUDIO_DRIVER_NAS 1)
      if(SDL_NAS_SHARED AND NOT HAVE_SDL_LOADSO)
        message_warn("You must have SDL_LoadObject() support for dynamic NAS loading")
      endif()
      FindLibraryAndSONAME("audio")
      if(SDL_NAS_SHARED AND AUDIO_LIB AND HAVE_SDL_LOADSO)
        set(SDL_AUDIO_DRIVER_NAS_DYNAMIC "\"${AUDIO_LIB_SONAME}\"")
        set(HAVE_NAS_SHARED TRUE)
      else()
        list(APPEND EXTRA_LIBS ${D_NAS_LIB})
      endif()
      set(HAVE_SDL_AUDIO TRUE)
    endif()
  endif()
endmacro()

# Requires:
# - PkgCheckModules
# Optional:
# - SDL_SNDIO_SHARED opt
# - HAVE_SDL_LOADSO opt
macro(CheckSNDIO)
  if(SDL_SNDIO)
    pkg_check_modules(PKG_SNDIO sndio)
    if(PKG_SNDIO_FOUND)
      set(HAVE_SNDIO TRUE)
      file(GLOB SNDIO_SOURCES ${SDL2_SOURCE_DIR}/src/audio/sndio/*.c)
      list(APPEND SOURCE_FILES ${SNDIO_SOURCES})
      set(SDL_AUDIO_DRIVER_SNDIO 1)
      list(APPEND EXTRA_CFLAGS ${PKG_SNDIO_CFLAGS})
      if(SDL_SNDIO_SHARED AND NOT HAVE_SDL_LOADSO)
        message_warn("You must have SDL_LoadObject() support for dynamic sndio loading")
      endif()
      FindLibraryAndSONAME("sndio" LIBDIRS ${PKG_SNDIO_LIBRARY_DIRS})
      if(SDL_SNDIO_SHARED AND SNDIO_LIB AND HAVE_SDL_LOADSO)
        set(SDL_AUDIO_DRIVER_SNDIO_DYNAMIC "\"${SNDIO_LIB_SONAME}\"")
        set(HAVE_SNDIO_SHARED TRUE)
      else()
        list(APPEND EXTRA_LIBS ${PKG_SNDIO_LDFLAGS})
      endif()
      set(HAVE_SDL_AUDIO TRUE)
    endif()
  endif()
endmacro()

# Requires:
# - PkgCheckModules
# Optional:
# - FUSIONSOUND_SHARED opt
# - HAVE_SDL_LOADSO opt
macro(CheckFusionSound)
  if(FUSIONSOUND)
    pkg_check_modules(PKG_FUSIONSOUND fusionsound>=1.0.0)
    if(PKG_FUSIONSOUND_FOUND)
      set(HAVE_FUSIONSOUND TRUE)
      file(GLOB FUSIONSOUND_SOURCES ${SDL2_SOURCE_DIR}/src/audio/fusionsound/*.c)
      list(APPEND SOURCE_FILES ${FUSIONSOUND_SOURCES})
      set(SDL_AUDIO_DRIVER_FUSIONSOUND 1)
      list(APPEND EXTRA_CFLAGS ${PKG_FUSIONSOUND_CFLAGS})
      if(FUSIONSOUND_SHARED AND NOT HAVE_SDL_LOADSO)
        message_warn("You must have SDL_LoadObject() support for dynamic FusionSound loading")
      endif()
      FindLibraryAndSONAME("fusionsound" LIBDIRS ${PKG_FUSIONSOUND_LIBRARY_DIRS})
      if(FUSIONSOUND_SHARED AND FUSIONSOUND_LIB AND HAVE_SDL_LOADSO)
        set(SDL_AUDIO_DRIVER_FUSIONSOUND_DYNAMIC "\"${FUSIONSOUND_LIB_SONAME}\"")
        set(HAVE_FUSIONSOUND_SHARED TRUE)
      else()
        list(APPEND EXTRA_LDFLAGS ${PKG_FUSIONSOUND_LDFLAGS})
      endif()
      set(HAVE_SDL_AUDIO TRUE)
    endif()
  endif()
endmacro()

# Requires:
# - SDL_LIBSAMPLERATE
# Optional:
# - SDL_LIBSAMPLERATE_SHARED opt
# - HAVE_SDL_LOADSO opt
macro(CheckLibSampleRate)
  if(SDL_LIBSAMPLERATE)
    find_package(SampleRate QUIET)
    if(SampleRate_FOUND AND TARGET SampleRate::samplerate)
      set(HAVE_LIBSAMPLERATE TRUE)
      set(HAVE_LIBSAMPLERATE_H TRUE)
      if(SDL_LIBSAMPLERATE_SHARED)
        target_include_directories(sdl-build-options SYSTEM INTERFACE $<TARGET_PROPERTY:SampleRate::samplerate,INTERFACE_INCLUDE_DIRECTORIES>)
        if(NOT HAVE_SDL_LOADSO)
          message_warn("You must have SDL_LoadObject() support for dynamic libsamplerate loading")
        else()
          get_property(_samplerate_type TARGET SampleRate::samplerate PROPERTY TYPE)
          if(_samplerate_type STREQUAL "SHARED_LIBRARY")
            set(HAVE_LIBSAMPLERATE_SHARED TRUE)
            if(WIN32 OR OS2)
              set(SDL_LIBSAMPLERATE_DYNAMIC "\"$<TARGET_FILE_NAME:SampleRate::samplerate>\"")
            else()
              set(SDL_LIBSAMPLERATE_DYNAMIC "\"$<TARGET_SONAME_FILE_NAME:SampleRate::samplerate>\"")
            endif()
          endif()
        endif()
      else()
        target_link_libraries(sdl-build-options INTERFACE SampleRate::samplerate)
        list(APPEND SDL_REQUIRES_PRIVATE SampleRate::samplerate)
      endif()
    else()
      check_include_file(samplerate.h HAVE_LIBSAMPLERATE_H)
      if(HAVE_LIBSAMPLERATE_H)
        set(HAVE_LIBSAMPLERATE TRUE)
        if(SDL_LIBSAMPLERATE_SHARED AND NOT HAVE_SDL_LOADSO)
          message_warn("You must have SDL_LoadObject() support for dynamic libsamplerate loading")
        endif()
        FindLibraryAndSONAME("samplerate")
        if(SDL_LIBSAMPLERATE_SHARED AND SAMPLERATE_LIB AND HAVE_SDL_LOADSO)
          set(SDL_LIBSAMPLERATE_DYNAMIC "\"${SAMPLERATE_LIB_SONAME}\"")
          set(HAVE_LIBSAMPLERATE_SHARED TRUE)
        else()
          list(APPEND EXTRA_LDFLAGS -lsamplerate)
        endif()
      endif()
    endif()
  endif()
endmacro()

# Requires:
# - n/a
# Optional:
# - SDL_X11_SHARED opt
# - HAVE_SDL_LOADSO opt
macro(CheckX11)
  cmake_push_check_state(RESET)
  if(SDL_X11)
    foreach(_LIB X11 Xext Xcursor Xi Xfixes Xrandr Xrender Xss)
        FindLibraryAndSONAME("${_LIB}")
    endforeach()

    set(X11_dirs)
    find_path(X_INCLUDEDIR
      NAMES X11/Xlib.h
      PATHS
        /usr/pkg/xorg/include
        /usr/X11R6/include
        /usr/X11R7/include
        /usr/local/include/X11
        /usr/include/X11
        /usr/openwin/include
        /usr/openwin/share/include
        /opt/graphics/OpenGL/include
        /opt/X11/include
    )

    if(X_INCLUDEDIR)
      list(APPEND EXTRA_CFLAGS "-I${X_INCLUDEDIR}")
      list(APPEND CMAKE_REQUIRED_INCLUDES ${X_INCLUDEDIR})
    endif()

    find_file(HAVE_XCURSOR_H NAMES "X11/Xcursor/Xcursor.h" HINTS "${X_INCLUDEDIR}")
    find_file(HAVE_XINPUT2_H NAMES "X11/extensions/XInput2.h" HINTS "${X_INCLUDEDIR}")
    find_file(HAVE_XRANDR_H NAMES "X11/extensions/Xrandr.h" HINTS "${X_INCLUDEDIR}")
    find_file(HAVE_XFIXES_H_ NAMES "X11/extensions/Xfixes.h" HINTS "${X_INCLUDEDIR}")
    find_file(HAVE_XRENDER_H NAMES "X11/extensions/Xrender.h" HINTS "${X_INCLUDEDIR}")
    find_file(HAVE_XSS_H NAMES "X11/extensions/scrnsaver.h" HINTS "${X_INCLUDEDIR}")
    find_file(HAVE_XSHAPE_H NAMES "X11/extensions/shape.h" HINTS "${X_INCLUDEDIR}")
    find_file(HAVE_XDBE_H NAMES "X11/extensions/Xdbe.h" HINTS "${X_INCLUDEDIR}")
    find_file(HAVE_XEXT_H NAMES "X11/extensions/Xext.h" HINTS "${X_INCLUDEDIR}")

    if(X11_LIB)
      if(NOT HAVE_XEXT_H)
        message_error("Missing Xext.h, maybe you need to install the libxext-dev package?")
      endif()

      set(HAVE_X11 TRUE)
      set(HAVE_SDL_VIDEO TRUE)

      file(GLOB X11_SOURCES ${SDL2_SOURCE_DIR}/src/video/x11/*.c)
      list(APPEND SOURCE_FILES ${X11_SOURCES})
      set(SDL_VIDEO_DRIVER_X11 1)

      # Note: Disabled on Apple because the dynamic mode backend for X11 doesn't
      # work properly on Apple during several issues like inconsistent paths
      # among platforms. See #6778 (https://github.com/libsdl-org/SDL/issues/6778)
      if(APPLE)
        set(SDL_X11_SHARED OFF)
      endif()

      check_symbol_exists(shmat "sys/shm.h" HAVE_SHMAT_IN_LIBC)
      if(NOT HAVE_SHMAT_IN_LIBC)
        check_library_exists(ipc shmat "" HAVE_SHMAT_IN_LIBIPC)
        if(HAVE_SHMAT_IN_LIBIPC)
          list(APPEND EXTRA_LIBS ipc)
        endif()
        if(NOT HAVE_SHMAT_IN_LIBIPC)
          list(APPEND EXTRA_CFLAGS "-DNO_SHARED_MEMORY")
        endif()
      endif()

      if(SDL_X11_SHARED)
        if(NOT HAVE_SDL_LOADSO)
          message_warn("You must have SDL_LoadObject() support for dynamic X11 loading")
          set(HAVE_X11_SHARED FALSE)
        else()
          set(HAVE_X11_SHARED TRUE)
        endif()
        if(X11_LIB)
          if(HAVE_X11_SHARED)
            set(SDL_VIDEO_DRIVER_X11_DYNAMIC "\"${X11_LIB_SONAME}\"")
          else()
            list(APPEND EXTRA_LIBS ${X11_LIB})
          endif()
        endif()
        if(XEXT_LIB)
          if(HAVE_X11_SHARED)
            set(SDL_VIDEO_DRIVER_X11_DYNAMIC_XEXT "\"${XEXT_LIB_SONAME}\"")
          else()
            list(APPEND EXTRA_LIBS ${XEXT_LIB_SONAME})
          endif()
        endif()
      else()
          list(APPEND EXTRA_LIBS ${X11_LIB} ${XEXT_LIB})
      endif()

      set(CMAKE_REQUIRED_LIBRARIES ${X11_LIB} ${X11_LIB})

      check_c_source_compiles("
          #include <X11/Xlib.h>
          int main(int argc, char **argv) {
            Display *display;
            XEvent event;
            XGenericEventCookie *cookie = &event.xcookie;
            XNextEvent(display, &event);
            XGetEventData(display, cookie);
            XFreeEventData(display, cookie);
            return 0; }" HAVE_XGENERICEVENT)
      if(HAVE_XGENERICEVENT)
        set(SDL_VIDEO_DRIVER_X11_SUPPORTS_GENERIC_EVENTS 1)
      endif()

      check_symbol_exists(XkbKeycodeToKeysym "X11/Xlib.h;X11/XKBlib.h" SDL_VIDEO_DRIVER_X11_HAS_XKBKEYCODETOKEYSYM)

      if(SDL_X11_XCURSOR AND HAVE_XCURSOR_H AND XCURSOR_LIB)
        set(HAVE_X11_XCURSOR TRUE)
        if(HAVE_X11_SHARED)
          set(SDL_VIDEO_DRIVER_X11_DYNAMIC_XCURSOR "\"${XCURSOR_LIB_SONAME}\"")
        else()
          list(APPEND EXTRA_LIBS ${XCURSOR_LIB})
        endif()
        set(SDL_VIDEO_DRIVER_X11_XCURSOR 1)
      endif()

      if(SDL_X11_XDBE AND HAVE_XDBE_H)
        set(HAVE_X11_XDBE TRUE)
        set(SDL_VIDEO_DRIVER_X11_XDBE 1)
      endif()

      if(SDL_X11_XINPUT AND HAVE_XINPUT2_H AND XI_LIB)
        set(HAVE_X11_XINPUT TRUE)
        if(HAVE_X11_SHARED)
          set(SDL_VIDEO_DRIVER_X11_DYNAMIC_XINPUT2 "\"${XI_LIB_SONAME}\"")
        else()
          list(APPEND EXTRA_LIBS ${XI_LIB})
        endif()
        set(SDL_VIDEO_DRIVER_X11_XINPUT2 1)

        # Check for multitouch
        check_c_source_compiles("
            #include <X11/Xlib.h>
            #include <X11/Xproto.h>
            #include <X11/extensions/XInput2.h>
            int event_type = XI_TouchBegin;
            XITouchClassInfo *t;
            Status XIAllowTouchEvents(Display *a,int b,unsigned int c,Window d,int f) {
              return (Status)0;
            }
            int main(int argc, char **argv) { return 0; }" HAVE_XINPUT2_MULTITOUCH)
        if(HAVE_XINPUT2_MULTITOUCH)
          set(SDL_VIDEO_DRIVER_X11_XINPUT2_SUPPORTS_MULTITOUCH 1)
        endif()
      endif()

      # check along with XInput2.h because we use Xfixes with XIBarrierReleasePointer
      if(SDL_X11_XFIXES AND HAVE_XFIXES_H_ AND HAVE_XINPUT2_H)
        check_c_source_compiles("
            #include <X11/Xlib.h>
            #include <X11/Xproto.h>
            #include <X11/extensions/XInput2.h>
            #include <X11/extensions/Xfixes.h>
            BarrierEventID b;
            int main(int argc, char **argv) { return 0; }" HAVE_XFIXES_H)
      endif()
      if(SDL_X11_XFIXES AND HAVE_XFIXES_H AND HAVE_XINPUT2_H AND XFIXES_LIB)
        if(HAVE_X11_SHARED)
          set(SDL_VIDEO_DRIVER_X11_DYNAMIC_XFIXES "\"${XFIXES_LIB_SONAME}\"")
        else()
          list(APPEND EXTRA_LIBS ${XFIXES_LIB})
        endif()
        set(SDL_VIDEO_DRIVER_X11_XFIXES 1)
        set(HAVE_X11_XFIXES TRUE)
      endif()

      if(SDL_X11_XRANDR AND HAVE_XRANDR_H AND XRANDR_LIB)
        if(HAVE_X11_SHARED)
          set(SDL_VIDEO_DRIVER_X11_DYNAMIC_XRANDR "\"${XRANDR_LIB_SONAME}\"")
        else()
          list(APPEND EXTRA_LIBS ${XRANDR_LIB})
        endif()
        set(SDL_VIDEO_DRIVER_X11_XRANDR 1)
        set(HAVE_X11_XRANDR TRUE)
      endif()

      if(SDL_X11_XSCRNSAVER AND HAVE_XSS_H AND XSS_LIB)
        if(HAVE_X11_SHARED)
          set(SDL_VIDEO_DRIVER_X11_DYNAMIC_XSS "\"${XSS_LIB_SONAME}\"")
        else()
          list(APPEND EXTRA_LIBS ${XSS_LIB})
        endif()
        set(SDL_VIDEO_DRIVER_X11_XSCRNSAVER 1)
        set(HAVE_X11_XSCRNSAVER TRUE)
      endif()

      if(SDL_X11_XSHAPE AND HAVE_XSHAPE_H)
        set(SDL_VIDEO_DRIVER_X11_XSHAPE 1)
        set(HAVE_X11_XSHAPE TRUE)
      endif()

      set(CMAKE_REQUIRED_LIBRARIES)
    endif()
  endif()
  if(NOT HAVE_X11)
    # Prevent Mesa from including X11 headers
    list(APPEND EXTRA_CFLAGS "-DMESA_EGL_NO_X11_HEADERS -DEGL_NO_X11")
  endif()
  cmake_pop_check_state()
endmacro()

macro(WaylandProtocolGen _SCANNER _CODE_MODE _XML _PROTL)
    set(_WAYLAND_PROT_C_CODE "${CMAKE_CURRENT_BINARY_DIR}/wayland-generated-protocols/${_PROTL}-protocol.c")
    set(_WAYLAND_PROT_H_CODE "${CMAKE_CURRENT_BINARY_DIR}/wayland-generated-protocols/${_PROTL}-client-protocol.h")

    add_custom_command(
        OUTPUT "${_WAYLAND_PROT_H_CODE}"
        DEPENDS "${_XML}"
        COMMAND "${_SCANNER}"
        ARGS client-header "${_XML}" "${_WAYLAND_PROT_H_CODE}"
    )

    add_custom_command(
        OUTPUT "${_WAYLAND_PROT_C_CODE}"
        DEPENDS "${_WAYLAND_PROT_H_CODE}"
        COMMAND "${_SCANNER}"
        ARGS "${_CODE_MODE}" "${_XML}" "${_WAYLAND_PROT_C_CODE}"
    )

    list(APPEND SDL_GENERATED_HEADERS "${_WAYLAND_PROT_H_CODE}")
    list(APPEND SOURCE_FILES "${_WAYLAND_PROT_C_CODE}")
endmacro()

# Requires:
# - EGL
# - PkgCheckModules
# Optional:
# - SDL_WAYLAND_SHARED opt
# - HAVE_SDL_LOADSO opt
macro(CheckWayland)
  if(SDL_WAYLAND)
    set(WAYLAND_FOUND FALSE)
    pkg_check_modules(PKG_WAYLAND "wayland-client>=1.18" wayland-egl wayland-cursor egl "xkbcommon>=0.5.0")

    if(PKG_WAYLAND_FOUND)
      set(WAYLAND_FOUND TRUE)
      find_program(WAYLAND_SCANNER NAMES wayland-scanner REQUIRED)
      execute_process(
        COMMAND ${WAYLAND_SCANNER} --version
        RESULT_VARIABLE WAYLAND_SCANNER_VERSION_RC
        ERROR_VARIABLE WAYLAND_SCANNER_VERSION
        ERROR_STRIP_TRAILING_WHITESPACE
      )
      if(NOT WAYLAND_SCANNER_VERSION_RC EQUAL 0)
        message(FATAL "Failed to get wayland-scanner version")
        set(WAYLAND_FOUND FALSE)
      endif()
      string(REPLACE "wayland-scanner " "" WAYLAND_SCANNER_VERSION ${WAYLAND_SCANNER_VERSION})

      string(COMPARE LESS ${WAYLAND_SCANNER_VERSION} "1.15.0" WAYLAND_SCANNER_PRE_1_15)
      if(WAYLAND_SCANNER_PRE_1_15)
        set(WAYLAND_SCANNER_CODE_MODE "code")
      else()
        set(WAYLAND_SCANNER_CODE_MODE "private-code")
      endif()
    endif()

    if(WAYLAND_FOUND)
      target_link_directories(sdl-build-options INTERFACE "${PKG_WAYLAND_LIBRARY_DIRS}")
      target_include_directories(sdl-build-options SYSTEM INTERFACE "${PKG_WAYLAND_INCLUDE_DIRS}")

      set(HAVE_WAYLAND TRUE)
      set(HAVE_SDL_VIDEO TRUE)

      file(GLOB WAYLAND_SOURCES ${SDL2_SOURCE_DIR}/src/video/wayland/*.c)
      list(APPEND SOURCE_FILES ${WAYLAND_SOURCES})

      # We have to generate some protocol interface code for some unstable Wayland features.
      file(MAKE_DIRECTORY "${CMAKE_CURRENT_BINARY_DIR}/wayland-generated-protocols")
      target_include_directories(sdl-build-options SYSTEM INTERFACE "${CMAKE_CURRENT_BINARY_DIR}/wayland-generated-protocols")

      file(GLOB WAYLAND_PROTOCOLS_XML RELATIVE "${SDL2_SOURCE_DIR}/wayland-protocols/" "${SDL2_SOURCE_DIR}/wayland-protocols/*.xml")
      foreach(_XML ${WAYLAND_PROTOCOLS_XML})
        string(REGEX REPLACE "\\.xml$" "" _PROTL "${_XML}")
        WaylandProtocolGen("${WAYLAND_SCANNER}" "${WAYLAND_SCANNER_CODE_MODE}" "${SDL2_SOURCE_DIR}/wayland-protocols/${_XML}" "${_PROTL}")
      endforeach()

      if(SDL_WAYLAND_QT_TOUCH)
          set(HAVE_WAYLAND_QT_TOUCH TRUE)
          set(SDL_VIDEO_DRIVER_WAYLAND_QT_TOUCH 1)
      endif()

      if(SDL_WAYLAND_SHARED AND NOT HAVE_SDL_LOADSO)
        message_warn("You must have SDL_LoadObject() support for dynamic Wayland loading")
      endif()
      FindLibraryAndSONAME(wayland-client LIBDIRS ${PKG_WAYLAND_LIBRARY_DIRS})
      FindLibraryAndSONAME(wayland-egl LIBDIRS ${PKG_WAYLAND_LIBRARY_DIRS})
      FindLibraryAndSONAME(wayland-cursor LIBDIRS ${PKG_WAYLAND_LIBRARY_DIRS})
      FindLibraryAndSONAME(xkbcommon LIBDIRS ${PKG_WAYLAND_LIBRARY_DIRS})
      if(SDL_WAYLAND_SHARED AND WAYLAND_CLIENT_LIB AND WAYLAND_EGL_LIB AND WAYLAND_CURSOR_LIB AND XKBCOMMON_LIB AND HAVE_SDL_LOADSO)
        set(SDL_VIDEO_DRIVER_WAYLAND_DYNAMIC "\"${WAYLAND_CLIENT_LIB_SONAME}\"")
        set(SDL_VIDEO_DRIVER_WAYLAND_DYNAMIC_EGL "\"${WAYLAND_EGL_LIB_SONAME}\"")
        set(SDL_VIDEO_DRIVER_WAYLAND_DYNAMIC_CURSOR "\"${WAYLAND_CURSOR_LIB_SONAME}\"")
        set(SDL_VIDEO_DRIVER_WAYLAND_DYNAMIC_XKBCOMMON "\"${XKBCOMMON_LIB_SONAME}\"")
        set(HAVE_WAYLAND_SHARED TRUE)
      else()
        list(APPEND EXTRA_LIBS ${PKG_WAYLAND_LIBRARIES})
      endif()

      if(SDL_WAYLAND_LIBDECOR)
        pkg_check_modules(PKG_LIBDECOR libdecor-0)
        if(PKG_LIBDECOR_FOUND)
            set(HAVE_WAYLAND_LIBDECOR TRUE)
            set(HAVE_LIBDECOR_H 1)
            target_link_directories(sdl-build-options INTERFACE "${PKG_LIBDECOR_LIBRARY_DIRS}")
            target_include_directories(sdl-build-options SYSTEM INTERFACE "${PKG_LIBDECOR_INCLUDE_DIRS}")
            if(SDL_WAYLAND_LIBDECOR_SHARED AND NOT HAVE_SDL_LOADSO)
                message_warn("You must have SDL_LoadObject() support for dynamic libdecor loading")
            endif()
            FindLibraryAndSONAME(decor-0 LIBDIRS ${PKG_LIBDECOR_LIBRARY_DIRS})
            if(SDL_WAYLAND_LIBDECOR_SHARED AND DECOR_0_LIB AND HAVE_SDL_LOADSO)
                set(HAVE_WAYLAND_LIBDECOR_SHARED TRUE)
                set(SDL_VIDEO_DRIVER_WAYLAND_DYNAMIC_LIBDECOR "\"${DECOR_0_LIB_SONAME}\"")
            else()
              list(APPEND EXTRA_LIBS ${PKG_LIBDECOR_LIBRARIES})
            endif()

            cmake_push_check_state()
            list(APPEND CMAKE_REQUIRED_FLAGS ${PKG_LIBDECOR_CFLAGS})
            list(APPEND CMAKE_REQUIRED_INCLUDES ${PKG_LIBDECOR_INCLUDE_DIRS})
            list(APPEND CMAKE_REQUIRED_LIBRARIES ${PKG_LIBDECOR_LINK_LIBRARIES})
            check_symbol_exists(libdecor_frame_get_max_content_size "libdecor.h" HAVE_LIBDECOR_FRAME_GET_MAX_CONTENT_SIZE)
            check_symbol_exists(libdecor_frame_get_min_content_size "libdecor.h" HAVE_LIBDECOR_FRAME_GET_MIN_CONTENT_SIZE)
            if(HAVE_LIBDECOR_FRAME_GET_MAX_CONTENT_SIZE AND HAVE_LIBDECOR_FRAME_GET_MIN_CONTENT_SIZE)
              set(SDL_HAVE_LIBDECOR_GET_MIN_MAX 1)
            endif()
            cmake_pop_check_state()
        endif()
      endif()

      set(SDL_VIDEO_DRIVER_WAYLAND 1)
    endif()
  endif()
endmacro()

# Requires:
# - n/a
#
macro(CheckCOCOA)
  if(SDL_COCOA)
    if(APPLE) # Apple always has Cocoa.
      set(HAVE_COCOA TRUE)
    endif()
    if(HAVE_COCOA)
      file(GLOB COCOA_SOURCES ${SDL2_SOURCE_DIR}/src/video/cocoa/*.m)
      list(APPEND SOURCE_FILES ${COCOA_SOURCES})
      set(SDL_VIDEO_DRIVER_COCOA 1)
      set(HAVE_SDL_VIDEO TRUE)
    endif()
  endif()
endmacro()

# Requires:
# - PkgCheckModules
# Optional:
# - DIRECTFB_SHARED opt
# - HAVE_SDL_LOADSO opt
macro(CheckDirectFB)
  if(SDL_DIRECTFB)
    pkg_check_modules(PKG_DIRECTFB directfb>=1.0.0)
    if(PKG_DIRECTFB_FOUND)
      set(HAVE_DIRECTFB TRUE)
      file(GLOB DIRECTFB_SOURCES ${SDL2_SOURCE_DIR}/src/video/directfb/*.c)
      list(APPEND SOURCE_FILES ${DIRECTFB_SOURCES})
      set(SDL_VIDEO_DRIVER_DIRECTFB 1)
      set(SDL_VIDEO_RENDER_DIRECTFB 1)
      list(APPEND EXTRA_CFLAGS ${PKG_DIRECTFB_CFLAGS})
      list(APPEND SDL_CFLAGS ${PKG_DIRECTFB_CFLAGS})
      if(SDL_DIRECTFB_SHARED AND NOT HAVE_SDL_LOADSO)
        message_warn("You must have SDL_LoadObject() support for dynamic DirectFB loading")
      endif()
      FindLibraryAndSONAME("directfb" LIBDIRS ${PKG_DIRECTFB_LIBRARY_DIRS})
      if(SDL_DIRECTFB_SHARED AND DIRECTFB_LIB AND HAVE_SDL_LOADSO)
        set(SDL_VIDEO_DRIVER_DIRECTFB_DYNAMIC "\"${DIRECTFB_LIB_SONAME}\"")
        set(HAVE_DIRECTFB_SHARED TRUE)
      else()
        list(APPEND EXTRA_LDFLAGS ${PKG_DIRECTFB_LDFLAGS})
      endif()
      set(HAVE_SDL_VIDEO TRUE)
    endif()
  endif()
endmacro()

# Requires:
# - n/a
macro(CheckVivante)
  if(SDL_VIVANTE)
    check_c_source_compiles("
        #include <gc_vdk.h>
        int main(int argc, char** argv) { return 0; }" HAVE_VIVANTE_VDK)
    check_c_source_compiles("
        #define LINUX
        #define EGL_API_FB
        #include <EGL/eglvivante.h>
        int main(int argc, char** argv) { return 0; }" HAVE_VIVANTE_EGL_FB)
    if(HAVE_VIVANTE_VDK OR HAVE_VIVANTE_EGL_FB)
      set(HAVE_VIVANTE TRUE)
      set(HAVE_SDL_VIDEO TRUE)

      file(GLOB VIVANTE_SOURCES ${SDL2_SOURCE_DIR}/src/video/vivante/*.c)
      list(APPEND SOURCE_FILES ${VIVANTE_SOURCES})
      set(SDL_VIDEO_DRIVER_VIVANTE 1)
      if(HAVE_VIVANTE_VDK)
        set(SDL_VIDEO_DRIVER_VIVANTE_VDK 1)
        find_library(VIVANTE_LIBRARY REQUIRED NAMES VIVANTE vivante drm_vivante)
        find_library(VIVANTE_VDK_LIBRARY VDK REQUIRED)
        list(APPEND EXTRA_LIBS ${VIVANTE_LIBRARY} ${VIVANTE_VDK_LIBRARY})
      else()
        list(APPEND SDL_CFLAGS -DLINUX -DEGL_API_FB)
        list(APPEND EXTRA_LIBS EGL)
      endif(HAVE_VIVANTE_VDK)
    endif()
  endif()
endmacro()

# Requires:
# - nada
macro(CheckGLX)
  if(SDL_OPENGL)
    check_c_source_compiles("
        #include <GL/glx.h>
        int main(int argc, char** argv) { return 0; }" HAVE_OPENGL_GLX)
    if(HAVE_OPENGL_GLX)
      set(SDL_VIDEO_OPENGL_GLX 1)
    endif()
  endif()
endmacro()

# Requires:
# - PkgCheckModules
macro(CheckEGL)
  if (SDL_OPENGL OR SDL_OPENGLES)
    pkg_check_modules(EGL egl)
    set(CMAKE_REQUIRED_DEFINITIONS "${CMAKE_REQUIRED_DEFINITIONS} ${EGL_CFLAGS}")
    check_c_source_compiles("
        #define EGL_API_FB
        #define MESA_EGL_NO_X11_HEADERS
        #define EGL_NO_X11
        #include <EGL/egl.h>
        #include <EGL/eglext.h>
        int main (int argc, char** argv) { return 0; }" HAVE_OPENGL_EGL)
    if(HAVE_OPENGL_EGL)
      set(SDL_VIDEO_OPENGL_EGL 1)
    endif()
  endif()
endmacro()

# Requires:
# - nada
macro(CheckOpenGL)
  if(SDL_OPENGL)
    check_c_source_compiles("
        #include <GL/gl.h>
        #include <GL/glext.h>
        int main(int argc, char** argv) { return 0; }" HAVE_OPENGL)
    if(HAVE_OPENGL)
      set(SDL_VIDEO_OPENGL 1)
      set(SDL_VIDEO_RENDER_OGL 1)
    endif()
  endif()
endmacro()

# Requires:
# - nada
macro(CheckOpenGLES)
  if(SDL_OPENGLES)
    check_c_source_compiles("
        #include <GLES/gl.h>
        #include <GLES/glext.h>
        int main (int argc, char** argv) { return 0; }" HAVE_OPENGLES_V1)
    if(HAVE_OPENGLES_V1)
        set(HAVE_OPENGLES TRUE)
        set(SDL_VIDEO_OPENGL_ES 1)
        set(SDL_VIDEO_RENDER_OGL_ES 1)
    endif()
    check_c_source_compiles("
        #include <GLES2/gl2.h>
        #include <GLES2/gl2ext.h>
        int main (int argc, char** argv) { return 0; }" HAVE_OPENGLES_V2)
    if(HAVE_OPENGLES_V2)
        set(HAVE_OPENGLES TRUE)
        set(SDL_VIDEO_OPENGL_ES2 1)
        set(SDL_VIDEO_RENDER_OGL_ES2 1)
    endif()
  endif()
endmacro()

# Requires:
# - EGL
macro(CheckQNXScreen)
  if(QNX AND HAVE_OPENGL_EGL)
    check_c_source_compiles("
        #include <screen/screen.h>
        int main (int argc, char** argv) { return 0; }" HAVE_QNX_SCREEN)
    if(HAVE_QNX_SCREEN)
      set(SDL_VIDEO_DRIVER_QNX 1)
      file(GLOB QNX_VIDEO_SOURCES ${SDL2_SOURCE_DIR}/src/video/qnx/*.c)
      list(APPEND SOURCE_FILES ${QNX_VIDEO_SOURCES})
      list(APPEND EXTRA_LIBS screen EGL)
    endif()
  endif()
endmacro()

# Requires:
# - nada
# Optional:
# - THREADS opt
# Sets:
# PTHREAD_CFLAGS
# PTHREAD_LIBS
macro(CheckPTHREAD)
  if(SDL_THREADS AND SDL_PTHREADS)
    if(ANDROID)
      # the android libc provides built-in support for pthreads, so no
      # additional linking or compile flags are necessary
    elseif(LINUX)
      set(PTHREAD_CFLAGS "-D_REENTRANT")
      set(PTHREAD_LDFLAGS "-pthread")
    elseif(BSDI)
      set(PTHREAD_CFLAGS "-D_REENTRANT -D_THREAD_SAFE")
      set(PTHREAD_LDFLAGS "")
    elseif(DARWIN)
      set(PTHREAD_CFLAGS "-D_THREAD_SAFE")
      # causes Carbon.p complaints?
      # set(PTHREAD_CFLAGS "-D_REENTRANT -D_THREAD_SAFE")
      set(PTHREAD_LDFLAGS "")
    elseif(FREEBSD)
      set(PTHREAD_CFLAGS "-D_REENTRANT -D_THREAD_SAFE")
      set(PTHREAD_LDFLAGS "-pthread")
    elseif(NETBSD)
      set(PTHREAD_CFLAGS "-D_REENTRANT -D_THREAD_SAFE")
      set(PTHREAD_LDFLAGS "-lpthread")
    elseif(OPENBSD)
      set(PTHREAD_CFLAGS "-D_REENTRANT")
      set(PTHREAD_LDFLAGS "-lpthread")
    elseif(SOLARIS)
      set(PTHREAD_CFLAGS "-D_REENTRANT")
      set(PTHREAD_LDFLAGS "-pthread -lposix4")
    elseif(SYSV5)
      set(PTHREAD_CFLAGS "-D_REENTRANT -Kthread")
      set(PTHREAD_LDFLAGS "")
    elseif(AIX)
      set(PTHREAD_CFLAGS "-D_REENTRANT -mthreads")
      set(PTHREAD_LDFLAGS "-pthread")
    elseif(HPUX)
      set(PTHREAD_CFLAGS "-D_REENTRANT")
      set(PTHREAD_LDFLAGS "-L/usr/lib -pthread")
    elseif(HAIKU)
      set(PTHREAD_CFLAGS "-D_REENTRANT")
      set(PTHREAD_LDFLAGS "")
    elseif(EMSCRIPTEN)
      set(PTHREAD_CFLAGS "-D_REENTRANT -pthread")
      set(PTHREAD_LDFLAGS "-pthread")
    elseif(QNX)
      # pthread support is baked in
    else()
      set(PTHREAD_CFLAGS "-D_REENTRANT")
      set(PTHREAD_LDFLAGS "-lpthread")
    endif()

    # Run some tests
    set(ORIG_CMAKE_REQUIRED_FLAGS "${CMAKE_REQUIRED_FLAGS}")
    set(CMAKE_REQUIRED_FLAGS "${CMAKE_REQUIRED_FLAGS} ${PTHREAD_CFLAGS} ${PTHREAD_LDFLAGS}")
    check_c_source_compiles("
      #include <pthread.h>
      int main(int argc, char** argv) {
        pthread_attr_t type;
        pthread_attr_init(&type);
        return 0;
      }" HAVE_PTHREADS)
    if(HAVE_PTHREADS)
      set(SDL_THREAD_PTHREAD 1)
      list(APPEND EXTRA_CFLAGS ${PTHREAD_CFLAGS})
      list(APPEND EXTRA_LDFLAGS ${PTHREAD_LDFLAGS})
      list(APPEND SDL_CFLAGS ${PTHREAD_CFLAGS})

      check_c_source_compiles("
        #include <pthread.h>
        int main(int argc, char **argv) {
          pthread_mutexattr_t attr;
          pthread_mutexattr_settype(&attr, PTHREAD_MUTEX_RECURSIVE);
          return 0;
        }" HAVE_RECURSIVE_MUTEXES)
      if(HAVE_RECURSIVE_MUTEXES)
        set(SDL_THREAD_PTHREAD_RECURSIVE_MUTEX 1)
      else()
        check_c_source_compiles("
            #include <pthread.h>
            int main(int argc, char **argv) {
              pthread_mutexattr_t attr;
              pthread_mutexattr_settype(&attr, PTHREAD_MUTEX_RECURSIVE_NP);
              return 0;
            }" HAVE_RECURSIVE_MUTEXES_NP)
        if(HAVE_RECURSIVE_MUTEXES_NP)
          set(SDL_THREAD_PTHREAD_RECURSIVE_MUTEX_NP 1)
        endif()
      endif()

      if(SDL_PTHREADS_SEM)
        check_c_source_compiles("#include <pthread.h>
                                 #include <semaphore.h>
                                 int main(int argc, char **argv) { return 0; }" HAVE_PTHREADS_SEM)
        if(HAVE_PTHREADS_SEM)
          check_c_source_compiles("
              #include <pthread.h>
              #include <semaphore.h>
              int main(int argc, char **argv) {
                  sem_timedwait(NULL, NULL);
                  return 0;
              }" HAVE_SEM_TIMEDWAIT)
        endif()
      endif()

      check_include_files("pthread.h" HAVE_PTHREAD_H)
      check_include_files("pthread_np.h" HAVE_PTHREAD_NP_H)
      if (HAVE_PTHREAD_H)
        check_c_source_compiles("
            #include <pthread.h>
            int main(int argc, char **argv) {
              #ifdef __APPLE__
              pthread_setname_np(\"\");
              #else
              pthread_setname_np(pthread_self(),\"\");
              #endif
              return 0;
            }" HAVE_PTHREAD_SETNAME_NP)
        if (HAVE_PTHREAD_NP_H)
          check_symbol_exists(pthread_set_name_np "pthread.h;pthread_np.h" HAVE_PTHREAD_SET_NAME_NP)
        endif()
      endif()

      set(SOURCE_FILES ${SOURCE_FILES}
          ${SDL2_SOURCE_DIR}/src/thread/pthread/SDL_systhread.c
          ${SDL2_SOURCE_DIR}/src/thread/pthread/SDL_sysmutex.c   # Can be faked, if necessary
          ${SDL2_SOURCE_DIR}/src/thread/pthread/SDL_syscond.c    # Can be faked, if necessary
          ${SDL2_SOURCE_DIR}/src/thread/pthread/SDL_systls.c
          )
      if(HAVE_PTHREADS_SEM)
        set(SOURCE_FILES ${SOURCE_FILES}
            ${SDL2_SOURCE_DIR}/src/thread/pthread/SDL_syssem.c)
      else()
        set(SOURCE_FILES ${SOURCE_FILES}
            ${SDL2_SOURCE_DIR}/src/thread/generic/SDL_syssem.c)
      endif()
      set(HAVE_SDL_THREADS TRUE)
    endif()
    set(CMAKE_REQUIRED_FLAGS "${ORIG_CMAKE_REQUIRED_FLAGS}")
  endif()
endmacro()

# Requires
# - nada
# Optional:
# Sets:
# USB_LIBS
# USB_CFLAGS
macro(CheckUSBHID)
  check_library_exists(usbhid hid_init "" LIBUSBHID)
  if(LIBUSBHID)
    check_include_file(usbhid.h HAVE_USBHID_H)
    if(HAVE_USBHID_H)
      set(USB_CFLAGS "-DHAVE_USBHID_H")
    endif()

    check_include_file(libusbhid.h HAVE_LIBUSBHID_H)
    if(HAVE_LIBUSBHID_H)
      set(USB_CFLAGS "${USB_CFLAGS} -DHAVE_LIBUSBHID_H")
    endif()
    set(USB_LIBS ${USB_LIBS} usbhid)
  else()
    check_include_file(usb.h HAVE_USB_H)
    if(HAVE_USB_H)
      set(USB_CFLAGS "-DHAVE_USB_H")
    endif()
    check_include_file(libusb.h HAVE_LIBUSB_H)
    if(HAVE_LIBUSB_H)
      set(USB_CFLAGS "${USB_CFLAGS} -DHAVE_LIBUSB_H")
    endif()
    check_library_exists(usb hid_init "" LIBUSB)
    if(LIBUSB)
      set(USB_LIBS ${USB_LIBS} usb)
    endif()
  endif()

  set(ORIG_CMAKE_REQUIRED_FLAGS "${CMAKE_REQUIRED_FLAGS}")
  set(CMAKE_REQUIRED_FLAGS "${CMAKE_REQUIRED_FLAGS} ${USB_CFLAGS}")
  set(CMAKE_REQUIRED_LIBRARIES "${USB_LIBS}")
  check_c_source_compiles("
       #include <sys/types.h>
        #if defined(HAVE_USB_H)
        #include <usb.h>
        #endif
        #ifdef __DragonFly__
        # include <bus/u4b/usb.h>
        # include <bus/u4b/usbhid.h>
        #else
        # include <dev/usb/usb.h>
        # include <dev/usb/usbhid.h>
        #endif
        #if defined(HAVE_USBHID_H)
        #include <usbhid.h>
        #elif defined(HAVE_LIBUSB_H)
        #include <libusb.h>
        #elif defined(HAVE_LIBUSBHID_H)
        #include <libusbhid.h>
        #endif
        int main(int argc, char **argv) {
          struct report_desc *repdesc;
          struct usb_ctl_report *repbuf;
          hid_kind_t hidkind;
          return 0;
        }" HAVE_USBHID)
  if(HAVE_USBHID)
    check_c_source_compiles("
          #include <sys/types.h>
          #if defined(HAVE_USB_H)
          #include <usb.h>
          #endif
          #ifdef __DragonFly__
          # include <bus/u4b/usb.h>
          # include <bus/u4b/usbhid.h>
          #else
          # include <dev/usb/usb.h>
          # include <dev/usb/usbhid.h>
          #endif
          #if defined(HAVE_USBHID_H)
          #include <usbhid.h>
          #elif defined(HAVE_LIBUSB_H)
          #include <libusb.h>
          #elif defined(HAVE_LIBUSBHID_H)
          #include <libusbhid.h>
          #endif
          int main(int argc, char** argv) {
            struct usb_ctl_report buf;
            if (buf.ucr_data) { }
            return 0;
          }" HAVE_USBHID_UCR_DATA)
    if(HAVE_USBHID_UCR_DATA)
      set(USB_CFLAGS "${USB_CFLAGS} -DUSBHID_UCR_DATA")
    endif()

    check_c_source_compiles("
          #include <sys/types.h>
          #if defined(HAVE_USB_H)
          #include <usb.h>
          #endif
          #ifdef __DragonFly__
          #include <bus/u4b/usb.h>
          #include <bus/u4b/usbhid.h>
          #else
          #include <dev/usb/usb.h>
          #include <dev/usb/usbhid.h>
          #endif
          #if defined(HAVE_USBHID_H)
          #include <usbhid.h>
          #elif defined(HAVE_LIBUSB_H)
          #include <libusb.h>
          #elif defined(HAVE_LIBUSBHID_H)
          #include <libusbhid.h>
          #endif
          int main(int argc, char **argv) {
            report_desc_t d;
            hid_start_parse(d, 1, 1);
            return 0;
          }" HAVE_USBHID_NEW)
    if(HAVE_USBHID_NEW)
      set(USB_CFLAGS "${USB_CFLAGS} -DUSBHID_NEW")
    endif()

    check_c_source_compiles("
        #include <machine/joystick.h>
        int main(int argc, char** argv) {
            struct joystick t;
            return 0;
        }" HAVE_MACHINE_JOYSTICK)
    if(HAVE_MACHINE_JOYSTICK)
      set(SDL_HAVE_MACHINE_JOYSTICK_H 1)
    endif()
    set(SDL_JOYSTICK_USBHID 1)
    file(GLOB BSD_JOYSTICK_SOURCES ${SDL2_SOURCE_DIR}/src/joystick/bsd/*.c)
    list(APPEND SOURCE_FILES ${BSD_JOYSTICK_SOURCES})
    list(APPEND EXTRA_CFLAGS ${USB_CFLAGS})
    list(APPEND EXTRA_LIBS ${USB_LIBS})
    set(HAVE_SDL_JOYSTICK TRUE)

    set(CMAKE_REQUIRED_LIBRARIES)
    set(CMAKE_REQUIRED_FLAGS "${ORIG_CMAKE_REQUIRED_FLAGS}")
  endif()
endmacro()

# Check for HIDAPI support
macro(CheckHIDAPI)
  set(HAVE_HIDAPI TRUE)
  if(SDL_HIDAPI)
    if(SDL_HIDAPI_LIBUSB)
      set(HAVE_LIBUSB FALSE)
      pkg_check_modules(PKG_LIBUSB libusb-1.0)
      if(PKG_LIBUSB_FOUND)
        check_include_file(libusb.h HAVE_LIBUSB_H ${PKG_LIBUSB_CFLAGS})
        if(HAVE_LIBUSB_H)
          set(HAVE_LIBUSB TRUE)
          set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${PKG_LIBUSB_CFLAGS}")
          if(HIDAPI_ONLY_LIBUSB)
            list(APPEND EXTRA_LIBS ${PKG_LIBUSB_LIBRARIES})
          elseif(OS2)
            set(SDL_LIBUSB_DYNAMIC "\"usb100.dll\"")
          else()
            # libusb is loaded dynamically, so don't add it to EXTRA_LIBS
            FindLibraryAndSONAME("usb-1.0" LIBDIRS ${PKG_LIBUSB_LIBRARY_DIRS})
            if(USB_1.0_LIB)
              set(SDL_LIBUSB_DYNAMIC "\"${USB_1.0_LIB_SONAME}\"")
            endif()
          endif()
        endif()
      endif()
      if(HIDAPI_ONLY_LIBUSB AND NOT HAVE_LIBUSB)
        set(HAVE_HIDAPI FALSE)
      endif()
      set(HAVE_HIDAPI_LIBUSB ${HAVE_LIBUSB})
    endif()

    if(HAVE_HIDAPI)
      if(ANDROID)
        list(APPEND SOURCE_FILES ${SDL2_SOURCE_DIR}/src/hidapi/android/hid.cpp)
      endif()
      if(IOS OR TVOS)
        list(APPEND SOURCE_FILES ${SDL2_SOURCE_DIR}/src/hidapi/ios/hid.m)
        set(SDL_FRAMEWORK_COREBLUETOOTH 1)
      endif()
      set(HAVE_SDL_HIDAPI TRUE)

      if(SDL_JOYSTICK AND SDL_HIDAPI_JOYSTICK)
        set(SDL_JOYSTICK_HIDAPI 1)
        set(HAVE_SDL_JOYSTICK TRUE)
        set(HAVE_HIDAPI_JOYSTICK TRUE)
        file(GLOB HIDAPI_JOYSTICK_SOURCES ${SDL2_SOURCE_DIR}/src/joystick/hidapi/*.c)
        list(APPEND SOURCE_FILES ${HIDAPI_JOYSTICK_SOURCES})
      endif()
    else()
      set(SDL_HIDAPI_DISABLED 1)
    endif()
  else()
    set(SDL_HIDAPI_DISABLED 1)
  endif()
endmacro()

# Requires:
# - n/a
macro(CheckRPI)
  if(SDL_RPI)
    pkg_check_modules(VIDEO_RPI bcm_host brcmegl)
    if (NOT VIDEO_RPI_FOUND)
      set(VIDEO_RPI_INCLUDE_DIRS "/opt/vc/include" "/opt/vc/include/interface/vcos/pthreads" "/opt/vc/include/interface/vmcs_host/linux/" )
      set(VIDEO_RPI_LIBRARY_DIRS "/opt/vc/lib" )
      set(VIDEO_RPI_LIBRARIES bcm_host )
      set(VIDEO_RPI_LDFLAGS "-Wl,-rpath,/opt/vc/lib")
    endif()
    listtostr(VIDEO_RPI_INCLUDE_DIRS VIDEO_RPI_INCLUDE_FLAGS "-I")
    listtostr(VIDEO_RPI_LIBRARY_DIRS VIDEO_RPI_LIBRARY_FLAGS "-L")

    set(ORIG_CMAKE_REQUIRED_FLAGS "${CMAKE_REQUIRED_FLAGS}")
    set(CMAKE_REQUIRED_FLAGS "${CMAKE_REQUIRED_FLAGS} ${VIDEO_RPI_INCLUDE_FLAGS} ${VIDEO_RPI_LIBRARY_FLAGS}")
    set(CMAKE_REQUIRED_LIBRARIES "${VIDEO_RPI_LIBRARIES}")
    check_c_source_compiles("
        #include <bcm_host.h>
        #include <EGL/eglplatform.h>
        int main(int argc, char **argv) {
          EGL_DISPMANX_WINDOW_T window;
          bcm_host_init();
        }" HAVE_RPI)
    set(CMAKE_REQUIRED_FLAGS "${ORIG_CMAKE_REQUIRED_FLAGS}")
    set(CMAKE_REQUIRED_LIBRARIES)

    if(SDL_VIDEO AND HAVE_RPI)
      set(HAVE_SDL_VIDEO TRUE)
      set(SDL_VIDEO_DRIVER_RPI 1)
      file(GLOB VIDEO_RPI_SOURCES ${SDL2_SOURCE_DIR}/src/video/raspberry/*.c)
      list(APPEND SOURCE_FILES ${VIDEO_RPI_SOURCES})
      list(APPEND EXTRA_LIBS ${VIDEO_RPI_LIBRARIES})
      # !!! FIXME: shouldn't be using CMAKE_C_FLAGS, right?
      set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${VIDEO_RPI_INCLUDE_FLAGS} ${VIDEO_RPI_LIBRARY_FLAGS}")
      list(APPEND EXTRA_LDFLAGS ${VIDEO_RPI_LDFLAGS})
    endif()
  endif()
endmacro()

# Requires:
# - EGL
# - PkgCheckModules
# Optional:
# - SDL_KMSDRM_SHARED opt
# - HAVE_SDL_LOADSO opt
macro(CheckKMSDRM)
  if(SDL_KMSDRM)
    pkg_check_modules(PKG_KMSDRM libdrm gbm egl)
    if(PKG_KMSDRM_FOUND AND HAVE_OPENGL_EGL)
      target_link_directories(sdl-build-options INTERFACE ${PKG_KMSDRM_LIBRARY_DIRS})
      target_include_directories(sdl-build-options SYSTEM INTERFACE "${PKG_KMSDRM_INCLUDE_DIRS}")
      set(HAVE_KMSDRM TRUE)
      set(HAVE_SDL_VIDEO TRUE)

      file(GLOB KMSDRM_SOURCES ${SDL2_SOURCE_DIR}/src/video/kmsdrm/*.c)
      list(APPEND SOURCE_FILES ${KMSDRM_SOURCES})

      list(APPEND EXTRA_CFLAGS ${PKG_KMSDRM_CFLAGS})

      set(SDL_VIDEO_DRIVER_KMSDRM 1)

      if(SDL_KMSDRM_SHARED AND NOT HAVE_SDL_LOADSO)
        message_warn("You must have SDL_LoadObject() support for dynamic KMS/DRM loading")
      endif()
      if(SDL_KMSDRM_SHARED AND HAVE_SDL_LOADSO)
        FindLibraryAndSONAME(drm LIBDIRS ${PKG_KMSDRM_LIBRARY_DIRS})
        FindLibraryAndSONAME(gbm LIBDIRS ${PKG_KMSDRM_LIBRARY_DIRS})
        set(SDL_VIDEO_DRIVER_KMSDRM_DYNAMIC "\"${DRM_LIB_SONAME}\"")
        set(SDL_VIDEO_DRIVER_KMSDRM_DYNAMIC_GBM "\"${GBM_LIB_SONAME}\"")
        set(HAVE_KMSDRM_SHARED TRUE)
      else()
        list(APPEND EXTRA_LIBS ${PKG_KMSDRM_LIBRARIES})
      endif()
    endif()
  endif()
endmacro()

macro(CheckLibUDev)
  if(SDL_LIBUDEV)
    check_include_file("libudev.h" HAVE_LIBUDEV_HEADER)
    if(HAVE_LIBUDEV_HEADER)
      set(HAVE_LIBUDEV_H TRUE)
      FindLibraryAndSONAME(udev)
      if(UDEV_LIB_SONAME)
        set(SDL_UDEV_DYNAMIC "\"${UDEV_LIB_SONAME}\"")
        set(HAVE_LIBUDEV TRUE)
      endif()
    endif()
  endif()
endmacro()


macro(CheckLibUnwind)
  set(found_libunwind FALSE)
  set(_libunwind_src "#include <libunwind.h>\nint main() {unw_context_t context; unw_getcontext(&context); return 0;}")

  if(NOT found_libunwind)
    cmake_push_check_state()
    check_c_source_compiles("${_libunwind_src}" LIBC_HAS_WORKING_LIBUNWIND)
    cmake_pop_check_state()
    if(LIBC_HAS_WORKING_LIBUNWIND)
      set(found_libunwind TRUE)
    endif()
  endif()

  if(NOT found_libunwind)
    cmake_push_check_state()
    list(APPEND CMAKE_REQUIRED_LIBRARIES "unwind")
    check_c_source_compiles("${_libunwind_src}" LIBUNWIND_HAS_WORKINGLIBUNWIND)
    cmake_pop_check_state()
    if(LIBUNWIND_HAS_WORKINGLIBUNWIND)
      set(found_libunwind TRUE)
      list(APPEND EXTRA_TEST_LIBS unwind)
    endif()
  endif()

  if(NOT found_libunwind)
    set(LibUnwind_PKG_CONFIG_SPEC libunwind libunwind-generic)
    pkg_check_modules(PC_LIBUNWIND IMPORTED_TARGET ${LibUnwind_PKG_CONFIG_SPEC})
    if(PC_LIBUNWIND_FOUND)
      cmake_push_check_state()
      list(APPEND CMAKE_REQUIRED_LIBRARIES ${PC_LIBUNWIND_LIBRARIES})
      list(APPEND CMAKE_REQUIRED_INCLUDES ${PC_LIBUNWIND_INCLUDE_DIRS})
      check_c_source_compiles("${_libunwind_src}" PC_LIBUNWIND_HAS_WORKING_LIBUNWIND)
      cmake_pop_check_state()
      if(PC_LIBUNWIND_HAS_WORKING_LIBUNWIND)
        set(found_libunwind TRUE)
        list(APPEND EXTRA_TEST_LIBS ${PC_LIBUNWIND_LIBRARIES})
        list(APPEND EXTRA_TEST_INCLUDES ${PC_LIBUNWIND_INCLUDE_DIRS})
      endif()
    endif()
  endif()

  if(found_libunwind)
    set(HAVE_LIBUNWIND_H TRUE)
  endif()
endmacro()
