LOCAL_PATH := $(call my-dir)

include $(CLEAR_VARS)

LOCAL_MODULE := main_gui_androidmk
LOCAL_SRC_FILES := ../main_gui.c
LOCAL_SHARED_LIBRARIES += SDL2
include $(BUILD_SHARED_LIBRARY)

$(call import-module,SDL2main)
$(call import-module,SDL2)
