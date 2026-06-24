LOCAL_PATH:= $(call my-dir)

HIDAPI_ROOT_REL:= ../..
HIDAPI_ROOT_ABS:= $(LOCAL_PATH)/../..

include $(CLEAR_VARS)

LOCAL_CPPFLAGS += -std=c++11

LOCAL_SRC_FILES := \
  $(HIDAPI_ROOT_REL)/android/hid.cpp

LOCAL_MODULE := libhidapi
LOCAL_LDLIBS := -llog

include $(BUILD_SHARED_LIBRARY)
