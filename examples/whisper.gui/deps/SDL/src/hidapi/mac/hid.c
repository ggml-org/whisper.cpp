/*******************************************************
 HIDAPI - Multi-Platform library for
 communication with HID devices.
 
 Alan Ott
 Signal 11 Software
 
 2010-07-03
 
 Copyright 2010, All Rights Reserved.
 
 At the discretion of the user of this library,
 this software may be licensed under the terms of the
 GNU Public License v3, a BSD-Style license, or the
 original HIDAPI license as outlined in the LICENSE.txt,
 LICENSE-gpl3.txt, LICENSE-bsd.txt, and LICENSE-orig.txt
 files located at the root of the source distribution.
 These files may also be found in the public source
 code repository located at:
 https://github.com/libusb/hidapi .
 ********************************************************/
#include "../../SDL_internal.h"

#include "SDL_hints.h"

/* See Apple Technical Note TN2187 for details on IOHidManager. */

#include <IOKit/hid/IOHIDManager.h>
#include <IOKit/hid/IOHIDKeys.h>
#include <CoreFoundation/CoreFoundation.h>
#include <wchar.h>
#include <pthread.h>
#include <sys/time.h>
#include <unistd.h>

#include "../hidapi/hidapi.h"

#define VALVE_USB_VID		0x28DE

/* Barrier implementation because Mac OSX doesn't have pthread_barrier.
 It also doesn't have clock_gettime(). So much for POSIX and SUSv2.
 This implementation came from Brent Priddy and was posted on
 StackOverflow. It is used with his permission. */
typedef int pthread_barrierattr_t;
typedef struct pthread_barrier {
	pthread_mutex_t mutex;
	pthread_cond_t cond;
	int count;
	int trip_count;
} pthread_barrier_t;

static int pthread_barrier_init(pthread_barrier_t *barrier, const pthread_barrierattr_t *attr, unsigned int count)
{
	if(count == 0) {
		errno = EINVAL;
		return -1;
	}
	
	if(pthread_mutex_init(&barrier->mutex, 0) < 0) {
		return -1;
	}
	if(pthread_cond_init(&barrier->cond, 0) < 0) {
		pthread_mutex_destroy(&barrier->mutex);
		return -1;
	}
	barrier->trip_count = count;
	barrier->count = 0;
	
	return 0;
}

static int pthread_barrier_destroy(pthread_barrier_t *barrier)
{
	pthread_cond_destroy(&barrier->cond);
	pthread_mutex_destroy(&barrier->mutex);
	return 0;
}

static int pthread_barrier_wait(pthread_barrier_t *barrier)
{
	pthread_mutex_lock(&barrier->mutex);
	++(barrier->count);
	if(barrier->count >= barrier->trip_count)
	{
		barrier->count = 0;
		pthread_cond_broadcast(&barrier->cond);
		pthread_mutex_unlock(&barrier->mutex);
		return 1;
	}
	else
	{
		pthread_cond_wait(&barrier->cond, &(barrier->mutex));
		pthread_mutex_unlock(&barrier->mutex);
		return 0;
	}
}

static int return_data(hid_device *dev, unsigned char *data, size_t length);

/* Linked List of input reports received from the device. */
struct input_report {
	uint8_t *data;
	size_t len;
	struct input_report *next;
};

struct hid_device_ {
	IOHIDDeviceRef device_handle;
	int blocking;
	int uses_numbered_reports;
	int disconnected;
	CFStringRef run_loop_mode;
	CFRunLoopRef run_loop;
	CFRunLoopSourceRef source;
	uint8_t *input_report_buf;
	CFIndex max_input_report_len;
	struct input_report *input_reports;
	
	pthread_t thread;
	pthread_mutex_t mutex; /* Protects input_reports */
	pthread_cond_t condition;
	pthread_barrier_t barrier; /* Ensures correct startup sequence */
	pthread_barrier_t shutdown_barrier; /* Ensures correct shutdown sequence */
	int shutdown_thread;
};

struct hid_device_list_node
{
	struct hid_device_ *dev;
	struct hid_device_list_node *next;
};

static 	IOHIDManagerRef hid_mgr = 0x0;
static 	struct hid_device_list_node *device_list = 0x0;

static hid_device *new_hid_device(void)
{
	hid_device *dev = (hid_device*)calloc(1, sizeof(hid_device));
	dev->device_handle = NULL;
	dev->blocking = 1;
	dev->uses_numbered_reports = 0;
	dev->disconnected = 0;
	dev->run_loop_mode = NULL;
	dev->run_loop = NULL;
	dev->source = NULL;
	dev->input_report_buf = NULL;
	dev->input_reports = NULL;
	dev->shutdown_thread = 0;
	
	/* Thread objects */
	pthread_mutex_init(&dev->mutex, NULL);
	pthread_cond_init(&dev->condition, NULL);
	pthread_barrier_init(&dev->barrier, NULL, 2);
	pthread_barrier_init(&dev->shutdown_barrier, NULL, 2);
	
	return dev;
}

static void free_hid_device(hid_device *dev)
{
	struct input_report *rpt;
	if (!dev)
		return;
	
	/* Delete any input reports still left over. */
	rpt = dev->input_reports;
	while (rpt) {
		struct input_report *next = rpt->next;
		free(rpt->data);
		free(rpt);
		rpt = next;
	}
	
	/* Free the string and the report buffer. The check for NULL
	 is necessary here as CFRelease() doesn't handle NULL like
	 free() and others do. */
	if (dev->run_loop_mode)
		CFRelease(dev->run_loop_mode);
	if (dev->source)
		CFRelease(dev->source);
	free(dev->input_report_buf);

	if (device_list) {
		if (device_list->dev == dev) {
			device_list = device_list->next;
		}
		else {
			struct hid_device_list_node *node = device_list;
			while (node) {
				if (node->next && node->next->dev == dev) {
					struct hid_device_list_node *new_next = node->next->next;
					free(node->next);
					node->next = new_next;
					break;
				}

				node = node->next;
			}
		}
	}
	
	/* Clean up the thread objects */
	pthread_barrier_destroy(&dev->shutdown_barrier);
	pthread_barrier_destroy(&dev->barrier);
	pthread_cond_destroy(&dev->condition);
	pthread_mutex_destroy(&dev->mutex);
	
	/* Free the structure itself. */
	free(dev);
}

#if 0
static void register_error(hid_device *device, const char *op)
{
	
}
#endif


static int32_t get_int_property(IOHIDDeviceRef device, CFStringRef key)
{
	CFTypeRef ref;
	int32_t value;
	
	ref = IOHIDDeviceGetProperty(device, key);
	if (ref) {
		if (CFGetTypeID(ref) == CFNumberGetTypeID()) {
			CFNumberGetValue((CFNumberRef) ref, kCFNumberSInt32Type, &value);
			return value;
		}
	}
	return 0;
}

static unsigned short get_vendor_id(IOHIDDeviceRef device)
{
	return get_int_property(device, CFSTR(kIOHIDVendorIDKey));
}

static unsigned short get_product_id(IOHIDDeviceRef device)
{
	return get_int_property(device, CFSTR(kIOHIDProductIDKey));
}


static int32_t get_max_report_length(IOHIDDeviceRef device)
{
	return get_int_property(device, CFSTR(kIOHIDMaxInputReportSizeKey));
}

static int get_string_property(IOHIDDeviceRef device, CFStringRef prop, wchar_t *buf, size_t len)
{
	CFStringRef str;
	
	if (!len)
		return 0;

	if (CFGetTypeID(prop) != CFStringGetTypeID())
		return 0;

	str = (CFStringRef)IOHIDDeviceGetProperty(device, prop);
	
	buf[0] = 0;
	
	if (str && CFGetTypeID(str) == CFStringGetTypeID()) {
		CFIndex used_buf_len, chars_copied;
		CFRange range;
		CFIndex str_len = CFStringGetLength(str);
		len --;
		range.location = 0;
		range.length = (str_len > len)? len: str_len;
		chars_copied = CFStringGetBytes(str,
										range,
										kCFStringEncodingUTF32LE,
										(char)'?',
										FALSE,
										(UInt8*)buf,
										len,
										&used_buf_len);
		
		buf[chars_copied] = 0;
		return (int)chars_copied;
	}
	else
		return 0;
	
}

static int get_string_property_utf8(IOHIDDeviceRef device, CFStringRef prop, char *buf, size_t len)
{
	CFStringRef str;
	if (!len)
		return 0;
	
	if (CFGetTypeID(prop) != CFStringGetTypeID())
		return 0;

	str = (CFStringRef)IOHIDDeviceGetProperty(device, prop);
	
	buf[0] = 0;
	
	if (str && CFGetTypeID(str) == CFStringGetTypeID()) {
		CFIndex used_buf_len, chars_copied;
		CFRange range;
		CFIndex str_len = CFStringGetLength(str);
		len--;
		range.location = 0;
		range.length = (str_len > len)? len: str_len;
		chars_copied = CFStringGetBytes(str,
										range,
										kCFStringEncodingUTF8,
										(char)'?',
										FALSE,
										(UInt8*)buf,
										len,
										&used_buf_len);
		
		buf[chars_copied] = 0;
		return (int)used_buf_len;
	}
	else
		return 0;
}


static int get_serial_number(IOHIDDeviceRef device, wchar_t *buf, size_t len)
{
	return get_string_property(device, CFSTR(kIOHIDSerialNumberKey), buf, len);
}

static int get_manufacturer_string(IOHIDDeviceRef device, wchar_t *buf, size_t len)
{
	return get_string_property(device, CFSTR(kIOHIDManufacturerKey), buf, len);
}

static int get_product_string(IOHIDDeviceRef device, wchar_t *buf, size_t len)
{
	return get_string_property(device, CFSTR(kIOHIDProductKey), buf, len);
}


/* Implementation of wcsdup() for Mac. */
static wchar_t *dup_wcs(const wchar_t *s)
{
	size_t len = wcslen(s);
	wchar_t *ret = (wchar_t *)malloc((len+1)*sizeof(wchar_t));
	wcscpy(ret, s);
	
	return ret;
}


static int make_path(IOHIDDeviceRef device, char *buf, size_t len)
{
	int res;
	unsigned short vid, pid;
	char transport[32];
	
	buf[0] = '\0';
	
	res = get_string_property_utf8(
								   device, CFSTR(kIOHIDTransportKey),
								   transport, sizeof(transport));
	
	if (!res)
		return -1;
	
	vid = get_vendor_id(device);
	pid = get_product_id(device);
	
	res = snprintf(buf, len, "%s_%04hx_%04hx_%p",
				   transport, vid, pid, device);
	
	
	buf[len-1] = '\0';
	return res+1;
}

static void hid_device_removal_callback(void *context, IOReturn result,
                                        void *sender, IOHIDDeviceRef hid_ref)
{
	// The device removal callback is sometimes called even after being
	// unregistered, leading to a crash when trying to access fields in
	// the already freed hid_device. We keep a linked list of all created
	// hid_device's so that the one being removed can be checked against
	// the list to see if it really hasn't been closed yet and needs to
	// be dealt with here.
	struct hid_device_list_node *node = device_list;
	while (node) {
		if (node->dev->device_handle == hid_ref) {
			node->dev->disconnected = 1;
			CFRunLoopStop(node->dev->run_loop);
			break;
		}

		node = node->next;
	}
}

static CFDictionaryRef
create_usage_match(const UInt32 page, const UInt32 usage, int *okay)
{
	CFDictionaryRef retval = NULL;
	CFNumberRef pageNumRef = CFNumberCreate(kCFAllocatorDefault, kCFNumberIntType, &page);
	CFNumberRef usageNumRef = CFNumberCreate(kCFAllocatorDefault, kCFNumberIntType, &usage);
	const void *keys[2] = { (void *) CFSTR(kIOHIDDeviceUsagePageKey), (void *) CFSTR(kIOHIDDeviceUsageKey) };
	const void *vals[2] = { (void *) pageNumRef, (void *) usageNumRef };

	if (pageNumRef && usageNumRef) {
		retval = CFDictionaryCreate(kCFAllocatorDefault, keys, vals, 2, &kCFTypeDictionaryKeyCallBacks, &kCFTypeDictionaryValueCallBacks);
	}

	if (pageNumRef) {
		CFRelease(pageNumRef);
	}
	if (usageNumRef) {
		CFRelease(usageNumRef);
	}

	if (!retval) {
		*okay = 0;
	}

	return retval;
}

static CFDictionaryRef
create_vendor_match(const UInt32 vendor, int *okay)
{
	CFDictionaryRef retval = NULL;
	CFNumberRef vidNumRef = CFNumberCreate(kCFAllocatorDefault, kCFNumberIntType, &vendor);
	const void *keys[1] = { (void *) CFSTR(kIOHIDVendorIDKey) };
	const void *vals[1] = { (void *) vidNumRef };

	if (vidNumRef) {
		retval = CFDictionaryCreate(kCFAllocatorDefault, keys, vals, 1, &kCFTypeDictionaryKeyCallBacks, &kCFTypeDictionaryValueCallBacks);
		CFRelease(vidNumRef);
	}

	if (!retval) {
		*okay = 0;
	}

	return retval;
}

/* Initialize the IOHIDManager. Return 0 for success and -1 for failure. */
static int init_hid_manager(void)
{
	int okay = 1;
	const void *vals[] = {
		(void *) create_usage_match(kHIDPage_GenericDesktop, kHIDUsage_GD_Joystick, &okay),
		(void *) create_usage_match(kHIDPage_GenericDesktop, kHIDUsage_GD_GamePad, &okay),
		(void *) create_usage_match(kHIDPage_GenericDesktop, kHIDUsage_GD_MultiAxisController, &okay),
		(void *) create_vendor_match(VALVE_USB_VID, &okay),
	};
	const size_t numElements = SDL_arraysize(vals);
	CFArrayRef matchingArray = okay ? CFArrayCreate(kCFAllocatorDefault, vals, numElements, &kCFTypeArrayCallBacks) : NULL;
	size_t i;

	for (i = 0; i < numElements; i++) {
		if (vals[i]) {
			CFRelease((CFTypeRef) vals[i]);
		}
	}

	/* Initialize all the HID Manager Objects */
	hid_mgr = IOHIDManagerCreate(kCFAllocatorDefault, kIOHIDOptionsTypeNone);
	if (hid_mgr) {
		IOHIDManagerSetDeviceMatchingMultiple(hid_mgr, matchingArray);
		IOHIDManagerScheduleWithRunLoop(hid_mgr, CFRunLoopGetCurrent(), kCFRunLoopDefaultMode);
		IOHIDManagerRegisterDeviceRemovalCallback(hid_mgr, hid_device_removal_callback, NULL);
	}
	
	if (matchingArray != NULL) {
		CFRelease(matchingArray);
	}

	return hid_mgr ? 0 : -1;
}

/* Initialize the IOHIDManager if necessary. This is the public function, and
 it is safe to call this function repeatedly. Return 0 for success and -1
 for failure. */
int HID_API_EXPORT hid_init(void)
{
	if (!hid_mgr) {
		return init_hid_manager();
	}
	
	/* Already initialized. */
	return 0;
}

int HID_API_EXPORT hid_exit(void)
{
	if (hid_mgr) {
		/* Close the HID manager. */
		IOHIDManagerClose(hid_mgr, kIOHIDOptionsTypeNone);
		CFRelease(hid_mgr);
		hid_mgr = NULL;
	}
	
	return 0;
}

static void process_pending_events(void) {
	SInt32 res;
	do {
		res = CFRunLoopRunInMode(kCFRunLoopDefaultMode, 0.001, FALSE);
	} while(res != kCFRunLoopRunFinished && res != kCFRunLoopRunTimedOut);
}

struct hid_device_info  HID_API_EXPORT *hid_enumerate(unsigned short vendor_id, unsigned short product_id)
{
	struct hid_device_info *root = NULL; // return object
	struct hid_device_info *cur_dev = NULL;
	CFIndex num_devices;
	CFSetRef device_set;
	IOHIDDeviceRef *device_array;
	int i;
	const char *hint = SDL_GetHint(SDL_HINT_HIDAPI_IGNORE_DEVICES);
	
	/* Set up the HID Manager if it hasn't been done */
	if (hid_init() < 0)
		return NULL;
	
	/* give the IOHIDManager a chance to update itself */
	process_pending_events();
	
	/* Get a list of the Devices */
	device_set = IOHIDManagerCopyDevices(hid_mgr);
	if (!device_set)
		return NULL;

	/* Convert the list into a C array so we can iterate easily. */	
	num_devices = CFSetGetCount(device_set);
	if (!num_devices) {
		CFRelease(device_set);
		return NULL;
	}
	device_array = (IOHIDDeviceRef*)calloc(num_devices, sizeof(IOHIDDeviceRef));
	CFSetGetValues(device_set, (const void **) device_array);
	
	/* Iterate over each device, making an entry for it. */	
	for (i = 0; i < num_devices; i++) {
		unsigned short dev_vid;
		unsigned short dev_pid;
#define BUF_LEN 256
		wchar_t buf[BUF_LEN];
		char cbuf[BUF_LEN];
		
		IOHIDDeviceRef dev = device_array[i];
		
		if (!dev) {
			continue;
		}

#if 0 // Prefer direct HID support as that has extended functionality
#if defined(SDL_JOYSTICK_MFI)
		// We want to prefer Game Controller support where available,
		// as Apple will likely be requiring that for supported devices.
		extern SDL_bool IOS_SupportedHIDDevice(IOHIDDeviceRef device);
		if (IOS_SupportedHIDDevice(dev)) {
			continue;
		}
#endif
#endif

		dev_vid = get_vendor_id(dev);
		dev_pid = get_product_id(dev);
		
		/* See if there are any devices we should skip in enumeration */
		if (hint) {
			char vendor_match[16], product_match[16];
			SDL_snprintf(vendor_match, sizeof(vendor_match), "0x%.4x/0x0000", dev_vid);
			SDL_snprintf(product_match, sizeof(product_match), "0x%.4x/0x%.4x", dev_vid, dev_pid);
			if (SDL_strcasestr(hint, vendor_match) || SDL_strcasestr(hint, product_match)) {
				continue;
			}
		}

		/* Check the VID/PID against the arguments */
		if ((vendor_id == 0x0 || dev_vid == vendor_id) &&
		    (product_id == 0x0 || dev_pid == product_id)) {
			struct hid_device_info *tmp;

			/* VID/PID match. Create the record. */
			tmp = (struct hid_device_info *)calloc(1, sizeof(struct hid_device_info));
			if (cur_dev) {
				cur_dev->next = tmp;
			}
			else {
				root = tmp;
			}
			cur_dev = tmp;
			
			// Get the Usage Page and Usage for this device.
			cur_dev->usage_page = get_int_property(dev, CFSTR(kIOHIDPrimaryUsagePageKey));
			cur_dev->usage = get_int_property(dev, CFSTR(kIOHIDPrimaryUsageKey));
			
			/* Fill out the record */
			cur_dev->next = NULL;
			make_path(dev, cbuf, sizeof(cbuf));
			cur_dev->path = strdup(cbuf);
			
			/* Serial Number */
			get_serial_number(dev, buf, BUF_LEN);
			cur_dev->serial_number = dup_wcs(buf);
			
			/* Manufacturer and Product strings */
			get_manufacturer_string(dev, buf, BUF_LEN);
			cur_dev->manufacturer_string = dup_wcs(buf);
			get_product_string(dev, buf, BUF_LEN);
			cur_dev->product_string = dup_wcs(buf);
			
			/* VID/PID */
			cur_dev->vendor_id = dev_vid;
			cur_dev->product_id = dev_pid;
			
			/* Release Number */
			cur_dev->release_number = get_int_property(dev, CFSTR(kIOHIDVersionNumberKey));
			
			/* Interface Number (Unsupported on Mac)*/
			cur_dev->interface_number = -1;
		}
	}
	
	free(device_array);
	CFRelease(device_set);
	
	return root;
}

void  HID_API_EXPORT hid_free_enumeration(struct hid_device_info *devs)
{
	/* This function is identical to the Linux version. Platform independent. */
	struct hid_device_info *d = devs;
	while (d) {
		struct hid_device_info *next = d->next;
		free(d->path);
		free(d->serial_number);
		free(d->manufacturer_string);
		free(d->product_string);
		free(d);
		d = next;
	}
}

hid_device * HID_API_EXPORT hid_open(unsigned short vendor_id, unsigned short product_id, const wchar_t *serial_number)
{
	/* This function is identical to the Linux version. Platform independent. */
	struct hid_device_info *devs, *cur_dev;
	const char *path_to_open = NULL;
	hid_device * handle = NULL;
	
	devs = hid_enumerate(vendor_id, product_id);
	cur_dev = devs;
	while (cur_dev) {
		if (cur_dev->vendor_id == vendor_id &&
		    cur_dev->product_id == product_id) {
			if (serial_number) {
				if (wcscmp(serial_number, cur_dev->serial_number) == 0) {
					path_to_open = cur_dev->path;
					break;
				}
			}
			else {
				path_to_open = cur_dev->path;
				break;
			}
		}
		cur_dev = cur_dev->next;
	}
	
	if (path_to_open) {
		/* Open the device */
		handle = hid_open_path(path_to_open, 0);
	}
	
	hid_free_enumeration(devs);
	
	return handle;
}

/* The Run Loop calls this function for each input report received.
 This function puts the data into a linked list to be picked up by
 hid_read(). */
static void hid_report_callback(void *context, IOReturn result, void *sender,
								IOHIDReportType report_type, uint32_t report_id,
								uint8_t *report, CFIndex report_length)
{
	struct input_report *rpt;
	hid_device *dev = (hid_device *)context;
	
	/* Make a new Input Report object */
	rpt = (struct input_report *)calloc(1, sizeof(struct input_report));
	rpt->data = (uint8_t *)calloc(1, report_length);
	memcpy(rpt->data, report, report_length);
	rpt->len = report_length;
	rpt->next = NULL;
	
	/* Lock this section */
	pthread_mutex_lock(&dev->mutex);
	
	/* Attach the new report object to the end of the list. */
	if (dev->input_reports == NULL) {
		/* The list is empty. Put it at the root. */
		dev->input_reports = rpt;
	}
	else {
		/* Find the end of the list and attach. */
		struct input_report *cur = dev->input_reports;
		int num_queued = 0;
		while (cur->next != NULL) {
			cur = cur->next;
			num_queued++;
		}
		cur->next = rpt;
		
		/* Pop one off if we've reached 30 in the queue. This
		 way we don't grow forever if the user never reads
		 anything from the device. */
		if (num_queued > 30) {
			return_data(dev, NULL, 0);
		}
	}
	
	/* Signal a waiting thread that there is data. */
	pthread_cond_signal(&dev->condition);
	
	/* Unlock */
	pthread_mutex_unlock(&dev->mutex);
	
}

/* This gets called when the read_thred's run loop gets signaled by
 hid_close(), and serves to stop the read_thread's run loop. */
static void perform_signal_callback(void *context)
{
	hid_device *dev = (hid_device *)context;
	CFRunLoopStop(dev->run_loop); //TODO: CFRunLoopGetCurrent()
}

static void *read_thread(void *param)
{
	hid_device *dev = (hid_device *)param;
	CFRunLoopSourceContext ctx;
    SInt32 code;
	
	/* Move the device's run loop to this thread. */
	IOHIDDeviceScheduleWithRunLoop(dev->device_handle, CFRunLoopGetCurrent(), dev->run_loop_mode);
	
	/* Create the RunLoopSource which is used to signal the
	 event loop to stop when hid_close() is called. */
	memset(&ctx, 0, sizeof(ctx));
	ctx.version = 0;
	ctx.info = dev;
	ctx.perform = &perform_signal_callback;
	dev->source = CFRunLoopSourceCreate(kCFAllocatorDefault, 0/*order*/, &ctx);
	CFRunLoopAddSource(CFRunLoopGetCurrent(), dev->source, dev->run_loop_mode);
	
	/* Store off the Run Loop so it can be stopped from hid_close()
	 and on device disconnection. */
	dev->run_loop = CFRunLoopGetCurrent();
	
	/* Notify the main thread that the read thread is up and running. */
	pthread_barrier_wait(&dev->barrier);
	
	/* Run the Event Loop. CFRunLoopRunInMode() will dispatch HID input
	 reports into the hid_report_callback(). */
	while (!dev->shutdown_thread && !dev->disconnected) {
		code = CFRunLoopRunInMode(dev->run_loop_mode, 1000/*sec*/, FALSE);
		/* Return if the device has been disconnected */
		if (code == kCFRunLoopRunFinished || code == kCFRunLoopRunStopped) {
			dev->disconnected = 1;
			break;
		}
		
		
		/* Break if The Run Loop returns Finished or Stopped. */
		if (code != kCFRunLoopRunTimedOut &&
		    code != kCFRunLoopRunHandledSource) {
			/* There was some kind of error. Setting
			 shutdown seems to make sense, but
			 there may be something else more appropriate */
			dev->shutdown_thread = 1;
			break;
		}
	}
	
	/* Now that the read thread is stopping, Wake any threads which are
	 waiting on data (in hid_read_timeout()). Do this under a mutex to
	 make sure that a thread which is about to go to sleep waiting on
	 the condition acutally will go to sleep before the condition is
	 signaled. */
	pthread_mutex_lock(&dev->mutex);
	pthread_cond_broadcast(&dev->condition);
	pthread_mutex_unlock(&dev->mutex);
	
	/* Wait here until hid_close() is called and makes it past
	 the call to CFRunLoopWakeUp(). This thread still needs to
	 be valid when that function is called on the other thread. */
	pthread_barrier_wait(&dev->shutdown_barrier);
	
	return NULL;
}

hid_device * HID_API_EXPORT hid_open_path(const char *path, int bExclusive)
{
	int i;
	hid_device *dev = NULL;
	CFIndex num_devices;
	CFSetRef device_set;
	IOHIDDeviceRef *device_array;

	dev = new_hid_device();
	
	/* Set up the HID Manager if it hasn't been done */
	if (hid_init() < 0)
		return NULL;

#if 0 /* We have a path because the IOHIDManager is already updated */
	/* give the IOHIDManager a chance to update itself */
	process_pending_events();
#endif

	device_set = IOHIDManagerCopyDevices(hid_mgr);
	if (!device_set)
		return NULL;
	
	num_devices = CFSetGetCount(device_set);
	device_array = (IOHIDDeviceRef *)calloc(num_devices, sizeof(IOHIDDeviceRef));
	CFSetGetValues(device_set, (const void **) device_array);	
	for (i = 0; i < num_devices; i++) {
		char cbuf[BUF_LEN];
		IOHIDDeviceRef os_dev = device_array[i];
		
		make_path(os_dev, cbuf, sizeof(cbuf));
		if (!strcmp(cbuf, path)) {
			// Matched Paths. Open this Device.
			IOReturn ret = IOHIDDeviceOpen(os_dev, kIOHIDOptionsTypeNone);
			if (ret == kIOReturnSuccess) {
				char str[32];
				struct hid_device_list_node *node;
				
				free(device_array);
				CFRelease(device_set);
				dev->device_handle = os_dev;
				
				/* Create the buffers for receiving data */
				dev->max_input_report_len = (CFIndex) get_max_report_length(os_dev);
				SDL_assert(dev->max_input_report_len > 0);
				dev->input_report_buf = (uint8_t *)calloc(dev->max_input_report_len, sizeof(uint8_t));
				
				/* Create the Run Loop Mode for this device.
				 printing the reference seems to work. */
				snprintf(str, sizeof(str), "HIDAPI_%p", os_dev);
				dev->run_loop_mode = 
				CFStringCreateWithCString(NULL, str, kCFStringEncodingASCII);
				
				/* Attach the device to a Run Loop */
				IOHIDDeviceRegisterInputReportCallback(
													   os_dev, dev->input_report_buf, dev->max_input_report_len,
													   &hid_report_callback, dev);

				node = (struct hid_device_list_node *)calloc(1, sizeof(struct hid_device_list_node));
				node->dev = dev;
				node->next = device_list;
				device_list = node;

				/* Start the read thread */
				pthread_create(&dev->thread, NULL, read_thread, dev);
				
				/* Wait here for the read thread to be initialized. */
				pthread_barrier_wait(&dev->barrier);
				
				return dev;
			}
			else {
				goto return_error;
			}
		}
	}
	
return_error:
	free(device_array);
	CFRelease(device_set);
	free_hid_device(dev);
	return NULL;
}

static int set_report(hid_device *dev, IOHIDReportType type, const unsigned char *data, size_t length)
{
	const char *pass_through_magic = "MAGIC0";
	size_t pass_through_magic_length = strlen(pass_through_magic);
	unsigned char report_id = data[0];
	const unsigned char *data_to_send;
	size_t length_to_send;
	IOReturn res;
	
	/* Return if the device has been disconnected. */
   	if (dev->disconnected)
   		return -1;
	
	if (report_id == 0x0) {
		/* Not using numbered Reports.
		 Don't send the report number. */
		data_to_send = data+1;
		length_to_send = length-1;
	}
	else if (length > 6 && memcmp(data, pass_through_magic, pass_through_magic_length) == 0) {
		report_id = data[pass_through_magic_length];
		data_to_send = data+pass_through_magic_length;
		length_to_send = length-pass_through_magic_length;
	}
	else {
		/* Using numbered Reports.
		 Send the Report Number */
		data_to_send = data;
		length_to_send = length;
	}
	
	if (!dev->disconnected) {
		res = IOHIDDeviceSetReport(dev->device_handle,
								   type,
								   report_id, /* Report ID*/
								   data_to_send, length_to_send);
		
		if (res == kIOReturnSuccess) {
			return (int)length;
		}
		else if (res == kIOReturnUnsupported) {
			/*printf("kIOReturnUnsupported\n");*/
			return -1;
		}
		else {
			/*printf("0x%x\n", res);*/
			return -1;
		}
	}
	
	return -1;
}

int HID_API_EXPORT hid_write(hid_device *dev, const unsigned char *data, size_t length)
{
	return set_report(dev, kIOHIDReportTypeOutput, data, length);
}

/* Helper function, so that this isn't duplicated in hid_read(). */
static int return_data(hid_device *dev, unsigned char *data, size_t length)
{
	/* Copy the data out of the linked list item (rpt) into the
	 return buffer (data), and delete the liked list item. */
	struct input_report *rpt = dev->input_reports;
	size_t len = 0;
	if (rpt != NULL) {
		len = (length < rpt->len)? length: rpt->len;
		if (data != NULL) {
			memcpy(data, rpt->data, len);
		}
		dev->input_reports = rpt->next;
		free(rpt->data);
		free(rpt);
	}
	return (int)len;
}

static int cond_wait(const hid_device *dev, pthread_cond_t *cond, pthread_mutex_t *mutex)
{
	while (!dev->input_reports) {
		int res = pthread_cond_wait(cond, mutex);
		if (res != 0)
			return res;
		
		/* A res of 0 means we may have been signaled or it may
		 be a spurious wakeup. Check to see that there's acutally
		 data in the queue before returning, and if not, go back
		 to sleep. See the pthread_cond_timedwait() man page for
		 details. */
		
		if (dev->shutdown_thread || dev->disconnected)
			return -1;
	}
	
	return 0;
}

static int cond_timedwait(const hid_device *dev, pthread_cond_t *cond, pthread_mutex_t *mutex, const struct timespec *abstime)
{
	while (!dev->input_reports) {
		int res = pthread_cond_timedwait(cond, mutex, abstime);
		if (res != 0)
			return res;
		
		/* A res of 0 means we may have been signaled or it may
		 be a spurious wakeup. Check to see that there's acutally
		 data in the queue before returning, and if not, go back
		 to sleep. See the pthread_cond_timedwait() man page for
		 details. */
		
		if (dev->shutdown_thread || dev->disconnected)
			return -1;
	}
	
	return 0;
	
}

int HID_API_EXPORT hid_read_timeout(hid_device *dev, unsigned char *data, size_t length, int milliseconds)
{
	int bytes_read = -1;
	
	/* Lock the access to the report list. */
	pthread_mutex_lock(&dev->mutex);
	
	/* There's an input report queued up. Return it. */
	if (dev->input_reports) {
		/* Return the first one */
		bytes_read = return_data(dev, data, length);
		goto ret;
	}
	
	/* Return if the device has been disconnected. */
	if (dev->disconnected) {
		bytes_read = -1;
		goto ret;
	}
	
	if (dev->shutdown_thread) {
		/* This means the device has been closed (or there
		 has been an error. An error code of -1 should
		 be returned. */
		bytes_read = -1;
		goto ret;
	}
	
	/* There is no data. Go to sleep and wait for data. */
	
	if (milliseconds == -1) {
		/* Blocking */
		int res;
		res = cond_wait(dev, &dev->condition, &dev->mutex);
		if (res == 0)
			bytes_read = return_data(dev, data, length);
		else {
			/* There was an error, or a device disconnection. */
			bytes_read = -1;
		}
	}
	else if (milliseconds > 0) {
		/* Non-blocking, but called with timeout. */
		int res;
		struct timespec ts;
		struct timeval tv;
		gettimeofday(&tv, NULL);
		TIMEVAL_TO_TIMESPEC(&tv, &ts);
		ts.tv_sec += milliseconds / 1000;
		ts.tv_nsec += (milliseconds % 1000) * 1000000;
		if (ts.tv_nsec >= 1000000000L) {
			ts.tv_sec++;
			ts.tv_nsec -= 1000000000L;
		}
		
		res = cond_timedwait(dev, &dev->condition, &dev->mutex, &ts);
		if (res == 0)
			bytes_read = return_data(dev, data, length);
		else if (res == ETIMEDOUT)
			bytes_read = 0;
		else
			bytes_read = -1;
	}
	else {
		/* Purely non-blocking */
		bytes_read = 0;
	}
	
ret:
	/* Unlock */
	pthread_mutex_unlock(&dev->mutex);
	return bytes_read;
}

int HID_API_EXPORT hid_read(hid_device *dev, unsigned char *data, size_t length)
{
	return hid_read_timeout(dev, data, length, (dev->blocking)? -1: 0);
}

int HID_API_EXPORT hid_set_nonblocking(hid_device *dev, int nonblock)
{
	/* All Nonblocking operation is handled by the library. */
	dev->blocking = !nonblock;
	
	return 0;
}

int HID_API_EXPORT hid_send_feature_report(hid_device *dev, const unsigned char *data, size_t length)
{
	return set_report(dev, kIOHIDReportTypeFeature, data, length);
}

int HID_API_EXPORT hid_get_feature_report(hid_device *dev, unsigned char *data, size_t length)
{
	CFIndex len = length;
	IOReturn res;
	int skipped_report_id = 0, report_number;

	/* Return if the device has been unplugged. */
	if (dev->disconnected)
		return -1;
	
	report_number = data[0];
	if (report_number == 0x0) {
		/* Offset the return buffer by 1, so that the report ID
		 will remain in byte 0. */
		data++;
		len--;
		skipped_report_id = 1;
	}
	
	res = IOHIDDeviceGetReport(dev->device_handle,
	                           kIOHIDReportTypeFeature,
	                           report_number, /* Report ID */
	                           data, &len);
	if (res != kIOReturnSuccess)
		return -1;

	if (skipped_report_id)
		len++;

	return (int)len;
}


void HID_API_EXPORT hid_close(hid_device *dev)
{
	if (!dev)
		return;
	
	/* Disconnect the report callback before close. */
	if (!dev->disconnected) {
		IOHIDDeviceRegisterInputReportCallback(
											   dev->device_handle, dev->input_report_buf, dev->max_input_report_len,
											   NULL, dev);
		IOHIDDeviceUnscheduleFromRunLoop(dev->device_handle, dev->run_loop, dev->run_loop_mode);
		IOHIDDeviceScheduleWithRunLoop(dev->device_handle, CFRunLoopGetMain(), kCFRunLoopDefaultMode);
	}
	
	/* Cause read_thread() to stop. */
	dev->shutdown_thread = 1;
	
	/* Wake up the run thread's event loop so that the thread can exit. */
	CFRunLoopSourceSignal(dev->source);
	CFRunLoopWakeUp(dev->run_loop);
	
	/* Notify the read thread that it can shut down now. */
	pthread_barrier_wait(&dev->shutdown_barrier);
	
	/* Wait for read_thread() to end. */
	pthread_join(dev->thread, NULL);
	
	/* Close the OS handle to the device, but only if it's not
	 been unplugged. If it's been unplugged, then calling
	 IOHIDDeviceClose() will crash. */
	if (!dev->disconnected) {
		IOHIDDeviceClose(dev->device_handle, kIOHIDOptionsTypeNone);
	}
	
	/* Clear out the queue of received reports. */
	pthread_mutex_lock(&dev->mutex);
	while (dev->input_reports) {
		return_data(dev, NULL, 0);
	}
	pthread_mutex_unlock(&dev->mutex);
	
	free_hid_device(dev);
}

int HID_API_EXPORT_CALL hid_get_manufacturer_string(hid_device *dev, wchar_t *string, size_t maxlen)
{
	return get_manufacturer_string(dev->device_handle, string, maxlen);
}

int HID_API_EXPORT_CALL hid_get_product_string(hid_device *dev, wchar_t *string, size_t maxlen)
{
	return get_product_string(dev->device_handle, string, maxlen);
}

int HID_API_EXPORT_CALL hid_get_serial_number_string(hid_device *dev, wchar_t *string, size_t maxlen)
{
	return get_serial_number(dev->device_handle, string, maxlen);
}

int HID_API_EXPORT_CALL hid_get_indexed_string(hid_device *dev, int string_index, wchar_t *string, size_t maxlen)
{
	// TODO:
	
	return 0;
}


HID_API_EXPORT const wchar_t * HID_API_CALL  hid_error(hid_device *dev)
{
	// TODO:
	
	return NULL;
}






#if 0
static int32_t get_location_id(IOHIDDeviceRef device)
{
	return get_int_property(device, CFSTR(kIOHIDLocationIDKey));
}

static int32_t get_usage(IOHIDDeviceRef device)
{
	int32_t res;
	res = get_int_property(device, CFSTR(kIOHIDDeviceUsageKey));
	if (!res)
		res = get_int_property(device, CFSTR(kIOHIDPrimaryUsageKey));
	return res;
}

static int32_t get_usage_page(IOHIDDeviceRef device)
{
	int32_t res;
	res = get_int_property(device, CFSTR(kIOHIDDeviceUsagePageKey));
	if (!res)
		res = get_int_property(device, CFSTR(kIOHIDPrimaryUsagePageKey));
	return res;
}

static int get_transport(IOHIDDeviceRef device, wchar_t *buf, size_t len)
{
	return get_string_property(device, CFSTR(kIOHIDTransportKey), buf, len);
}


int main(void)
{
	IOHIDManagerRef mgr;
	int i;
	
	mgr = IOHIDManagerCreate(kCFAllocatorDefault, kIOHIDOptionsTypeNone);
	IOHIDManagerSetDeviceMatching(mgr, NULL);
	IOHIDManagerOpen(mgr, kIOHIDOptionsTypeNone);
	
	CFSetRef device_set = IOHIDManagerCopyDevices(mgr);
	
	CFIndex num_devices = CFSetGetCount(device_set);
	IOHIDDeviceRef *device_array = calloc(num_devices, sizeof(IOHIDDeviceRef));
	CFSetGetValues(device_set, (const void **) device_array);
	
	for (i = 0; i < num_devices; i++) {
		IOHIDDeviceRef dev = device_array[i];
		printf("Device: %p\n", dev);
		printf("  %04hx %04hx\n", get_vendor_id(dev), get_product_id(dev));
		
		wchar_t serial[256], buf[256];
		char cbuf[256];
		get_serial_number(dev, serial, 256);
		
		
		printf("  Serial: %ls\n", serial);
		printf("  Loc: %ld\n", get_location_id(dev));
		get_transport(dev, buf, 256);
		printf("  Trans: %ls\n", buf);
		make_path(dev, cbuf, 256);
		printf("  Path: %s\n", cbuf);
		
	}
	
	return 0;
}
#endif
