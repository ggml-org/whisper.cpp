/*******************************************************
 HIDAPI - Multi-Platform library for
 communication with HID devices.

 Alan Ott
 Signal 11 Software

 8/22/2009

 Copyright 2009, All Rights Reserved.
 
 At the discretion of the user of this library,
 this software may be licensed under the terms of the
 GNU General Public License v3, a BSD-Style license, or the
 original HIDAPI license as outlined in the LICENSE.txt,
 LICENSE-gpl3.txt, LICENSE-bsd.txt, and LICENSE-orig.txt
 files located at the root of the source distribution.
 These files may also be found in the public source
 code repository located at:
        https://github.com/libusb/hidapi .
********************************************************/
#include "../../SDL_internal.h"

#include "SDL_hints.h"

#include <windows.h>

#ifndef _WIN32_WINNT_WIN8
#define _WIN32_WINNT_WIN8   0x0602
#endif

#if 0 /* can cause redefinition errors on some toolchains */
#ifdef __MINGW32__
#include <ntdef.h>
#include <winbase.h>
#endif

#ifdef __CYGWIN__
#include <ntdef.h>
#define _wcsdup wcsdup
#endif
#endif /* */

#ifndef _NTDEF_
typedef LONG NTSTATUS;
#endif

/* The maximum number of characters that can be passed into the
   HidD_Get*String() functions without it failing.*/
#define MAX_STRING_WCHARS 0xFFF

/*#define HIDAPI_USE_DDK*/

/* The timeout in milliseconds for waiting on WriteFile to 
   complete in hid_write. The longest observed time to do a output
   report that we've seen is ~200-250ms so let's double that */
#define HID_WRITE_TIMEOUT_MILLISECONDS 500

/* We will only enumerate devices that match these usages */
#define USAGE_PAGE_GENERIC_DESKTOP 0x0001
#define USAGE_JOYSTICK 0x0004
#define USAGE_GAMEPAD 0x0005
#define USAGE_MULTIAXISCONTROLLER 0x0008
#define USB_VENDOR_VALVE 0x28de

#ifdef __cplusplus
extern "C" {
#endif
	#include <setupapi.h>
	#include <winioctl.h>
	#ifdef HIDAPI_USE_DDK
		#include <hidsdi.h>
	#endif

	/* Copied from inc/ddk/hidclass.h, part of the Windows DDK. */
	#define HID_OUT_CTL_CODE(id)  \
		CTL_CODE(FILE_DEVICE_KEYBOARD, (id), METHOD_OUT_DIRECT, FILE_ANY_ACCESS)
	#define IOCTL_HID_GET_FEATURE                   HID_OUT_CTL_CODE(100)

#ifdef __cplusplus
} /* extern "C" */
#endif

#include "../hidapi/hidapi.h"

/*#include <stdio.h>*/
/*#include <stdlib.h>*/

/* SDL C runtime functions */
#include "SDL_stdinc.h"

#define calloc SDL_calloc
#define free SDL_free
#define malloc SDL_malloc
#define memcpy SDL_memcpy
#define memset SDL_memset
#define strcmp SDL_strcmp
#define strlen SDL_strlen
#define strstr SDL_strstr
#define strtol SDL_strtol
#define wcscmp SDL_wcscmp
#define _wcsdup SDL_wcsdup


#undef MIN
#define MIN(x,y) ((x) < (y)? (x): (y))

#ifdef _MSC_VER
	/* Yes, we have some unreferenced formal parameters */
	#pragma warning(disable:4100)
#endif

#ifdef __cplusplus
extern "C" {
#endif

#ifndef HIDAPI_USE_DDK
	/* Since we're not building with the DDK, and the HID header
	   files aren't part of the SDK, we have to define all this
	   stuff here. In lookup_functions(), the function pointers
	   defined below are set. */
	typedef struct _HIDD_ATTRIBUTES{
		ULONG Size;
		USHORT VendorID;
		USHORT ProductID;
		USHORT VersionNumber;
	} HIDD_ATTRIBUTES, *PHIDD_ATTRIBUTES;

	typedef USHORT USAGE;
	typedef struct _HIDP_CAPS {
		USAGE Usage;
		USAGE UsagePage;
		USHORT InputReportByteLength;
		USHORT OutputReportByteLength;
		USHORT FeatureReportByteLength;
		USHORT Reserved[17];
		USHORT fields_not_used_by_hidapi[10];
	} HIDP_CAPS, *PHIDP_CAPS;
	typedef void* PHIDP_PREPARSED_DATA;
	#define HIDP_STATUS_SUCCESS 0x110000

	typedef BOOLEAN (__stdcall *HidD_GetAttributes_)(HANDLE device, PHIDD_ATTRIBUTES attrib);
	typedef BOOLEAN (__stdcall *HidD_GetSerialNumberString_)(HANDLE device, PVOID buffer, ULONG buffer_len);
	typedef BOOLEAN (__stdcall *HidD_GetManufacturerString_)(HANDLE handle, PVOID buffer, ULONG buffer_len);
	typedef BOOLEAN (__stdcall *HidD_GetProductString_)(HANDLE handle, PVOID buffer, ULONG buffer_len);
	typedef BOOLEAN (__stdcall *HidD_SetFeature_)(HANDLE handle, PVOID data, ULONG length);
	typedef BOOLEAN (__stdcall *HidD_GetFeature_)(HANDLE handle, PVOID data, ULONG length);
	typedef BOOLEAN (__stdcall *HidD_GetIndexedString_)(HANDLE handle, ULONG string_index, PVOID buffer, ULONG buffer_len);
	typedef BOOLEAN (__stdcall *HidD_GetPreparsedData_)(HANDLE handle, PHIDP_PREPARSED_DATA *preparsed_data);
	typedef BOOLEAN (__stdcall *HidD_FreePreparsedData_)(PHIDP_PREPARSED_DATA preparsed_data);
	typedef NTSTATUS (__stdcall *HidP_GetCaps_)(PHIDP_PREPARSED_DATA preparsed_data, HIDP_CAPS *caps);
	typedef BOOLEAN (__stdcall *HidD_SetNumInputBuffers_)(HANDLE handle, ULONG number_buffers);
	typedef BOOLEAN(__stdcall *HidD_SetOutputReport_ )(HANDLE handle, PVOID buffer, ULONG buffer_len);
	static HidD_GetAttributes_ HidD_GetAttributes;
	static HidD_GetSerialNumberString_ HidD_GetSerialNumberString;
	static HidD_GetManufacturerString_ HidD_GetManufacturerString;
	static HidD_GetProductString_ HidD_GetProductString;
	static HidD_SetFeature_ HidD_SetFeature;
	static HidD_GetFeature_ HidD_GetFeature;
	static HidD_GetIndexedString_ HidD_GetIndexedString;
	static HidD_GetPreparsedData_ HidD_GetPreparsedData;
	static HidD_FreePreparsedData_ HidD_FreePreparsedData;
	static HidP_GetCaps_ HidP_GetCaps;
	static HidD_SetNumInputBuffers_ HidD_SetNumInputBuffers;
	static HidD_SetOutputReport_ HidD_SetOutputReport;

	static HMODULE lib_handle = NULL;
	static BOOLEAN initialized = FALSE;
#endif /* HIDAPI_USE_DDK */

struct hid_device_ {
		HANDLE device_handle;
		BOOL blocking;
		USHORT output_report_length;
		size_t input_report_length;
		void *last_error_str;
		DWORD last_error_num;
		BOOL read_pending;
		char *read_buf;
		OVERLAPPED ol;
		OVERLAPPED write_ol;
		BOOL use_hid_write_output_report;
};

static BOOL
IsWindowsVersionOrGreater(WORD wMajorVersion, WORD wMinorVersion, WORD wServicePackMajor)
{
	OSVERSIONINFOEXW osvi;
	DWORDLONG const dwlConditionMask = VerSetConditionMask(
		VerSetConditionMask(
			VerSetConditionMask(
				0, VER_MAJORVERSION, VER_GREATER_EQUAL ),
			VER_MINORVERSION, VER_GREATER_EQUAL ),
		VER_SERVICEPACKMAJOR, VER_GREATER_EQUAL );

	memset(&osvi, 0, sizeof(osvi));
	osvi.dwOSVersionInfoSize = sizeof( osvi );
	osvi.dwMajorVersion = wMajorVersion;
	osvi.dwMinorVersion = wMinorVersion;
	osvi.wServicePackMajor = wServicePackMajor;

	return VerifyVersionInfoW(&osvi, VER_MAJORVERSION | VER_MINORVERSION | VER_SERVICEPACKMAJOR, dwlConditionMask) != FALSE;
}

static hid_device *new_hid_device()
{
	hid_device *dev = (hid_device*) calloc(1, sizeof(hid_device));
	dev->device_handle = INVALID_HANDLE_VALUE;
	dev->blocking = TRUE;
	dev->output_report_length = 0;
	dev->input_report_length = 0;
	dev->last_error_str = NULL;
	dev->last_error_num = 0;
	dev->read_pending = FALSE;
	dev->read_buf = NULL;
	memset(&dev->ol, 0, sizeof(dev->ol));
	dev->ol.hEvent = CreateEvent(NULL, FALSE, FALSE /*initial state f=nonsignaled*/, NULL);
	memset(&dev->write_ol, 0, sizeof(dev->write_ol));
	dev->write_ol.hEvent = CreateEvent(NULL, FALSE, FALSE /*initial state f=nonsignaled*/, NULL);

	return dev;
}

static void free_hid_device(hid_device *dev)
{
	CloseHandle(dev->ol.hEvent);
	CloseHandle(dev->write_ol.hEvent);
	CloseHandle(dev->device_handle);
	LocalFree(dev->last_error_str);
	free(dev->read_buf);
	free(dev);
}

static void register_error(hid_device *device, const char *op)
{
	WCHAR *ptr, *msg;

	DWORD count = FormatMessageW(FORMAT_MESSAGE_ALLOCATE_BUFFER |
		FORMAT_MESSAGE_FROM_SYSTEM |
		FORMAT_MESSAGE_IGNORE_INSERTS,
		NULL,
		GetLastError(),
		MAKELANGID(LANG_NEUTRAL, SUBLANG_DEFAULT),
		(LPWSTR)&msg, 0/*sz*/,
		NULL);
	if (!count)
		return;
	
	/* Get rid of the CR and LF that FormatMessage() sticks at the
	   end of the message. Thanks Microsoft! */
	ptr = msg;
	while (*ptr) {
		if (*ptr == '\r') {
			*ptr = 0x0000;
			break;
		}
		ptr++;
	}

	/* Store the message off in the Device entry so that
	   the hid_error() function can pick it up. */
	LocalFree(device->last_error_str);
	device->last_error_str = msg;
}

#ifndef HIDAPI_USE_DDK
static int lookup_functions(void)
{
	lib_handle = LoadLibrary(TEXT("hid.dll"));
	if (lib_handle) {
#define RESOLVE(x) x = (x##_)GetProcAddress(lib_handle, #x); if (!x) return -1;
		RESOLVE(HidD_GetAttributes);
		RESOLVE(HidD_GetSerialNumberString);
		RESOLVE(HidD_GetManufacturerString);
		RESOLVE(HidD_GetProductString);
		RESOLVE(HidD_SetFeature);
		RESOLVE(HidD_GetFeature);
		RESOLVE(HidD_GetIndexedString);
		RESOLVE(HidD_GetPreparsedData);
		RESOLVE(HidD_FreePreparsedData);
		RESOLVE(HidP_GetCaps);
		RESOLVE(HidD_SetNumInputBuffers);
		RESOLVE(HidD_SetOutputReport);
#undef RESOLVE
	}
	else
		return -1;

	return 0;
}
#endif

static HANDLE open_device(const char *path, BOOL enumerate, BOOL bExclusive )
{
	HANDLE handle;
	// Opening with access 0 causes keyboards to stop responding in some system configurations
	// http://steamcommunity.com/discussions/forum/1/1843493219428923893
	// Thanks to co-wie (Ka-wei Low <kawei@mac.com>) for help narrowing down the problem on his system
	//DWORD desired_access = (enumerate)? 0: (GENERIC_WRITE | GENERIC_READ);
	DWORD desired_access = ( GENERIC_WRITE | GENERIC_READ );
	DWORD share_mode = bExclusive ? 0 : ( FILE_SHARE_READ | FILE_SHARE_WRITE );

	handle = CreateFileA(path,
		desired_access,
		share_mode,
		NULL,
		OPEN_EXISTING,
		FILE_FLAG_OVERLAPPED,/*FILE_ATTRIBUTE_NORMAL,*/
		0);

	return handle;
}

int HID_API_EXPORT hid_init(void)
{
#ifndef HIDAPI_USE_DDK
	if (!initialized) {
		if (lookup_functions() < 0) {
			hid_exit();
			return -1;
		}
		initialized = TRUE;
	}
#endif
	return 0;
}

int HID_API_EXPORT hid_exit(void)
{
#ifndef HIDAPI_USE_DDK
	if (lib_handle)
		FreeLibrary(lib_handle);
	lib_handle = NULL;
	initialized = FALSE;
#endif
	return 0;
}

int hid_blacklist(unsigned short vendor_id, unsigned short product_id)
{
    size_t i;
    static const struct { unsigned short vid; unsigned short pid; } known_bad[] = {
        /* Causes deadlock when asking for device details... */
        { 0x1B1C, 0x1B3D },  /* Corsair Gaming keyboard */
        { 0x1532, 0x0109 },  /* Razer Lycosa Gaming keyboard */
        { 0x1532, 0x010B },  /* Razer Arctosa Gaming keyboard */
        { 0x045E, 0x0822 },  /* Microsoft Precision Mouse */
        { 0x0D8C, 0x0014 },  /* Sharkoon Skiller SGH2 headset */
        { 0x1CCF, 0x0000 },  /* All Konami Amusement Devices */

        /* Turns into an Android controller when enumerated... */
        { 0x0738, 0x2217 }   /* SPEEDLINK COMPETITION PRO */
    };

    for (i = 0; i < (sizeof(known_bad)/sizeof(known_bad[0])); i++) {
        if ((vendor_id == known_bad[i].vid) && (product_id == known_bad[i].pid || known_bad[i].pid == 0x0000)) {
            return 1;
        }
    }

    return 0;
}

struct hid_device_info HID_API_EXPORT * HID_API_CALL hid_enumerate(unsigned short vendor_id, unsigned short product_id)
{
	BOOL res;
	struct hid_device_info *root = NULL; /* return object */
	struct hid_device_info *cur_dev = NULL;

	/* Windows objects for interacting with the driver. */
	GUID InterfaceClassGuid = {0x4d1e55b2, 0xf16f, 0x11cf, {0x88, 0xcb, 0x00, 0x11, 0x11, 0x00, 0x00, 0x30} };
	SP_DEVINFO_DATA devinfo_data;
	SP_DEVICE_INTERFACE_DATA device_interface_data;
	SP_DEVICE_INTERFACE_DETAIL_DATA_A *device_interface_detail_data = NULL;
	HDEVINFO device_info_set = INVALID_HANDLE_VALUE;
	int device_index = 0;
	const char *hint = SDL_GetHint(SDL_HINT_HIDAPI_IGNORE_DEVICES);

	if (hid_init() < 0)
		return NULL;

	/* Initialize the Windows objects. */
	memset(&devinfo_data, 0x0, sizeof(devinfo_data));
	devinfo_data.cbSize = sizeof(SP_DEVINFO_DATA);
	device_interface_data.cbSize = sizeof(SP_DEVICE_INTERFACE_DATA);

	/* Get information for all the devices belonging to the HID class. */
	device_info_set = SetupDiGetClassDevsA(&InterfaceClassGuid, NULL, NULL, DIGCF_PRESENT | DIGCF_DEVICEINTERFACE);
	
	/* Iterate over each device in the HID class, looking for the right one. */
	
	for (;;) {
		HANDLE write_handle = INVALID_HANDLE_VALUE;
		DWORD required_size = 0;
		HIDD_ATTRIBUTES attrib;

		res = SetupDiEnumDeviceInterfaces(device_info_set,
			NULL,
			&InterfaceClassGuid,
			device_index,
			&device_interface_data);
		
		if (!res) {
			/* A return of FALSE from this function means that
			   there are no more devices. */
			break;
		}

		/* Call with 0-sized detail size, and let the function
		   tell us how long the detail struct needs to be. The
		   size is put in &required_size. */
		res = SetupDiGetDeviceInterfaceDetailA(device_info_set,
			&device_interface_data,
			NULL,
			0,
			&required_size,
			NULL);

		/* Allocate a long enough structure for device_interface_detail_data. */
		device_interface_detail_data = (SP_DEVICE_INTERFACE_DETAIL_DATA_A*) malloc(required_size);
		device_interface_detail_data->cbSize = sizeof(SP_DEVICE_INTERFACE_DETAIL_DATA_A);

		/* Get the detailed data for this device. The detail data gives us
		   the device path for this device, which is then passed into
		   CreateFile() to get a handle to the device. */
		res = SetupDiGetDeviceInterfaceDetailA(device_info_set,
			&device_interface_data,
			device_interface_detail_data,
			required_size,
			NULL,
			NULL);

		if (!res) {
			/* register_error(dev, "Unable to call SetupDiGetDeviceInterfaceDetail");
			   Continue to the next device. */
			goto cont;
		}

		/* XInput devices don't get real HID reports and are better handled by the raw input driver */
		if (strstr(device_interface_detail_data->DevicePath, "&ig_") != NULL) {
			goto cont;
		}

		/* Make sure this device is of Setup Class "HIDClass" and has a
		   driver bound to it. */
		/* In the main HIDAPI tree this is a loop which will erroneously open 
			devices that aren't HID class. Please preserve this delta if we ever
			update to take new changes */
		{
			char driver_name[256];

			/* Populate devinfo_data. This function will return failure
			   when there are no more interfaces left. */
			res = SetupDiEnumDeviceInfo(device_info_set, device_index, &devinfo_data);

			if (!res)
				goto cont;

			res = SetupDiGetDeviceRegistryPropertyA(device_info_set, &devinfo_data,
			               SPDRP_CLASS, NULL, (PBYTE)driver_name, sizeof(driver_name), NULL);
			if (!res)
				goto cont;

			if (strcmp(driver_name, "HIDClass") == 0) {
				/* See if there's a driver bound. */
				res = SetupDiGetDeviceRegistryPropertyA(device_info_set, &devinfo_data,
				           SPDRP_DRIVER, NULL, (PBYTE)driver_name, sizeof(driver_name), NULL);
				if (!res)
					goto cont;
			}
			else
			{
				goto cont;
			}
		}

		//wprintf(L"HandleName: %s\n", device_interface_detail_data->DevicePath);

		/* Open a handle to the device */
		write_handle = open_device(device_interface_detail_data->DevicePath, TRUE, FALSE);

		/* Check validity of write_handle. */
		if (write_handle == INVALID_HANDLE_VALUE) {
			/* Unable to open the device. */
			//register_error(dev, "CreateFile");
			goto cont;
		}		


		/* Get the Vendor ID and Product ID for this device. */
		attrib.Size = sizeof(HIDD_ATTRIBUTES);
		HidD_GetAttributes(write_handle, &attrib);
		//wprintf(L"Product/Vendor: %x %x\n", attrib.ProductID, attrib.VendorID);

		/* See if there are any devices we should skip in enumeration */
		if (hint) {
			char vendor_match[16], product_match[16];
			SDL_snprintf(vendor_match, sizeof(vendor_match), "0x%.4x/0x0000", attrib.VendorID);
			SDL_snprintf(product_match, sizeof(product_match), "0x%.4x/0x%.4x", attrib.VendorID, attrib.ProductID);
			if (SDL_strcasestr(hint, vendor_match) || SDL_strcasestr(hint, product_match)) {
				continue;
			}
		}

		/* Check the VID/PID to see if we should add this
		   device to the enumeration list. */
		if ((vendor_id == 0x0 || attrib.VendorID == vendor_id) &&
		    (product_id == 0x0 || attrib.ProductID == product_id) &&
			!hid_blacklist(attrib.VendorID, attrib.ProductID)) {

			#define WSTR_LEN 512
			const char *str;
			struct hid_device_info *tmp;
			PHIDP_PREPARSED_DATA pp_data = NULL;
			HIDP_CAPS caps;
			BOOLEAN hidp_res;
			NTSTATUS nt_res;
			wchar_t wstr[WSTR_LEN]; /* TODO: Determine Size */
			size_t len;

			/* Get the Usage Page and Usage for this device. */
			hidp_res = HidD_GetPreparsedData(write_handle, &pp_data);
			if (hidp_res) {
				nt_res = HidP_GetCaps(pp_data, &caps);
				HidD_FreePreparsedData(pp_data);
				if (nt_res != HIDP_STATUS_SUCCESS) {
					goto cont_close;
				}
			}
			else {
				goto cont_close;
			}

			/* SDL Modification: Ignore the device if it's not a gamepad. This limits compatibility
			   risk from devices that may respond poorly to our string queries below. */
			if (attrib.VendorID != USB_VENDOR_VALVE) {
				if (caps.UsagePage != USAGE_PAGE_GENERIC_DESKTOP) {
					goto cont_close;
				}
				if (caps.Usage != USAGE_JOYSTICK && caps.Usage != USAGE_GAMEPAD && caps.Usage != USAGE_MULTIAXISCONTROLLER) {
					goto cont_close;
				}
			}

			/* VID/PID match. Create the record. */
			tmp = (struct hid_device_info*) calloc(1, sizeof(struct hid_device_info));
			if (cur_dev) {
				cur_dev->next = tmp;
			}
			else {
				root = tmp;
			}
			cur_dev = tmp;
			
			/* Fill out the record */
			cur_dev->usage_page = caps.UsagePage;
			cur_dev->usage = caps.Usage;
			cur_dev->next = NULL;
			str = device_interface_detail_data->DevicePath;
			if (str) {
				len = strlen(str);
				cur_dev->path = (char*) calloc(len+1, sizeof(char));
				memcpy(cur_dev->path, str, len+1);
			}
			else
				cur_dev->path = NULL;

			/* Serial Number */
			hidp_res = HidD_GetSerialNumberString(write_handle, wstr, sizeof(wstr));
			wstr[WSTR_LEN-1] = 0x0000;
			if (hidp_res) {
				cur_dev->serial_number = _wcsdup(wstr);
			}

			/* Manufacturer String */
			hidp_res = HidD_GetManufacturerString(write_handle, wstr, sizeof(wstr));
			wstr[WSTR_LEN-1] = 0x0000;
			if (hidp_res) {
				cur_dev->manufacturer_string = _wcsdup(wstr);
			}

			/* Product String */
			hidp_res = HidD_GetProductString(write_handle, wstr, sizeof(wstr));
			wstr[WSTR_LEN-1] = 0x0000;
			if (hidp_res) {
				cur_dev->product_string = _wcsdup(wstr);
			}

			/* VID/PID */
			cur_dev->vendor_id = attrib.VendorID;
			cur_dev->product_id = attrib.ProductID;

			/* Release Number */
			cur_dev->release_number = attrib.VersionNumber;

			/* Interface Number. It can sometimes be parsed out of the path
			   on Windows if a device has multiple interfaces. See
			   http://msdn.microsoft.com/en-us/windows/hardware/gg487473 or
			   search for "Hardware IDs for HID Devices" at MSDN. If it's not
			   in the path, it's set to -1. */
			cur_dev->interface_number = -1;
			if (cur_dev->path) {
				char *interface_component = strstr(cur_dev->path, "&mi_");
				if (interface_component) {
					char *hex_str = interface_component + 4;
					char *endptr = NULL;
					cur_dev->interface_number = strtol(hex_str, &endptr, 16);
					if (endptr == hex_str) {
						/* The parsing failed. Set interface_number to -1. */
						cur_dev->interface_number = -1;
					}
				}
			}
		}

cont_close:
		CloseHandle(write_handle);
cont:
		/* We no longer need the detail data. It can be freed */
		free(device_interface_detail_data);

		device_index++;

	}

	/* Close the device information handle. */
	SetupDiDestroyDeviceInfoList(device_info_set);

	return root;

}

void  HID_API_EXPORT HID_API_CALL hid_free_enumeration(struct hid_device_info *devs)
{
	/* TODO: Merge this with the Linux version. This function is platform-independent. */
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


HID_API_EXPORT hid_device * HID_API_CALL hid_open(unsigned short vendor_id, unsigned short product_id, const wchar_t *serial_number)
{
	/* TODO: Merge this functions with the Linux version. This function should be platform independent. */
	struct hid_device_info *devs, *cur_dev;
	const char *path_to_open = NULL;
	hid_device *handle = NULL;
	
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

HID_API_EXPORT hid_device * HID_API_CALL hid_open_path(const char *path, int bExclusive)
{
	hid_device *dev;
	HIDP_CAPS caps;
	PHIDP_PREPARSED_DATA pp_data = NULL;
	BOOLEAN res;
	NTSTATUS nt_res;

	if (hid_init() < 0) {
		return NULL;
	}

	dev = new_hid_device();

	/* Open a handle to the device */
	dev->device_handle = open_device(path, FALSE, bExclusive);

	/* Check validity of write_handle. */
	if (dev->device_handle == INVALID_HANDLE_VALUE) {
		/* Unable to open the device. */
		register_error(dev, "CreateFile");
		goto err;
	}

	/* Set the Input Report buffer size to 64 reports. */
	res = HidD_SetNumInputBuffers(dev->device_handle, 64);
	if (!res) {
		register_error(dev, "HidD_SetNumInputBuffers");
		goto err;
	}

	/* Get the Input Report length for the device. */
	res = HidD_GetPreparsedData(dev->device_handle, &pp_data);
	if (!res) {
		register_error(dev, "HidD_GetPreparsedData");
		goto err;
	}
	nt_res = HidP_GetCaps(pp_data, &caps);
	if (nt_res != HIDP_STATUS_SUCCESS) {
		register_error(dev, "HidP_GetCaps");	
		goto err_pp_data;
	}
	dev->output_report_length = caps.OutputReportByteLength;
	dev->input_report_length = caps.InputReportByteLength;
	HidD_FreePreparsedData(pp_data);

	/* On Windows 7, we need to use hid_write_output_report() over Bluetooth */
	if (dev->output_report_length > 512) {
		dev->use_hid_write_output_report = !IsWindowsVersionOrGreater( HIBYTE( _WIN32_WINNT_WIN8 ), LOBYTE( _WIN32_WINNT_WIN8 ), 0 );
	}

	dev->read_buf = (char*) malloc(dev->input_report_length);

	return dev;

err_pp_data:
		HidD_FreePreparsedData(pp_data);
err:	
		free_hid_device(dev);
		return NULL;
}

int HID_API_EXPORT HID_API_CALL hid_write_output_report(hid_device *dev, const unsigned char *data, size_t length)
{
	BOOL res;
	res = HidD_SetOutputReport(dev->device_handle, (void *)data, (ULONG)length);
	if (res)
		return (int)length;
	else
		return -1;
}

static int hid_write_timeout(hid_device *dev, const unsigned char *data, size_t length, int milliseconds)
{
	DWORD bytes_written;
	BOOL res;
	unsigned char *buf;

	if (dev->use_hid_write_output_report) {
		return hid_write_output_report(dev, data, length);
	}

	/* Make sure the right number of bytes are passed to WriteFile. Windows
	   expects the number of bytes which are in the _longest_ report (plus
	   one for the report number) bytes even if the data is a report
	   which is shorter than that. Windows gives us this value in
	   caps.OutputReportByteLength. If a user passes in fewer bytes than this,
	   create a temporary buffer which is the proper size. */
	if (length >= dev->output_report_length) {
		/* The user passed the right number of bytes. Use the buffer as-is. */
		buf = (unsigned char *) data;
	} else {
		/* Create a temporary buffer and copy the user's data
		   into it, padding the rest with zeros. */
		buf = (unsigned char *) malloc(dev->output_report_length);
		memcpy(buf, data, length);
		memset(buf + length, 0, dev->output_report_length - length);
		length = dev->output_report_length;
	}

	res = WriteFile( dev->device_handle, buf, ( DWORD ) length, NULL, &dev->write_ol );
	if (!res) {
		if (GetLastError() != ERROR_IO_PENDING) {
			/* WriteFile() failed. Return error. */
			register_error(dev, "WriteFile");
			bytes_written = (DWORD) -1;
			goto end_of_function;
		}
	}

	/* Wait here until the write is done. This makes hid_write() synchronous. */
	res = WaitForSingleObject(dev->write_ol.hEvent, milliseconds);
	if (res != WAIT_OBJECT_0)
	{
		// There was a Timeout.
		bytes_written = (DWORD) -1;
		register_error(dev, "WriteFile/WaitForSingleObject Timeout");
		goto end_of_function;
	}

	res = GetOverlappedResult(dev->device_handle, &dev->write_ol, &bytes_written, FALSE/*F=don't_wait*/);
	if (!res) {
		/* The Write operation failed. */
		register_error(dev, "WriteFile");
		bytes_written = (DWORD) -1;
		goto end_of_function;
	}

end_of_function:
	if (buf != data)
		free(buf);

	return bytes_written;
}

int HID_API_EXPORT HID_API_CALL hid_write(hid_device *dev, const unsigned char *data, size_t length)
{
	return hid_write_timeout(dev, data, length, HID_WRITE_TIMEOUT_MILLISECONDS);
}

int HID_API_EXPORT HID_API_CALL hid_read_timeout(hid_device *dev, unsigned char *data, size_t length, int milliseconds)
{
	DWORD bytes_read = 0;
	size_t copy_len = 0;
	BOOL res;

	/* Copy the handle for convenience. */
	HANDLE ev = dev->ol.hEvent;

	if (!dev->read_pending) {
		/* Start an Overlapped I/O read. */
		dev->read_pending = TRUE;
		memset(dev->read_buf, 0, dev->input_report_length);
		ResetEvent(ev);
		res = ReadFile(dev->device_handle, dev->read_buf, (DWORD)dev->input_report_length, &bytes_read, &dev->ol);
		
		if (!res) {
			if (GetLastError() != ERROR_IO_PENDING) {
				/* ReadFile() has failed.
				   Clean up and return error. */
				CancelIo(dev->device_handle);
				dev->read_pending = FALSE;
				goto end_of_function;
			}
		}
	}

	/* See if there is any data yet. */
	res = WaitForSingleObject(ev, milliseconds >= 0 ? milliseconds : INFINITE);
	if (res != WAIT_OBJECT_0) {
		/* There was no data this time. Return zero bytes available,
			but leave the Overlapped I/O running. */
		return 0;
	}

	/* Get the number of bytes read. The actual data has been copied to the data[]
	   array which was passed to ReadFile(). We must not wait here because we've
	   already waited on our event above, and since it's auto-reset, it will have
	   been reset back to unsignalled by now. */
	res = GetOverlappedResult(dev->device_handle, &dev->ol, &bytes_read, FALSE/*don't wait*/);
	
	/* Set pending back to false, even if GetOverlappedResult() returned error. */
	dev->read_pending = FALSE;

	if (res && bytes_read > 0) {
		if (dev->read_buf[0] == 0x0) {
			/* If report numbers aren't being used, but Windows sticks a report
			   number (0x0) on the beginning of the report anyway. To make this
			   work like the other platforms, and to make it work more like the
			   HID spec, we'll skip over this byte. */
			bytes_read--;
			copy_len = length > bytes_read ? bytes_read : length;
			memcpy(data, dev->read_buf+1, copy_len);
		}
		else {
			/* Copy the whole buffer, report number and all. */
			copy_len = length > bytes_read ? bytes_read : length;
			memcpy(data, dev->read_buf, copy_len);
		}
	}
	
end_of_function:
	if (!res) {
		register_error(dev, "GetOverlappedResult");
		return -1;
	}
	
	return (int)copy_len;
}

int HID_API_EXPORT HID_API_CALL hid_read(hid_device *dev, unsigned char *data, size_t length)
{
	return hid_read_timeout(dev, data, length, (dev->blocking)? -1: 0);
}

int HID_API_EXPORT HID_API_CALL hid_set_nonblocking(hid_device *dev, int nonblock)
{
	dev->blocking = !nonblock;
	return 0; /* Success */
}

int HID_API_EXPORT HID_API_CALL hid_send_feature_report(hid_device *dev, const unsigned char *data, size_t length)
{
	BOOL res = HidD_SetFeature(dev->device_handle, (PVOID)data, (ULONG)length);
	if (!res) {
		register_error(dev, "HidD_SetFeature");
		return -1;
	}

	return (int)length;
}


int HID_API_EXPORT HID_API_CALL hid_get_feature_report(hid_device *dev, unsigned char *data, size_t length)
{
	BOOL res;
#if 0
	res = HidD_GetFeature(dev->device_handle, (PVOID)data, (ULONG)length);
	if (!res) {
		register_error(dev, "HidD_GetFeature");
		return -1;
	}
	return 0; /* HidD_GetFeature() doesn't give us an actual length, unfortunately */
#else
	DWORD bytes_returned;

	OVERLAPPED ol;
	memset(&ol, 0, sizeof(ol));

	res = DeviceIoControl(dev->device_handle,
		IOCTL_HID_GET_FEATURE,
		data, (DWORD)length,
		data, (DWORD)length,
		&bytes_returned, &ol);

	if (!res) {
		if (GetLastError() != ERROR_IO_PENDING) {
			/* DeviceIoControl() failed. Return error. */
			register_error(dev, "Send Feature Report DeviceIoControl");
			return -1;
		}
	}

	/* Wait here until the write is done. This makes
	   hid_get_feature_report() synchronous. */
	res = GetOverlappedResult(dev->device_handle, &ol, &bytes_returned, TRUE/*wait*/);
	if (!res) {
		/* The operation failed. */
		register_error(dev, "Send Feature Report GetOverLappedResult");
		return -1;
	}

	return bytes_returned;
#endif
}

void HID_API_EXPORT HID_API_CALL hid_close(hid_device *dev)
{
	typedef BOOL (WINAPI *CancelIoEx_t)(HANDLE hFile, LPOVERLAPPED lpOverlapped);
	CancelIoEx_t CancelIoExFunc = (CancelIoEx_t)GetProcAddress(GetModuleHandle(TEXT("kernel32.dll")), "CancelIoEx");

	if (!dev)
		return;

	if (CancelIoExFunc) {
		CancelIoExFunc(dev->device_handle, NULL);
	} else {
		/* Windows XP, this will only cancel I/O on the current thread */
		CancelIo(dev->device_handle);
	}
	if (dev->read_pending) {
		DWORD bytes_read = 0;

		GetOverlappedResult(dev->device_handle, &dev->ol, &bytes_read, TRUE/*wait*/);
	}
	free_hid_device(dev);
}

int HID_API_EXPORT_CALL HID_API_CALL hid_get_manufacturer_string(hid_device *dev, wchar_t *string, size_t maxlen)
{
	BOOL res;

	res = HidD_GetManufacturerString(dev->device_handle, string, (ULONG)(sizeof(wchar_t) * MIN(maxlen, MAX_STRING_WCHARS)));
	if (!res) {
		register_error(dev, "HidD_GetManufacturerString");
		return -1;
	}

	return 0;
}

int HID_API_EXPORT_CALL HID_API_CALL hid_get_product_string(hid_device *dev, wchar_t *string, size_t maxlen)
{
	BOOL res;

	res = HidD_GetProductString(dev->device_handle, string, (ULONG)(sizeof(wchar_t) * MIN(maxlen, MAX_STRING_WCHARS)));
	if (!res) {
		register_error(dev, "HidD_GetProductString");
		return -1;
	}

	return 0;
}

int HID_API_EXPORT_CALL HID_API_CALL hid_get_serial_number_string(hid_device *dev, wchar_t *string, size_t maxlen)
{
	BOOL res;

	res = HidD_GetSerialNumberString(dev->device_handle, string, (ULONG)(sizeof(wchar_t) * MIN(maxlen, MAX_STRING_WCHARS)));
	if (!res) {
		register_error(dev, "HidD_GetSerialNumberString");
		return -1;
	}

	return 0;
}

int HID_API_EXPORT_CALL HID_API_CALL hid_get_indexed_string(hid_device *dev, int string_index, wchar_t *string, size_t maxlen)
{
	BOOL res;

	res = HidD_GetIndexedString(dev->device_handle, string_index, string, (ULONG)(sizeof(wchar_t) * MIN(maxlen, MAX_STRING_WCHARS)));
	if (!res) {
		register_error(dev, "HidD_GetIndexedString");
		return -1;
	}

	return 0;
}

HID_API_EXPORT const wchar_t * HID_API_CALL  hid_error(hid_device *dev)
{
	return (wchar_t*)dev->last_error_str;
}


#if 0

/*#define PICPGM*/
/*#define S11*/
#define P32
#ifdef S11
	unsigned short VendorID = 0xa0a0;
	unsigned short ProductID = 0x0001;
#endif

#ifdef P32
	unsigned short VendorID = 0x04d8;
	unsigned short ProductID = 0x3f;
#endif

#ifdef PICPGM
	unsigned short VendorID = 0x04d8;
	unsigned short ProductID = 0x0033;
#endif

int __cdecl main(int argc, char* argv[])
{
	int i, res;
	unsigned char buf[65];

	UNREFERENCED_PARAMETER(argc);
	UNREFERENCED_PARAMETER(argv);

	/* Set up the command buffer. */
	memset(buf,0x00,sizeof(buf));
	buf[0] = 0;
	buf[1] = 0x81;
	

	/* Open the device. */
	int handle = open(VendorID, ProductID, L"12345");
	if (handle < 0)
		printf("unable to open device\n");


	/* Toggle LED (cmd 0x80) */
	buf[1] = 0x80;
	res = write(handle, buf, 65);
	if (res < 0)
		printf("Unable to write()\n");

	/* Request state (cmd 0x81) */
	buf[1] = 0x81;
	write(handle, buf, 65);
	if (res < 0)
		printf("Unable to write() (2)\n");

	/* Read requested state */
	read(handle, buf, 65);
	if (res < 0)
		printf("Unable to read()\n");

	/* Print out the returned buffer. */
	for (i = 0; i < 4; i++)
		printf("buf[%d]: %d\n", i, buf[i]);

	return 0;
}
#endif

#ifdef __cplusplus
} /* extern "C" */
#endif
