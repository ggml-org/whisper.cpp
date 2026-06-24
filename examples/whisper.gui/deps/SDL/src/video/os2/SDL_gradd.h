/*
gradd.h structures and constants -- only the ones used by SDL_os2vman.c.

Based on public knowledge from around the internet including pages from
http://www.osfree.org and http://www.edm2.com
*/

#ifndef SDL_gradd_h_
#define SDL_gradd_h_

typedef struct _INITPROCOUT {
  ULONG     ulLength;         /*  Length of the INITPROCOUT data structure, in bytes. */
  ULONG     ulVRAMVirt;       /*  32-bit virtual address of VRAM. */
} INITPROCOUT;
typedef INITPROCOUT *PINITPROCOUT;

#define RC_SUCCESS 0

typedef ULONG GID;
typedef ULONG (_System FNVMIENTRY) (
  GID       gid, ULONG ulFunction,
  PVOID     pIn,
  PVOID     pOut /* PINITPROCOUT */
);

#define VMI_CMD_INITPROC 1
#define VMI_CMD_TERMPROC 3
#define VMI_CMD_QUERYMODES 5
#define VMI_CMD_SETMODE 6
#define VMI_CMD_PALETTE 7
#define VMI_CMD_BITBLT 8
#define VMI_CMD_LINE  9
#define VMI_CMD_REQUESTHW 14
#define VMI_CMD_QUERYCURRENTMODE 0x1001

#define QUERYMODE_NUM_MODES 0x01
#define QUERYMODE_MODE_DATA 0x02

typedef struct _HWPALETTEINFO {
  ULONG     ulLength;         /* Size of the HWPALETTEINFO data structure, in bytes. */
  ULONG     fFlags;           /* Palette flag. */
  ULONG     ulStartIndex;     /* Starting palette index. */
  ULONG     ulNumEntries;     /* Number of palette slots to query or set. */
  PRGB2     pRGBs;            /* Pointer to the array of RGB values. */
} HWPALETTEINFO;
typedef HWPALETTEINFO *PHWPALETTEINFO;

#define PALETTE_GET 0x01
#define PALETTE_SET 0x02

typedef struct _BMAPINFO {
  ULONG     ulLength;         /* Length of the BMAPINFO data structure, in bytes. */
  ULONG     ulType;           /* Description of the Blt. */
  ULONG     ulWidth;          /* Width in pels of the bit map. */
  ULONG     ulHeight;         /* Height in pels of the bit map. */
  ULONG     ulBpp;            /* Number of bits per pel/color depth. */
  ULONG     ulBytesPerLine;   /* Number of aligned bytes per line. */
  PBYTE     pBits;            /* Pointer to bit-map bits. */
} BMAPINFO;
typedef BMAPINFO *PBMAPINFO;

#define BMAP_VRAM 0
#define BMAP_MEMORY 1

typedef struct _LINEPACK {
  ULONG        ulStyleStep;   /* Value to be added to ulStyleValue. */
  ULONG        ulStyleValue;  /* Style value at the current pel. */
  ULONG        ulFlags;       /* Flags used for the LINEPACK data structure. */
  struct _LINEPACK *plpkNext; /* Pointer to next LINEPACK data structure. */
  ULONG        ulAbsDeltaX;   /* Clipped Bresenham Delta X, absolute. */
  ULONG        ulAbsDeltaY;   /* Clipped Bresenham Delta Y, absolute. */
  POINTL       ptlClipStart;  /* Pointer to location for device to perform Bresenham algorithm. */
  POINTL       ptlClipEnd;    /* Ending location for Bresenham algorithm (see ptlClipStart). */
  POINTL       ptlStart;      /* Pointer to starting location for line. */
  POINTL       ptlEnd;        /* Ending location for line. */
  LONG         lClipStartError;/* Standard Bresenham error at the clipped start point. */
} LINEPACK;
typedef LINEPACK *PLINEPACK;

typedef struct _LINEINFO {
  ULONG         ulLength;     /* Length of LINEINFO data structure. */
  ULONG         ulType;       /* Defines line type. */
  ULONG         ulStyleMask;  /* A 32-bit style mask. */
  ULONG         cLines;       /* Count of lines to be drawn. */
  ULONG         ulFGColor;    /* Line Foreground color. */
  ULONG         ulBGColor;    /* Line Background color. */
  USHORT        usForeROP;    /* Line Foreground mix. */
  USHORT        usBackROP;    /* Line Background mix. */
  PBMAPINFO     pDstBmapInfo; /* Pointer to destination surface bit map. */
  PLINEPACK     alpkLinePack; /* Pointer to LINEPACK data structure. */
  PRECTL        prclBounds;   /* Pointer to bounding rect of a clipped line. */
} LINEINFO;
typedef LINEINFO *PLINEINFO;

#define LINE_DO_FIRST_PEL 0x02
#define LINE_DIR_Y_POSITIVE 0x04
#define LINE_HORIZONTAL 0x08
#define LINE_DIR_X_POSITIVE 0x20
#define LINE_VERTICAL 0x1000
#define LINE_DO_LAST_PEL 0x4000
#define LINE_SOLID 0x01

typedef struct _BLTRECT {
  ULONG     ulXOrg;           /* X origin of the destination Blt. */
  ULONG     ulYOrg;           /* Y origin of the destination Blt. */
  ULONG     ulXExt;           /* X extent of the BitBlt. */
  ULONG     ulYExt;           /* Y extent of the BitBlt. */
} BLTRECT;
typedef BLTRECT *PBLTRECT;

typedef struct _BITBLTINFO {
  ULONG     ulLength;         /* Length of the BITBLTINFO data structure, in bytes. */
  ULONG     ulBltFlags;       /* Flags for rendering of rasterized data. */
  ULONG     cBlits;           /* Count of Blts to be performed. */
  ULONG     ulROP;            /* Raster operation. */
  ULONG     ulMonoBackROP;    /* Background mix if B_APPLY_BACK_ROP is set. */
  ULONG     ulSrcFGColor;     /* Monochrome source Foreground color. */
  ULONG     ulSrcBGColor;     /* Monochrome source Background color and transparent color. */
  ULONG     ulPatFGColor;     /* Monochrome pattern Foreground color. */
  ULONG     ulPatBGColor;     /* Monochrome pattern Background color. */
  PBYTE     abColors;         /* Pointer to color translation table. */
  PBMAPINFO pSrcBmapInfo;     /* Pointer to source bit map (BMAPINFO) */
  PBMAPINFO pDstBmapInfo;     /* Pointer to destination bit map (BMAPINFO). */
  PBMAPINFO pPatBmapInfo;     /* Pointer to pattern bit map (BMAPINFO). */
  PPOINTL   aptlSrcOrg;       /* Pointer to array of source origin POINTLs. */
  PPOINTL   aptlPatOrg;       /* Pointer to array of pattern origin POINTLs. */
  PBLTRECT  abrDst;           /* Pointer to array of Blt rects. */
  PRECTL    prclSrcBounds;    /* Pointer to source bounding rect of source Blts. */
  PRECTL    prclDstBounds;    /* Pointer to destination bounding rect of destination Blts. */
} BITBLTINFO;
typedef BITBLTINFO *PBITBLTINFO;

#define BF_DEFAULT_STATE 0x0
#define BF_ROP_INCL_SRC (0x01 << 2)
#define BF_PAT_HOLLOW   (0x01 << 8)

typedef struct _GDDMODEINFO {
  ULONG     ulLength;         /* Size of the GDDMODEINFO data structure, in bytes. */
  ULONG     ulModeId;         /* ID used to make SETMODE request. */
  ULONG     ulBpp;            /* Number of colors (bpp). */
  ULONG     ulHorizResolution;/* Number of horizontal pels. */
  ULONG     ulVertResolution; /* Number of vertical scan lines. */
  ULONG     ulRefreshRate;    /* Refresh rate in Hz. */
  PBYTE     pbVRAMPhys;       /* Physical address of VRAM. */
  ULONG     ulApertureSize;   /* Size of VRAM, in bytes. */
  ULONG     ulScanLineSize;   /* Size of one scan line, in bytes. */

  ULONG     fccColorEncoding, ulTotalVRAMSize, cColors;
} GDDMODEINFO;
typedef GDDMODEINFO *PGDDMODEINFO;

typedef struct _HWREQIN {
  ULONG     ulLength;         /* Size of the HWREQIN data structure, in bytes. */
  ULONG     ulFlags;          /* Request option flags. */
  ULONG     cScrChangeRects;  /* Count of screen rectangles affected by HWREQIN. */
  PRECTL    arectlScreen;     /* Array of screen rectangles affected by HWREQIN. */
} HWREQIN;
typedef HWREQIN *PHWREQIN;

#define REQUEST_HW 0x01

/*
BOOL GreDeath(HDC hdc, PVOID pInstance, LONG lFunction);
LONG GreResurrection(HDC hdc, LONG cbVmem, PULONG pReserved, PVOID pInstance, LONG lFunction);
*/
#define GreDeath(h) (BOOL)Gre32Entry3((ULONG)(h), 0, 0x40B7L)
#define GreResurrection(h,n,r) (LONG)Gre32Entry5((ULONG)(h), (ULONG)(n), (ULONG)(r), 0, 0x40B8L)
ULONG _System Gre32Entry3(ULONG, ULONG, ULONG);
ULONG _System Gre32Entry5(ULONG, ULONG, ULONG, ULONG, ULONG);

#endif /* SDL_gradd_h_ */
