// Regression for oversized setup allocations in the bundled Vorbis decoder.
// Only uses local synthetic allocation requests; no audio files or model needed.
#include <limits.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define STB_VORBIS_NO_STDIO
#define STB_VORBIS_NO_PUSHDATA_API
#include "../examples/stb_vorbis.c"

int main(void) {
   vorb state;
   unsigned char buffer[32];
   void *p;

   memset(&state, 0, sizeof(state));

#if SIZE_MAX > UINT_MAX
   // Originally size_t -> int narrowing allocated a tiny buffer here.
   if ((p = setup_malloc(&state, (size_t) UINT_MAX + 12)) != NULL) {
      free(p);
      fprintf(stderr, "setup_malloc accepted a truncated size\n");
      return 1;
   }
   if ((p = setup_temp_malloc(&state, (size_t) UINT_MAX + 12)) != NULL) {
      free(p);
      fprintf(stderr, "setup_temp_malloc accepted a truncated size\n");
      return 2;
   }
#endif

   // A valid, small allocation must still succeed.
   p = setup_malloc(&state, 16);
   if (!p) return 3;
   setup_free(&state, p);

   // Fixed buffer allocations must not cross the temporary region.
   memset(&state, 0, sizeof(state));
   state.alloc.alloc_buffer = (char *) buffer;
   state.alloc.alloc_buffer_length_in_bytes = sizeof(buffer);
   state.temp_offset = sizeof(buffer);
   if (setup_malloc(&state, 16) != buffer) return 4;
   if (setup_temp_malloc(&state, 16) != buffer + 16) return 5;
   if (setup_malloc(&state, 1) != NULL) return 6;
   if (setup_temp_malloc(&state, 1) != NULL) return 7;

   // Powers beyond INT_MAX must not be converted to int in lookup1_values.
   if (lookup1_values(16519105, 65) != 1) return 8;
   if (lookup1_values(8, 3) != 2) return 9;
   return 0;
}
