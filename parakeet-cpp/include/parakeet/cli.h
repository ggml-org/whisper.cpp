#pragma once

// C entry parakeet_cli_main(argc, argv): same flags as the parakeet binary (--help lists them).

#include "export.h"

#ifdef __cplusplus
extern "C" {
#endif

PARAKEET_API int parakeet_cli_main(int argc, char ** argv);

#ifdef __cplusplus
}
#endif
