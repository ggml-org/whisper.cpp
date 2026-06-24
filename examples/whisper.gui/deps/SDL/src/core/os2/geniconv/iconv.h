#ifndef ICONV_H_ /* minimal iconv.h header based on public knowledge */
#define ICONV_H_

#include <stddef.h> /* size_t */
#include <errno.h>

typedef void *iconv_t;

#ifdef __cplusplus
extern "C" {
#endif

extern iconv_t iconv_open(const char *, const char *);
extern size_t iconv(iconv_t, char **, size_t *, char **, size_t *);
extern int iconv_close(iconv_t);

#ifdef __cplusplus
}
#endif

#endif /* ICONV_H_ */
