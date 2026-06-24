AC_DEFUN([AC_CHECK_DEFINE],[AC_REQUIRE([AC_PROG_CPP])dnl
  AC_CACHE_CHECK(for $1 in $2, ac_cv_define_$1,
    AC_EGREP_CPP([YES_IS_DEFINED], [
#include <$2>
#ifdef $1
YES_IS_DEFINED
#endif
    ], ac_cv_define_$1=yes, ac_cv_define_$1=no)
  )
  if test "$ac_cv_define_$1" = "yes" ; then
    AC_DEFINE([HAVE_$1],[],[Added by AC_CHECK_DEFINE])
  fi
])dnl

