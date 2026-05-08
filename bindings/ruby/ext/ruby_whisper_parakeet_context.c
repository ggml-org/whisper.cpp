#include "ruby_whisper.h"



void
init_ruby_whisper_parakeet_context(VALUE *mParakeet)
{
  VALUE cParakeetContext = rb_define_class_under(*mParakeet, "Context", rb_cObject);
}
