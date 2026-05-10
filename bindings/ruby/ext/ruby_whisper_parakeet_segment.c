#include "ruby_whisper.h"

void
init_ruby_whisper_parakeet_segment(VALUE *mParakeet)
{
  rb_define_class_under(*mParakeet, "Segment", rb_cObject);
}
