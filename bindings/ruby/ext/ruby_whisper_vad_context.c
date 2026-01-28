#include "ruby_whisper.h"

extern ID id_to_s;

extern VALUE cVADContext;

extern const rb_data_type_t ruby_whisper_vad_params_type;
extern VALUE ruby_whisper_vad_detect(VALUE self, VALUE file_path, VALUE params);
extern VALUE ruby_whisper_normalize_model_path(VALUE model_path);
extern full_parsed_args parse_full_args(int argc, VALUE *argv);
extern void release_samples(struct full_parsed_args *parsed);

extern VALUE ruby_whisper_vad_segments_s_init(struct whisper_vad_segments *segments);

static size_t
ruby_whisper_vad_context_memsize(const void *p)
{
  const ruby_whisper_vad_context *rwvc = p;
  size_t size = sizeof(rwvc);
  if (!rwvc) {
    return 0;
  }
  if (rwvc->context) {
    size += sizeof(rwvc->context);
  }
  return size;
}

static void
ruby_whisper_vad_context_free(void *p)
{
  ruby_whisper_vad_context *rwvc = (ruby_whisper_vad_context *)p;
  if (rwvc->context) {
    whisper_vad_free(rwvc->context);
    rwvc->context = NULL;
  }
  xfree(rwvc);
}

const rb_data_type_t ruby_whisper_vad_context_type = {
  "ruby_whisper_vad_context",
  {0, ruby_whisper_vad_context_free, ruby_whisper_vad_context_memsize,},
  0, 0,
  0
};

static VALUE
ruby_whisper_vad_context_s_allocate(VALUE klass)
{
  ruby_whisper_vad_context *rwvc;
  VALUE obj = TypedData_Make_Struct(klass, ruby_whisper_vad_context, &ruby_whisper_vad_context_type, rwvc);
  rwvc->context = NULL;
  return obj;
}

static VALUE
ruby_whisper_vad_context_initialize(VALUE self, VALUE model_path)
{
  ruby_whisper_vad_context *rwvc;
  struct whisper_vad_context *context;

  model_path = ruby_whisper_normalize_model_path(model_path);
  context = whisper_vad_init_from_file_with_params(StringValueCStr(model_path), whisper_vad_default_context_params());
  if (context == NULL) {
    rb_raise(rb_eRuntimeError, "Failed to initialize whisper VAD context");
  }
  TypedData_Get_Struct(self, ruby_whisper_vad_context, &ruby_whisper_vad_context_type, rwvc);
  rwvc->context = context;

  return Qnil;
}

static VALUE
ruby_whisper_vad_segments_from_samples(int argc, VALUE *argv, VALUE self)
{
  ruby_whisper_vad_context *rwvc;
  ruby_whisper_vad_params *rwvp;
  GetVADContext(self, rwvc);
  VALUE params = argv[0];
  GetVADParams(params, rwvp);

  struct full_parsed_args parsed = parse_full_args(argc, argv);
  struct whisper_vad_segments *segments = whisper_vad_segments_from_samples(
    rwvc->context,
    rwvp->params,
    parsed.samples,
    parsed.n_samples
  );
  release_samples(&parsed);

  return ruby_whisper_vad_segments_s_init(segments);
}

void init_ruby_whisper_vad_context(VALUE *mVAD)
{
  cVADContext = rb_define_class_under(*mVAD, "Context", rb_cObject);
  rb_define_alloc_func(cVADContext, ruby_whisper_vad_context_s_allocate);
  rb_define_method(cVADContext, "initialize", ruby_whisper_vad_context_initialize, 1);
  rb_define_method(cVADContext, "segments_from_samples", ruby_whisper_vad_segments_from_samples, -1);
  rb_define_method(cVADContext, "detect", ruby_whisper_vad_detect, 2);
}
