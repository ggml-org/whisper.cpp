#include <ruby.h>
#include <ruby/memory_view.h>
#include <ruby/encoding.h>

typedef struct {
    VALUE audio_path;
    int   n_samples;
    const char *audio_path_str;
    float      *data;
    short      *samples;
} jfk_alloc_args;

static VALUE
jfk_reader_alloc_resources(VALUE arg)
{
    jfk_alloc_args *a = (jfk_alloc_args *)arg;
    a->audio_path_str = StringValueCStr(a->audio_path);
    a->data    = ALLOC_N(float, a->n_samples);
    a->samples = ALLOC_N(short, a->n_samples);
    return Qnil;
}

static VALUE
jfk_reader_initialize(VALUE self, VALUE audio_path)
{
  rb_iv_set(self, "audio_path", audio_path);
  return Qnil;
}

static bool
jfk_reader_get_memory_view(const VALUE obj, rb_memory_view_t *view, int flags)
{
  VALUE audio_path = rb_iv_get(obj, "audio_path");
  // n_samples is a fixed constant (not derived from user input).
  const int n_samples = 176000;

  jfk_alloc_args args = {
    .audio_path = audio_path,
    .n_samples  = n_samples,
    .audio_path_str = NULL,
    .data    = NULL,
    .samples = NULL,
  };

  int state;
  rb_protect(jfk_reader_alloc_resources, (VALUE)&args, &state);
  if (state) {
    if (args.samples) xfree(args.samples);
    if (args.data)    xfree(args.data);
    return false;
  }

  FILE *file = fopen(args.audio_path_str, "rb");
  if (file == NULL) {
    xfree(args.samples);
    xfree(args.data);
    return false;
  }

  fseek(file, 78, SEEK_SET);
  fread(args.samples, sizeof(short), n_samples, file);
  fclose(file);
  for (int i = 0; i < n_samples; i++) {
    args.data[i] = args.samples[i] / 32768.0;
  }
  xfree(args.samples);

  view->obj = obj;
  view->data = (void *)args.data;
  view->byte_size = sizeof(float) * n_samples;
  view->readonly = true;
  view->format = "f";
  view->item_size = sizeof(float);
  view->item_desc.components = NULL;
  view->item_desc.length = 0;
  view->ndim = 1;
  view->shape = NULL;
  view->sub_offsets = NULL;
  view->private_data = NULL;

  return true;
}

static bool
jfk_reader_release_memory_view(const VALUE obj, rb_memory_view_t *view)
{
  if (view->data) {
    xfree(view->data);
    view->data = NULL;
  }
  return true;
}

static bool
jfk_reader_memory_view_available_p(const VALUE obj)
{
  return true;
}

static const rb_memory_view_entry_t jfk_reader_view_entry = {
  jfk_reader_get_memory_view,
  jfk_reader_release_memory_view,
  jfk_reader_memory_view_available_p
};

void Init_jfk_reader(void)
{
  VALUE cJFKReader = rb_define_class("JFKReader", rb_cObject);
  rb_memory_view_register(cJFKReader, &jfk_reader_view_entry);
  rb_define_method(cJFKReader, "initialize", jfk_reader_initialize, 1);
}
