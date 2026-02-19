# quantize

Tool for integer quantization of Whisper `ggml` model files

## Features

- Standard uniform quantization (Q4_0, Q4_1, Q5_0, Q5_1, Q8_0, Q2_K, Q3_K, Q4_K, Q5_K, Q6_K)
- **Mixed precision quantization** - quantize different layers with different quantization types (NEW!)

## Basic Usage

```bash
./quantize model-f32.bin model-quant.bin type
```

Where `type` is one of: q4_0, q4_1, q5_0, q5_1, q8_0, q2_k, q3_k, q4_k, q5_k, q6_k

## Mixed Precision Quantization

You can now specify different quantization types for different tensors using the `--tensor-type` option:

```bash
./quantize [--tensor-type PATTERN=TYPE ...] model-f32.bin model-quant.bin default_type
```

### Examples

**Quantize encoder with Q8_0 (higher quality) and decoder with Q4_0 (smaller size):**
```bash
./quantize \
  --tensor-type 'encoder\..*\.weight'=q8_0 \
  --tensor-type 'decoder\..*\.weight'=q4_0 \
  model-f32.bin model-mixed.bin q4_k
```

**Keep attention layers at higher precision:**
```bash
./quantize \
  --tensor-type '.*attn.*'=q8_0 \
  model-f32.bin model-mixed.bin q4_0
```

For more detailed documentation and examples, see [README_MIXED_PRECISION.md](README_MIXED_PRECISION.md).
