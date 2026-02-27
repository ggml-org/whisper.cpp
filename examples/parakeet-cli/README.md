# whisper.cpp/examples/parakeet-cli

This is an example of using the [Parakeet] model in whisper.cpp.

### Download converted model
```console
$ hf download danbev/parakeet parakeet-tdt-0.6b-v3.bin --local-dir models
```

### Building
```console
$ cmake -B build -S .
$ cmake --build build --target parakeet-cli -j 12
```

### Usage
```console
$ ./build/bin/parakeet-cli --help

usage: ./build/bin/parakeet-cli [options] file0 file1 ...
supported audio formats: flac, mp3, ogg, wav

options:
  -h,     --help              [default] show this help message and exit
  -t N,   --threads N         [4      ] number of threads to use during computation
  -cl N,  --chunk-length N    [10000  ] chunk length in milliseconds
  -lc N,  --left-context N    [10000  ] left context in milliseconds
  -rc N,  --right-context N   [4960   ] right context in milliseconds
  -m,     --model FILE        [models/ggml-parakeet-tdt-0.6b-v3.bin] model path
  -f,     --file FILE         [       ] input audio file
  -ng,    --no-gpu            [false  ] disable GPU
  -dev N, --device N          [0      ] GPU device to use
  -fa,    --flash-attn        [true   ] enable flash attention
  -nfa,   --no-flash-attn     [false  ] disable flash attention
  -ps,    --print-segments    [false  ] print segment information
```

### Example
```console
$ ./build/bin/parakeet-cli -m models/parakeet-tdt-0.6b-v3.bin -f samples/jfk.wav
Processing audio (176000 samples, 11.00 seconds)
Processing audio: total_frames=1101, chunk_size=1101
parakeet_decode: starting decode with n_frames=138
And so, my fellow Americans, ask not what your country can do for you, ask what you can do for your country.
```

To print segment information:
```console
$ ./build/bin/parakeet-cli -m models/parakeet-tdt-0.6b-v3.bin -f samples/jfk.wav --print-segments
Processing audio (176000 samples, 11.00 seconds)
Processing audio: total_frames=1101, chunk_size=1101
parakeet_decode: starting decode with n_frames=138
And so, my fellow Americans, ask not what your country can do for you, ask what you can do for your country.

Segments (1):
Segment 0: [0 -> 1101] "And so, my fellow Americans, ask not what your country can do for you, ask what you can do for your country."
Tokens [38]:
  [ 0] id= 1976 frame=  3 dur_idx= 4 dur_val= 4 p=0.9996 plog=-15.6206 t0=  24 t1=  56 word_start=true "▁And"
  [ 1] id=  547 frame=  7 dur_idx= 4 dur_val= 4 p=0.9999 plog=-18.7922 t0=  56 t1=  88 word_start=true "▁so"
  [ 2] id= 7877 frame= 11 dur_idx= 2 dur_val= 2 p=0.8451 plog=-14.5929 t0=  88 t1=  88 word_start=false ","
  [ 3] id= 1103 frame= 13 dur_idx= 3 dur_val= 3 p=0.9996 plog=-15.6127 t0= 104 t1= 128 word_start=true "▁my"
  [ 4] id=  309 frame= 16 dur_idx= 1 dur_val= 1 p=0.9912 plog=-11.9635 t0= 128 t1= 136 word_start=true "▁f"
  [ 5] id=  530 frame= 17 dur_idx= 2 dur_val= 2 p=1.0000 plog=-13.5239 t0= 136 t1= 152 word_start=false "ell"
  [ 6] id=  596 frame= 19 dur_idx= 3 dur_val= 3 p=1.0000 plog=-16.3120 t0= 152 t1= 176 word_start=false "ow"
  [ 7] id= 3213 frame= 22 dur_idx= 4 dur_val= 4 p=0.9999 plog=-10.1462 t0= 176 t1= 208 word_start=true "▁Amer"
  [ 8] id=  404 frame= 26 dur_idx= 4 dur_val= 4 p=1.0000 plog=-25.0910 t0= 208 t1= 240 word_start=false "ic"
  [ 9] id=  667 frame= 30 dur_idx= 4 dur_val= 4 p=1.0000 plog=-27.1707 t0= 240 t1= 272 word_start=false "ans"
  [10] id= 7877 frame= 37 dur_idx= 4 dur_val= 4 p=0.9094 plog=-16.3405 t0= 272 t1= 272 word_start=false ","
  [11] id=  279 frame= 41 dur_idx= 4 dur_val= 4 p=0.9980 plog=-19.7244 t0= 328 t1= 360 word_start=true "▁a"
  [12] id=  583 frame= 45 dur_idx= 4 dur_val= 4 p=1.0000 plog=-24.5312 t0= 360 t1= 392 word_start=false "sk"
  [13] id= 1491 frame= 53 dur_idx= 4 dur_val= 4 p=1.0000 plog=-23.2991 t0= 424 t1= 456 word_start=true "▁not"
  [14] id= 3470 frame= 65 dur_idx= 4 dur_val= 4 p=0.9995 plog=-16.7306 t0= 520 t1= 552 word_start=true "▁what"
  [15] id= 3629 frame= 69 dur_idx= 2 dur_val= 2 p=0.8139 plog=-11.6486 t0= 552 t1= 568 word_start=true "▁your"
  [16] id=  867 frame= 75 dur_idx= 1 dur_val= 1 p=0.9980 plog=-12.5265 t0= 600 t1= 608 word_start=true "▁co"
  [17] id=  331 frame= 76 dur_idx= 2 dur_val= 2 p=1.0000 plog=-11.6697 t0= 608 t1= 624 word_start=false "un"
  [18] id=  958 frame= 78 dur_idx= 2 dur_val= 2 p=1.0000 plog=-11.3621 t0= 624 t1= 640 word_start=false "tr"
  [19] id= 7893 frame= 80 dur_idx= 2 dur_val= 2 p=1.0000 plog=-14.3245 t0= 640 t1= 656 word_start=false "y"
  [20] id= 2059 frame= 82 dur_idx= 3 dur_val= 3 p=1.0000 plog=-17.7694 t0= 656 t1= 680 word_start=true "▁can"
  [21] id=  458 frame= 85 dur_idx= 4 dur_val= 4 p=1.0000 plog=-23.2510 t0= 680 t1= 712 word_start=true "▁do"
  [22] id=  509 frame= 89 dur_idx= 4 dur_val= 4 p=1.0000 plog=-23.0688 t0= 712 t1= 744 word_start=true "▁for"
  [23] id= 1180 frame= 93 dur_idx= 4 dur_val= 4 p=0.9999 plog=-25.0567 t0= 744 t1= 776 word_start=true "▁you"
  [24] id= 7877 frame= 98 dur_idx= 4 dur_val= 4 p=0.8820 plog=-14.2549 t0= 776 t1= 776 word_start=false ","
  [25] id=  279 frame=102 dur_idx= 3 dur_val= 3 p=0.9992 plog=-16.8176 t0= 816 t1= 840 word_start=true "▁a"
  [26] id=  583 frame=105 dur_idx= 4 dur_val= 4 p=1.0000 plog=-21.0352 t0= 840 t1= 872 word_start=false "sk"
  [27] id= 3470 frame=109 dur_idx= 3 dur_val= 3 p=0.9999 plog=-15.4659 t0= 872 t1= 896 word_start=true "▁what"
  [28] id= 1180 frame=112 dur_idx= 4 dur_val= 4 p=0.9997 plog=-17.6392 t0= 896 t1= 928 word_start=true "▁you"
  [29] id= 2059 frame=116 dur_idx= 3 dur_val= 3 p=0.9999 plog=-15.5484 t0= 928 t1= 952 word_start=true "▁can"
  [30] id=  458 frame=119 dur_idx= 2 dur_val= 2 p=1.0000 plog=-15.9953 t0= 952 t1= 968 word_start=true "▁do"
  [31] id=  509 frame=121 dur_idx= 3 dur_val= 3 p=1.0000 plog=-15.9605 t0= 968 t1= 992 word_start=true "▁for"
  [32] id= 3629 frame=124 dur_idx= 2 dur_val= 2 p=0.9994 plog=-12.2083 t0= 992 t1=1008 word_start=true "▁your"
  [33] id=  867 frame=126 dur_idx= 2 dur_val= 2 p=0.9969 plog=-9.1252 t0=1008 t1=1024 word_start=true "▁co"
  [34] id=  331 frame=128 dur_idx= 1 dur_val= 1 p=0.9999 plog=-12.6911 t0=1024 t1=1032 word_start=false "un"
  [35] id=  958 frame=129 dur_idx= 1 dur_val= 1 p=1.0000 plog=-8.8885 t0=1032 t1=1040 word_start=false "tr"
  [36] id= 7893 frame=130 dur_idx= 2 dur_val= 2 p=1.0000 plog=-14.1441 t0=1040 t1=1056 word_start=false "y"
  [37] id= 7883 frame=132 dur_idx= 4 dur_val= 4 p=0.9567 plog=-11.5227 t0=1056 t1=1056 word_start=false "."
```

### Model conversion
Clone the original model from Hugging Face:
```console
$ git clone https://huggingface.co/nvidia/parakeet-tdt-0.6b-v3
```
Convert the model:
```console
(venv) $ python models/convert-parakeet-to-ggml.py \
    --model <path to cloned model> \
    --use-f32 \
    --out-dir models \
    --out-name ggml-parakeet-tdt-0.6b-v3.bin
```

[Parakeet]: https://huggingface.co/nvidia/parakeet-tdt-0.6b-v3
