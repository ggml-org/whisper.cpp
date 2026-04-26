# pyannote-audio 導入手順

whisper.cpp（音声認識）と pyannote-audio（話者識別）を組み合わせた、  
**日本語会議録音の話者別文字起こしパイプライン**の構築手順です。

## 前提条件

- OS: Windows 11
- Anaconda / Miniconda がインストール済みであること
- Git for Windows がインストール済みであること
- リポジトリ: `c:\work\30.Projects\102.AI_Projects\whisper.cpp\whisper.cpp`

---

## ステップ 1: whisper.cpp のビルド

```bash
cd c:\work\30.Projects\102.AI_Projects\whisper.cpp\whisper.cpp

cmake -B build
cmake --build build -j --config Release
```

ビルド完了後、以下のバイナリが生成されます：

```
build/bin/Release/whisper-cli.exe
```

---

## ステップ 2: Python 仮想環境の作成

> **注意:** Python 3.13 は pyannote.audio 非対応のため、**3.11 を使用すること**。

```bash
conda create -n whisper-diarize python=3.11 -y
```

---

## ステップ 3: パッケージのインストール

`--index-url` の適用範囲の問題により、**2段階でインストール**する必要があります。

### Step A: PyTorch（CPU専用ビルド）

```bash
conda run -n whisper-diarize pip install torch torchaudio --index-url https://download.pytorch.org/whl/cpu
```

### Step B: pyannote.audio および関連パッケージ

```bash
conda run -n whisper-diarize pip install -r examples/python/requirements-diarize.txt
```

`requirements-diarize.txt` の内容（`examples/python/requirements-diarize.txt`）：

```
# Speaker diarization
pyannote.audio>=3.1.0

# Hugging Face model hub access
huggingface_hub>=0.20.0

# Audio reading（Windows で torchcodec/FFmpeg 不要にするための回避策）
soundfile>=0.12.0
```

> **Windows の注意事項:**  
> pyannote.audio 4.x は音声読み込みに `torchcodec`（FFmpeg 必須）を使用しますが、  
> conda-forge の FFmpeg は Windows 日本語環境でインストールエラーになる場合があります。  
> 代わりに `soundfile` でWAVを読み込み、テンソルとして直接 pyannote に渡す方式を採用しています。

### インストール確認

```bash
C:/work/60.Tools/Anaconda/miniconda3/envs/whisper-diarize/python.exe -c ^
  "import torch, pyannote.audio, soundfile; ^
   print('torch:', torch.__version__); ^
   print('pyannote:', pyannote.audio.__version__); ^
   print('soundfile:', soundfile.__version__); ^
   print('CPU only:', not torch.cuda.is_available())"
```

期待される出力：

```
torch: 2.11.0+cpu
pyannote.audio: 4.0.4
soundfile: 0.13.1
CPU only: True
```

---

## ステップ 4: HuggingFace トークンの取得と利用規約への同意

pyannote のモデルはゲート付きリポジトリのため、以下の手順が**全て必要**です。

### 4-1. HuggingFace アカウント作成

https://huggingface.co でサインアップ（既存アカウントがあればスキップ）。

### 4-2. 利用規約への同意（3つ全て必要）

以下の各ページにアクセスし、**"Agree and access repository"** をクリックする。

| モデル | URL |
|--------|-----|
| speaker-diarization-3.1 | https://huggingface.co/pyannote/speaker-diarization-3.1 |
| segmentation-3.0 | https://huggingface.co/pyannote/segmentation-3.0 |
| speaker-diarization-community-1 | https://huggingface.co/pyannote/speaker-diarization-community-1 |

> **注意:** 3つ目の `speaker-diarization-community-1` は pyannote.audio 4.x から追加された依存リポジトリです。  
> 同意しないと実行時に `403 Forbidden` エラーが発生します。

### 4-3. アクセストークンの発行

1. https://huggingface.co/settings/tokens にアクセス
2. "New token" をクリック
3. 権限: `read` を選択して作成
4. `hf_` で始まるトークン文字列を控えておく

---

## ステップ 5: 日本語モデルのダウンロード

```bash
cd c:\work\30.Projects\102.AI_Projects\whisper.cpp\whisper.cpp

# 精度重視（推奨、2.9GB）
bash models/download-ggml-model.sh large-v3

# 速度重視（動作確認用、142MB）
bash models/download-ggml-model.sh base
```

| モデル | サイズ | 日本語精度 |
|--------|--------|------------|
| large-v3 | 2.9 GB | 最高 |
| medium | 1.5 GB | 高 |
| small | 466 MB | 中 |
| base | 142 MB | 低（動作確認用途） |

---

## ステップ 6: 音声ファイルの準備

whisper.cpp は **16kHz / 16bit / モノラル WAV** のみ対応。  
他の形式の場合は ffmpeg で変換する。

```bash
ffmpeg -i 会議録音.mp4 -ar 16000 -ac 1 -c:a pcm_s16le 会議録音.wav
```

---

## ステップ 7: 実行

スクリプト: `examples/python/diarize.py`

```bash
cd c:\work\30.Projects\102.AI_Projects\whisper.cpp\whisper.cpp

C:/work/60.Tools/Anaconda/miniconda3/envs/whisper-diarize/python.exe \
  examples/python/diarize.py \
  -f 会議録音.wav \
  -m large-v3 \
  --hf-token hf_xxxxxxxxxxxx \
  --language ja
```

### オプション一覧

| オプション | 省略形 | デフォルト | 説明 |
|---|---|---|---|
| `--file` | `-f` | （必須） | 入力WAVファイル |
| `--model` | `-m` | `large-v3` | ggmlモデル名 |
| `--language` | `-l` | `ja` | 言語コード |
| `--hf-token` | | （必須） | HuggingFace トークン |
| `--num-speakers` | | 自動検出 | 参加者数（既知の場合に指定すると精度向上） |
| `--threads` | `-t` | `4` | whisper-cli スレッド数 |
| `--output-json` | | OFF | JSON形式で出力 |

### 出力例

```
[00:00:00 --> 00:00:03]  SPEAKER_00: 本日はお集まりいただきありがとうございます。
[00:00:03 --> 00:00:07]  SPEAKER_01: よろしくお願いします。
[00:00:07 --> 00:00:12]  SPEAKER_00: では、議題に入りましょう。
[00:00:12 --> 00:00:18]  SPEAKER_02: 先週の進捗を報告します。
```

---

## トラブルシューティング

### `torchcodec` の警告が大量に出る

```
UserWarning: torchcodec is not installed correctly so built-in audio decoding will fail.
```

**→ 無視して問題ありません。**  
`soundfile` による回避策が有効になっているため、実際の動作に影響しません。

### `403 Forbidden` / `GatedRepoError`

pyannote モデルへのアクセス権がありません。  
**→ ステップ 4-2 の3リポジトリ全てに同意済みか確認してください。**

### `TypeError: Pipeline.from_pretrained() got an unexpected keyword argument 'use_auth_token'`

pyannote.audio 4.x で `use_auth_token` 引数が廃止されました。  
**→ `token=` に変更してください（`diarize.py` は修正済み）。**

### `AttributeError: 'DiarizeOutput' object has no attribute 'itertracks'`

pyannote.audio 4.x で出力型が `DiarizeOutput` に変更されました。  
**→ `result.speaker_diarization.itertracks()` を使用してください（`diarize.py` は修正済み）。**

---

## 環境情報（動作確認済み）

| 項目 | バージョン |
|---|---|
| OS | Windows 11 Pro 10.0.26200 |
| Python | 3.11 (conda) |
| torch | 2.11.0+cpu |
| pyannote.audio | 4.0.4 |
| soundfile | 0.13.1 |
| whisper.cpp | v1.8.4 (master) |
| whisper-cliモデル | ggml-base.bin / ggml-large-v3.bin |
