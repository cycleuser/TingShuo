# TingShuo 听说

**Generate SRT/LRC subtitles from audio/video files using multiple speech-to-text engines, with optional LLM polishing.**

TingShuo recursively scans directories for media files, transcribes them using your choice of STT engine, and outputs subtitle files in SRT or LRC format. Optionally, subtitles can be polished using an LLM (Ollama or OpenAI-compatible API) or NLP sentence segmentation (nltk) to produce natural, complete sentences.

## Features

- **4 STT Engines**: faster-whisper, Vosk, OpenAI Whisper, whisper.cpp
- **2 Output Formats**: SRT (SubRip) and LRC (lyrics)
- **Subtitle Translation**: Translate subtitles to multiple target languages using NLLB or LLM
- **Multi-language UI**: Interface supports English, Chinese, Japanese, Korean, French, German, Spanish, Italian, Portuguese, Russian
- **LLM Polishing**: Merge fragmented subtitles into natural sentences via Ollama or OpenAI-compatible API
- **NLP Polishing**: Sentence boundary detection via nltk (no LLM required)
- **CLI + GUI**: Full command-line interface and tkinter graphical interface
- **Recursive Scanning**: Process entire directory trees of media files
- **HuggingFace Mirror**: Built-in support for HF mirror (useful in China mainland)
- **Flexible Output**: Save subtitles alongside source files or to a custom directory
- **Settings Persistence**: UI language and preferences saved to `~/.config/tingshuo/settings.json`

## Installation

### From PyPI

```bash
# Base install (no STT engine included)
pip install tingshuo

# With a specific engine:
pip install tingshuo[faster-whisper]   # Recommended
pip install tingshuo[vosk]
pip install tingshuo[whisper]
pip install tingshuo[whisper-cpp]

# With NLP polishing:
pip install tingshuo[nlp]

# Everything:
pip install tingshuo[all]
```

### From Source

```bash
git clone https://github.com/cycleuser/TingShuo.git
cd tingshuo
pip install -e .[faster-whisper,nlp]
```

## Prerequisites

- **Python 3.9+**
- **ffmpeg** must be installed and available on your PATH
  - Linux: `sudo apt install ffmpeg`
  - macOS: `brew install ffmpeg`
  - Windows: Download from [ffmpeg.org](https://ffmpeg.org/download.html) and add to PATH

## Quick Start

### CLI

**Basic transcription (SRT):**
```bash
tingshuo -i ./videos -e faster-whisper -f srt
```

**Generate LRC files to a specific output directory:**
```bash
tingshuo -i ./audio -e vosk -f lrc -o ./subtitles
```

**With LLM polishing (Ollama):**
```bash
tingshuo -i ./media --polish-llm --ollama-model qwen2.5
```

**With LLM polishing (OpenAI-compatible API):**
```bash
tingshuo -i ./media --polish-llm --api-url https://api.example.com --api-key sk-xxx --api-model gpt-4o-mini
```

**With NLP polishing:**
```bash
tingshuo -i ./media --polish-nlp -l en
```

**Specify language and model:**
```bash
tingshuo -i ./videos -e faster-whisper -m large-v3 -l zh
```

**Use HuggingFace mirror (China mainland):**
```bash
tingshuo -i ./videos -e faster-whisper --hf-mirror https://hf-mirror.com
```

**Translate subtitles to multiple languages (NLLB):**
```bash
tingshuo -i ./videos -e faster-whisper --translate --target-lang zh,ja,ko
```

**Translate subtitles using LLM:**
```bash
tingshuo -i ./videos -e faster-whisper --translate --target-lang zh --trans-backend llm --ollama-model qwen2.5
```

**Download a model before transcription:**
```bash
tingshuo --download -e faster-whisper -m large-v3
tingshuo --download -e faster-whisper -m large-v3 --hf-mirror https://hf-mirror.com
```

**Download all models for an engine:**
```bash
tingshuo --download-all -e faster-whisper
```

**List installed Ollama models:**
```bash
tingshuo --list-ollama-models
tingshuo --list-ollama-models --ollama-url http://192.168.1.100:11434
```

### GUI

```bash
tingshuo --gui
```

The GUI provides:
- Directory selection with browse buttons
- Engine and model selection dropdowns
- **Language dropdown** with common languages (auto-detect, zh, en, ja, ko, etc.) or type custom codes
- **Model download buttons** (Download / Download All) with progress feedback
- Format toggle (SRT/LRC)
- Polishing options (None / LLM / NLP) with configuration panels
- **Translation panel**: Enable translation, select target languages, choose backend (NLLB or LLM)
- **Ollama model dropdown** with Refresh button to query installed models from the server
- **Menu bar**: Help > Settings (UI language), Help > About (version info)
- **Multi-language interface**: Settings allow switching between 10 UI languages
- HuggingFace mirror toggle
- Progress bar and real-time log output
- Start/Stop controls

## CLI Reference

```
usage: tingshuo [-h] [--version] [--gui] [-i DIR] [-o DIR] [-f {srt,lrc}]
                [--no-recursive] [-e ENGINE] [-m NAME] [-l CODE]
                [--hf-mirror URL] [--download] [--download-all]
                [--list-ollama-models] [--polish-llm | --polish-nlp]
                [--ollama-url URL] [--ollama-model NAME] [--api-url URL]
                [--api-key KEY] [--api-model NAME] [-v]
                [--translate] [--target-lang CODES]
                [--trans-backend {nllb,llm}] [--nllb-model NAME]
```

### Input/Output

| Argument | Description |
|----------|-------------|
| `-i`, `--input DIR` | Input directory containing audio/video files (required) |
| `-o`, `--output DIR` | Output directory for subtitles (default: same as source) |
| `-f`, `--format {srt,lrc}` | Subtitle format (default: srt) |
| `--no-recursive` | Do not scan subdirectories |

### STT Engine

| Argument | Description |
|----------|-------------|
| `-e`, `--engine` | Engine: `faster-whisper`, `vosk`, `whisper`, `whisper-cpp` (default: faster-whisper) |
| `-m`, `--model NAME` | Model name or path (default: engine-specific, usually "base") |
| `-l`, `--language CODE` | Language code: zh, en, ja, etc. Use "auto" for auto-detection (default: auto) |

### HuggingFace Mirror

| Argument | Description |
|----------|-------------|
| `--hf-mirror URL` | HuggingFace mirror URL, e.g. `https://hf-mirror.com` |

### Model Management

| Argument | Description |
|----------|-------------|
| `--download` | Download the model specified by `-e` and `-m`, then exit |
| `--download-all` | Download all known models for the engine specified by `-e`, then exit |
| `--list-ollama-models` | List installed Ollama models from the server (uses `--ollama-url`), then exit |

### Subtitle Polishing

| Argument | Description |
|----------|-------------|
| `--polish-llm` | Polish with LLM (Ollama or OpenAI-compatible API) |
| `--polish-nlp` | Polish with NLP sentence segmentation (nltk) |
| `--ollama-url URL` | Ollama API URL (default: http://localhost:11434) |
| `--ollama-model NAME` | Ollama model name (default: qwen2.5) |
| `--api-url URL` | OpenAI-compatible API base URL |
| `--api-key KEY` | API key for OpenAI-compatible service |
| `--api-model NAME` | Model name for API |

### Other

| Argument | Description |
|----------|-------------|
| `--gui` | Launch graphical interface |
| `-v`, `--verbose` | Enable debug logging |
| `--version` | Show version and exit |

### Translation

| Argument | Description |
|----------|-------------|
| `--translate` | Enable subtitle translation to target language(s) |
| `--target-lang CODES` | Comma-separated target language codes, e.g. `zh,en,ja` |
| `--trans-backend {nllb,llm}` | Translation backend: `nllb` (Helsinki-NLP/NLLB) or `llm` (default: nllb) |
| `--nllb-model NAME` | NLLB model name (default: facebook/nllb-200-distilled-600M) |

## Supported Formats

### Input (Audio/Video)

**Audio**: mp3, wav, flac, aac, ogg, wma, m4a, opus

**Video**: mp4, mkv, avi, mov, wmv, flv, webm, ts, m4v, mpg, mpeg

### Output

**SRT** (SubRip Text):
```
1
00:00:01,500 --> 00:00:04,200
This is the first subtitle line.

2
00:00:05,000 --> 00:00:08,300
This is the second subtitle line.
```

**LRC** (Lyrics):
```
[ti:filename]
[re:TingShuo v0.1.0]

[00:01.50]This is the first subtitle line.
[00:05.00]This is the second subtitle line.
```

## STT Engines

### faster-whisper (Recommended)

CTranslate2-based Whisper implementation. Fast, supports GPU acceleration.

```bash
pip install faster-whisper
```

**Models**: tiny, base, small, medium, large-v2, large-v3

### Vosk

Lightweight offline speech recognition. Lower accuracy but very fast on CPU.

```bash
pip install vosk
```

**Models**: Downloaded automatically by language, or specify a local path with `-m /path/to/model`.

### OpenAI Whisper

The original Whisper model from OpenAI.

```bash
pip install openai-whisper
```

**Models**: tiny, base, small, medium, large

### whisper.cpp

C++ implementation of Whisper via Python bindings. Very fast on CPU.

```bash
pip install pywhispercpp
```

**Models**: tiny, base, small, medium, large

## Subtitle Polishing

### LLM Polishing

Sends subtitle segments to an LLM to merge fragments into complete, natural sentences.

**With Ollama (local):**

1. Install and start [Ollama](https://ollama.com)
2. Pull a model: `ollama pull qwen2.5`
3. Run: `tingshuo -i ./media --polish-llm --ollama-model qwen2.5`

**With Ollama (LAN):**
```bash
tingshuo -i ./media --polish-llm --ollama-url http://192.168.1.100:11434 --ollama-model qwen2.5
```

**With OpenAI-compatible API:**
```bash
tingshuo -i ./media --polish-llm --api-url https://api.openai.com --api-key sk-xxx --api-model gpt-4o-mini
```

### NLP Polishing

Uses nltk sentence tokenization to detect sentence boundaries and merge fragments. No LLM or network access required.

```bash
pip install nltk
tingshuo -i ./media --polish-nlp -l en
```

Supports English, German, French, Spanish, Italian, Portuguese, and more via nltk. For Chinese/Japanese/Korean, uses punctuation-based sentence splitting.

## Subtitle Translation

TingShuo can automatically translate generated subtitles to multiple target languages. Translated subtitles are saved as separate files with language codes (e.g., `video.zh.srt`, `video.ja.srt`).

### NLLB Translation (Recommended)

Uses Helsinki-NLP/NLLB models for high-quality offline translation supporting 200+ languages.

```bash
# Install dependencies
pip install transformers sentencepiece

# Translate to Chinese and Japanese
tingshuo -i ./videos -e faster-whisper --translate --target-lang zh,ja

# Use a larger NLLB model for better quality
tingshuo -i ./videos --translate --target-lang zh --nllb-model facebook/nllb-200-distilled-1.3B
```

Available NLLB models: `facebook/nllb-200-distilled-600M` (default), `facebook/nllb-200-distilled-1.3B`, `facebook/nllb-200-3.3B`

### LLM Translation

Uses Ollama or OpenAI-compatible API for translation.

```bash
# Translate using Ollama
tingshuo -i ./videos --translate --target-lang zh --trans-backend llm --ollama-model qwen2.5

# Translate using OpenAI API
tingshuo -i ./videos --translate --target-lang zh --trans-backend llm --api-url https://api.openai.com --api-key sk-xxx --api-model gpt-4o-mini
```

## HuggingFace Mirror

For users in China mainland who have difficulty downloading models from HuggingFace:

```bash
tingshuo -i ./videos -e faster-whisper --hf-mirror https://hf-mirror.com
```

Or set the environment variable directly:
```bash
export HF_ENDPOINT=https://hf-mirror.com
tingshuo -i ./videos -e faster-whisper
```

## License

This project is licensed under the GNU General Public License v3.0. See [LICENSE](LICENSE) for details.
